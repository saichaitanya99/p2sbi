"""
channel_grid_report.py
----------------------
Decide whether several stations (or epochs) can share ONE trained model.

Stations with the same nominal band still differ in their exact per-channel
centre frequencies. This tool takes a set of archives, derives a nominal grid,
and computes -- for each archive -- the DM error that results from training on
the nominal grid and inferring on that archive's true grid. That number, against
your DM errorbar, is what decides the question.

    conda run -n penv_psr python -m p2sbi.channel_grid_report archives/*.ar \
        --config configs/german_J2145.yaml

    # or, without psrchive, from saved .pt files written by real_archive_loader
    conda run -n penv python -m p2sbi.channel_grid_report real_data/real_*.pt

Outputs
-------
  * per-archive channel-grid deviation from the nominal grid
  * the induced fractional DM scale error, and its absolute size at the DM
    prior edge, per archive
  * a verdict on sharing one model
  * a ready-to-paste `channel_freqs_mhz:` block for the nominal grid

The physics
-----------
PSRCHIVE dedisperses at the header DM using the archive's TRUE channel
frequencies, so the residual sweep in the portrait is

    s_i = dDM_true * u_i ,      u_i = 1 / nu_i,true^2

A network trained on the nominal grid reads that sweep against

    x_i = 1 / nu_i,nom^2

and the constant term is absorbed by the ToA parameter, so the recovered dDM is
the least-squares projection of u onto x:

    dDM_fit / dDM_true = Cov(u, x) / Var(x)

The error is therefore a SCALE error, proportional to dDM -- worst at the prior
edge, zero at dDM = 0. There is no constant offset.
"""

import argparse
import os
import sys

import numpy as np

from .pulsar_config import get_config, resolve_config_path


def channel_freqs_from_archive(path):
    """True per-channel centre frequencies, ascending."""
    import psrchive
    arch = psrchive.Archive_load(path)
    arch.tscrunch()
    arch.pscrunch()
    sub = arch.get_Integration(0)
    f = np.array([sub.get_centre_frequency(i) for i in range(arch.get_nchan())],
                 dtype=float)
    if f.size > 1 and f[0] > f[-1]:
        f = f[::-1].copy()
    return f, arch.get_nbin()


def channel_freqs_from_pt(path):
    """Channel frequencies recorded by real_archive_loader.py, if present."""
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d.get("metadata", {})
    f = meta.get("channel_freqs_mhz")
    if f is None:
        raise KeyError(
            f"{os.path.basename(path)} has no metadata['channel_freqs_mhz'] — it "
            f"predates that field. Re-run real_archive_loader.py, or pass the .ar "
            f"archives directly."
        )
    return np.asarray(f, dtype=float), int(d["portrait"].shape[-1])


def _per_channel_crb(cfg, freqs, snrs=(7, 30, 120)):
    """Cramer-Rao bound on each channel's pulse position, in phase bins.

    The best any estimator — including the network — can do at a given S/N.
    Returns {snr: array(n_chan)}.
    """
    from .pulsarDMvary import PulsarProfileSimulator, build_components

    sim = PulsarProfileSimulator(
        n_phase_bins=cfg.n_bins, n_freq_channels=len(freqs),
        freq_low_mhz=float(freqs[0]), freq_high_mhz=float(freqs[-1]),
        reference_freq_mhz=cfg.ref_freq_mhz, period_ms=cfg.period_ms,
        freqs_mhz=freqs,
    )
    rng_state = np.random.get_state()
    np.random.seed(0)
    s = cfg.sample()
    s[cfg.dm_param] = 0.0
    if "toa" in s:
        s["toa"] = 0.0
    np.random.set_state(rng_state)

    clean = sim.generate_portrait(
        build_components(cfg, s), epoch_mjd=cfg.epoch_mjd, dm=0.0,
        tau_scatter_ms=0.0, scattering_index=-4.4, dm_smearing=False,
        add_noise=False, snr=1e6,
    )["portrait"]

    A = clean.mean(axis=0).max()
    out = {}
    for snr in snrs:
        sigma_pix = (A / snr) * np.sqrt(len(freqs))     # 'avg_peak' noise model
        out[snr] = np.array([
            1.0 / np.sqrt(np.sum(np.gradient(clean[i]) ** 2) / sigma_pix ** 2)
            for i in range(len(freqs))
        ])
    return out


def dm_scale_error(nu_true, nu_nom):
    """Fractional dDM scale error from assuming nu_nom when truth is nu_true.

    Returns (frac_error, per_channel_bin_offset_at_unit_dm).
    """
    u = 1.0 / nu_true ** 2
    x = 1.0 / nu_nom ** 2
    xc = x - x.mean()
    uc = u - u.mean()
    slope = float(np.dot(uc, xc) / np.dot(xc, xc))   # dDM_fit / dDM_true
    return slope - 1.0, u - x


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="PSRCHIVE .ar archives or .pt files. Prefix with "
                         "'LABEL=' to group them (e.g. DE601=a.ar DE601=b.ar "
                         "DE603=c.ar); with two or more epochs per label the "
                         "report decomposes the grid scatter into between-station "
                         "and within-station parts, which is what decides whether "
                         "per-station channel lists are worth having.")
    ap.add_argument("--config", default=None,
                    help="Config supplying the period, DM prior and (if set) the "
                         "current nominal channel grid to compare against")
    ap.add_argument("--nominal", choices=["mean", "config"], default="mean",
                    help="Nominal grid: 'mean' = average of the inputs (default), "
                         "'config' = the config's channel_freqs_mhz")
    ap.add_argument("--dm_errorbar", type=float, default=1e-4,
                    help="Your achieved epoch-wise DM errorbar in pc/cm^3, used for "
                         "the verdict (default 1e-4)")
    args = ap.parse_args()

    cfg = get_config(resolve_config_path(args.config))
    P0_ms, N_BINS = cfg.period_ms, cfg.n_bins
    K_DM = 4.1488064239e6      # MHz^2 ms / (pc cm^-3), as in _apply_dispersion
    dm_edge = max(abs(p.low) for p in cfg.inferred if p.name == cfg.dm_param)

    # ---------- load every grid ----------
    grids, names, labels = [], [], []
    for spec in args.inputs:
        label, _, p = spec.partition("=") if "=" in spec else ("", "", spec)
        try:
            if p.lower().endswith(".pt"):
                f, nb = channel_freqs_from_pt(p)
            else:
                f, nb = channel_freqs_from_archive(p)
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(p)}: {e}")
            continue
        grids.append(f)
        names.append(os.path.basename(p))
        labels.append(label or "(ungrouped)")

    if not grids:
        sys.exit("No channel grids could be read.")

    n_ch = {g.size for g in grids}
    if len(n_ch) > 1:
        print(f"ERROR: inputs have different channel counts {sorted(n_ch)}. "
              f"They cannot share a model at all — the feature width differs.")
        sys.exit(1)
    n_ch = n_ch.pop()

    G = np.vstack(grids)   # (n_archive, n_chan)

    print("=" * 72)
    print(f"  Channel-grid comparison — {len(grids)} inputs, {n_ch} channels each")
    print("=" * 72)
    print(f"  Config      : {os.path.basename(cfg.path)}  ({cfg.name})")
    print(f"  Period      : {P0_ms:.9f} ms   bins: {N_BINS}")
    print(f"  DM prior edge: {dm_edge:g} pc/cm^3")
    print(f"  Your DM errorbar: {args.dm_errorbar:g} pc/cm^3")

    # ---------- nominal grid ----------
    if args.nominal == "config":
        if not cfg.has_explicit_freqs:
            sys.exit("--nominal config requires instrument.channel_freqs_mhz in the config")
        nominal = cfg.freqs_mhz
        if nominal.size != n_ch:
            sys.exit(f"config grid has {nominal.size} channels, inputs have {n_ch}")
        src = f"config {os.path.basename(cfg.path)}"
    else:
        nominal = G.mean(axis=0)
        src = f"mean of the {len(grids)} inputs"

    print(f"  Nominal grid: {src}")

    # ---------- spread across inputs ----------
    spread = G.std(axis=0)
    dev    = G - nominal
    print(f"\n  Per-channel spread across inputs: "
          f"max {spread.max():.4f} MHz, mean {spread.mean():.4f} MHz")
    print(f"  Deviation from nominal:           "
          f"max |{np.abs(dev).max():.4f}| MHz")

    # ---------- the number that decides it ----------
    # sweep in bins per unit dDM, for scaling the per-channel offsets
    bins_per_dm = K_DM / P0_ms * N_BINS

    print(f"\n  {'input':<44} {'max|dnu|':>9} {'dDM scale':>10} {'|dDM| @edge':>12} {'vs errbar':>10}")
    print(f"  {'-'*44} {'-'*9} {'-'*10} {'-'*12} {'-'*10}")
    worst = 0.0
    rows = []
    for nm, g in zip(names, grids):
        frac, du = dm_scale_error(g, nominal)
        abs_err = abs(frac) * dm_edge
        ratio   = abs_err / args.dm_errorbar
        worst   = max(worst, ratio)
        rows.append((nm, frac, abs_err, ratio))
        print(f"  {nm[:44]:<44} {np.abs(g-nominal).max():>9.4f} "
              f"{frac:>+10.2e} {abs_err:>12.2e} {ratio:>9.2f}x")

    # ---------- the part that is NOT a DM error ----------
    #
    # Splitting the frequency mismatch into two pieces matters, because they do
    # different damage:
    #
    #   * the component parallel to 1/nu^2 is degenerate with dDM, so it is a
    #     pure scale error on the inferred DM — the column above;
    #   * the orthogonal remainder is NOT absorbable by any DM. It distorts the
    #     shape of the sweep into something the network never saw in training.
    #     It does not bias DM in a simple way; it pushes the observation out of
    #     the training distribution, which typically shows up as an inflated or
    #     displaced posterior.
    #
    # Scale: the full-band sweep spans `sweep_bins` bins per unit dDM, so the
    # orthogonal residual is quoted both in bins and as a fraction of the sweep
    # actually available at the prior edge.
    sweep_span = (1.0 / nominal.min() ** 2 - 1.0 / nominal.max() ** 2)
    sweep_bins = K_DM * sweep_span / P0_ms * N_BINS       # bins per pc/cm^3
    print(f"\n  Sweep lever arm: {sweep_bins:.0f} bins per pc/cm^3 "
          f"({dm_edge*sweep_bins:.1f} bins at the prior edge)")

    # The orthogonal residual does NOT bias DM — it is orthogonal to the DM
    # basis by construction. Quoting it as a "DM equivalent" against a DM
    # errorbar (which an earlier version of this tool did) compares two
    # different things and is alarmist: it reported 0.69x the DM errorbar for
    # an effect that cannot move DM at all.
    #
    # What it really is: a per-channel distortion of the sweep SHAPE, i.e. an
    # out-of-distribution perturbation. So the honest yardstick is the noise on
    # the per-channel pulse position — if the distortion is below that, no
    # estimator can see it. The bound is the Cramer-Rao limit for a shift,
    # sigma_i = (sum (dp/dphi)^2 / sigma_pix^2)^-1/2, computed on a noiseless
    # portrait from this config, and the pattern's detectability is the matched
    # filter combination sqrt(sum (r_i/sigma_i)^2) over all channels.
    crb = _per_channel_crb(cfg, nominal)

    print(f"\n  Orthogonal (non-DM) sweep distortion at dDM = {dm_edge:g},")
    print(f"  against the per-channel position noise at this config's S/N prior:")
    print(f"  {'input':<34} {'max bins':>9} {'rms bins':>9}" +
          "".join(f"{('S/N=%g' % s):>10}" for s in sorted(crb)))
    print(f"  {'-'*34} {'-'*9} {'-'*9}" + "".join(f" {'-'*9}" for _ in crb))
    worst_orth = 0.0
    for nm, g in zip(names, grids):
        frac, du = dm_scale_error(g, nominal)
        x  = 1.0 / nominal ** 2
        xc = x - x.mean()
        r  = (du - du.mean()) - frac * xc      # strip the dDM-degenerate part
        r_bins = r * dm_edge * K_DM / P0_ms * N_BINS
        cols = ""
        for s in sorted(crb):
            det = float(np.sqrt(np.sum((r_bins / crb[s]) ** 2)))
            worst_orth = max(worst_orth, det)
            cols += f"{det:>10.2f}"
        print(f"  {nm[:34]:<34} {np.abs(r_bins).max():>9.3f} "
              f"{r_bins.std():>9.3f}{cols}")
    print(f"\n  The S/N columns are the pattern's detectability in sigma:")
    print(f"  sqrt(sum (r_i/sigma_i)^2) over channels, sigma_i = Cramer-Rao bound")
    print(f"  on that channel's pulse position. Below ~1 the mismatch is buried in")
    print(f"  the noise and cannot affect inference at all.")

    # ---------- would PER-STATION grids actually help? ----------
    #
    # Only if the scatter BETWEEN stations is larger than the scatter WITHIN a
    # station across epochs. A per-station channel list can remove the former;
    # nothing in the config can remove the latter, because the config is fixed
    # at training time while the epochs keep moving.
    #
    # So the comparison that decides it is, for each archive:
    #   (a) distortion against the GLOBAL nominal grid   -> one shared model
    #   (b) distortion against ITS OWN STATION's mean    -> per-station models
    # If (b) is not much smaller than (a), per-station lists buy nothing.
    uniq = sorted(set(labels))
    if len(uniq) > 1 and any(labels.count(u) > 1 for u in uniq):
        print(f"\n{'-'*72}")
        print(f"  DOES A PER-STATION GRID HELP?  ({len(uniq)} groups)")
        print(f"{'-'*72}")

        group_mean = {u: G[[i for i, l in enumerate(labels) if l == u]].mean(axis=0)
                      for u in uniq}

        # Per-channel variance decomposition
        means = np.vstack([group_mean[u] for u in uniq])
        between = means.std(axis=0)
        within  = np.array([
            G[[i for i, l in enumerate(labels) if l == u]].std(axis=0)
            for u in uniq
        ]).mean(axis=0)
        print(f"  per-channel scatter:  between stations {between.mean():.4f} MHz "
              f"(max {between.max():.4f})")
        print(f"                        within  stations {within.mean():.4f} MHz "
              f"(max {within.max():.4f})")

        def _both_dm(true_g, nom_g):
            """(DM bias, non-DM distortion), both in pc/cm^3 at the prior edge.

            BOTH terms are needed. A smooth/constant frequency offset is almost
            entirely DM-degenerate: it produces a large DM SCALE error and
            almost no orthogonal distortion. Random per-channel scatter does
            the reverse. Judging on the orthogonal term alone therefore misses
            precisely the effect a per-station grid would remove — a constant
            +0.25 MHz offset gives DM scale error -5.1e-3 (1.3e-4 pc/cm^3 at
            the prior edge) with an orthogonal residual of only 1e-8.
            """
            frac, du = dm_scale_error(true_g, nom_g)
            x  = 1.0 / nom_g ** 2
            xc = x - x.mean()
            r  = (du - du.mean()) - frac * xc
            orth_bins = np.abs(r).max() * dm_edge * K_DM / P0_ms * N_BINS
            return abs(frac) * dm_edge, orth_bins / sweep_bins

        print(f"\n  All values are pc/cm^3 at the DM prior edge.")
        print(f"  {'input':<26} {'stn':<7} {'--- vs global ---':>21} "
              f"{'-- vs own station --':>21} {'gain':>6}")
        print(f"  {'':<26} {'':<7} {'DM bias':>10} {'non-DM':>10} "
              f"{'DM bias':>10} {'non-DM':>10} {'':>6}")
        print(f"  {'-'*26} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
        gains = []
        for nm, lab, g in zip(names, labels, grids):
            a_dm, a_or = _both_dm(g, nominal)
            b_dm, b_or = _both_dm(g, group_mean[lab])
            a_tot, b_tot = max(a_dm, a_or), max(b_dm, b_or)
            gn = b_tot / a_tot if a_tot > 0 else 1.0
            gains.append(gn)
            print(f"  {nm[:26]:<26} {lab[:7]:<7} {a_dm:>10.2e} {a_or:>10.2e} "
                  f"{b_dm:>10.2e} {b_or:>10.2e} {gn:>5.2f}x")

        med_gain = float(np.median(gains))
        # Which term drives the shared-grid error, so the reason given is right
        _adm = np.median([_both_dm(g, nominal)[0] for g in grids])
        _aor = np.median([_both_dm(g, nominal)[1] for g in grids])
        _drv = "DM bias (smooth offsets)" if _adm >= _aor else \
               "non-DM distortion (random scatter)"
        print(f"\n  Shared-grid error is dominated by: {_drv}")
        print(f"  Median error after switching to per-station grids: "
              f"{med_gain:.2f}x the shared-grid error")

        if med_gain > 0.7:
            print(f"\n  [NO]   per-station channel lists remove only "
                  f"{100*(1-med_gain):.0f}% of the error.")
            if _aor > _adm:
                print(f"         The mismatch is random per-channel scatter, which is")
                print(f"         irreducible by ANY fixed grid: it varies epoch to")
                print(f"         epoch, so a per-station list cannot capture it.")
            else:
                print(f"         The between-station part is smooth and therefore")
                print(f"         largely absorbed by the DM scale anyway.")
            print(f"         Six grids would cost six datasets and six models for")
            print(f"         almost nothing. Use ONE nominal grid.")
        elif med_gain > 0.3:
            print(f"\n  [MAYBE] per-station grids remove {100*(1-med_gain):.0f}% of the")
            print(f"         error. Worth it only if the shared-grid verdict above is")
            print(f"         FAIL and you would rather pay 6x than implement")
            print(f"         per-observation frequencies (Option B).")
        else:
            print(f"\n  [YES]  per-station grids remove {100*(1-med_gain):.0f}% of the "
                  f"error — the stations differ")
            print(f"         systematically and a shared nominal grid gives each one a")
            print(f"         FIXED bias that will not average down over epochs.")
            print(f"         Either give each station its own channel_freqs_mhz (six")
            print(f"         datasets, six models), or condition one model on the")
            print(f"         per-observation grid — Option B, which also handles the")
            print(f"         epoch-to-epoch residual that per-station grids cannot.")
    elif len(uniq) > 1:
        print(f"\n  [NOTE] {len(uniq)} groups given but only one epoch each, so the")
        print(f"         between/within-station split cannot be measured. Pass two or")
        print(f"         more epochs per station to find out whether per-station")
        print(f"         channel lists would help.")
    else:
        print(f"\n  [NOTE] inputs are ungrouped. Label them 'STATION=path' (two or")
        print(f"         more epochs each) to measure whether per-station channel")
        print(f"         lists would help, or whether epoch scatter dominates.")

    # ---------- verdict ----------
    print("\n" + "=" * 72)
    print(f"  TWO SEPARATE QUESTIONS, TWO SEPARATE ANSWERS")
    print("=" * 72)
    print(f"  1. Does the grid mismatch BIAS DM?")
    print(f"     Worst DM bias {worst:.2f}x your {args.dm_errorbar:g} errorbar.")
    if worst < 0.3:
        print(f"     [OK]   no. The bias is a scale error proportional to dDM and")
        print(f"            this small even at the prior edge; at a typical dDM it is")
        print(f"            smaller still.")
    elif worst < 1.0:
        print(f"     [WARN] a per-station systematic of this size, fixed per station,")
        print(f"            so it will not average down across epochs.")
    else:
        print(f"     [FAIL] yes — too large to share one nominal grid.")

    print(f"\n  2. Does it push observations OUT OF the training distribution?")
    print(f"     Worst pattern detectability {worst_orth:.2f} sigma "
          f"(at the highest S/N in the prior).")
    if worst_orth < 1.0:
        print(f"     [OK]   no. The distortion is below the per-channel position")
        print(f"            noise everywhere in the S/N prior, so no estimator can")
        print(f"            see it.")
    elif worst_orth < 5.0:
        print(f"     [WARN] detectable at the top of the S/N prior, though not at")
        print(f"            typical S/N. It cannot bias DM (it is orthogonal to the")
        print(f"            DM direction) but the brightest observations are mildly")
        print(f"            out of distribution. Worth confirming on simulations")
        print(f"            before trusting the highest-S/N epochs.")
    else:
        print(f"     [FAIL] the brightest observations are clearly outside the")
        print(f"            training distribution. Condition the model on the")
        print(f"            per-observation grid.")

    print()
    if worst < 1.0 and worst_orth < 5.0:
        print(f"  => ONE model on the nominal grid. Give each station its own YAML")
        print(f"     differing ONLY in the `observatory:` block, which is excluded")
        print(f"     from the fingerprint, so all of them share that model.")
    else:
        print(f"  => Do not share one nominal-grid model as it stands. Either train")
        print(f"     per station, or put the per-channel frequencies into the feature")
        print(f"     vector, which also handles the")
        print(f"     epoch-to-epoch drift that per-station grids cannot.")
    print()
    print(f"  CAVEAT: this is a linear-projection estimate of how the frequency")
    print(f"  mismatch maps onto DM. It is the right order of magnitude and the")
    print(f"  right way to rank the options, but it is not a substitute for the")
    print(f"  definitive test: simulate on one station's grid, infer with a model")
    print(f"  trained on the nominal grid, and measure the posterior shift.")
    print("=" * 72)

    # ---------- emit the nominal grid ----------
    print(f"\n  Nominal grid for the config ({src}):\n")
    bw = np.diff(nominal).mean()
    print(f"  n_freq_channels:    {n_ch}")
    print(f"  n_phase_bins:       {N_BINS}")
    print(f"  freq_low_mhz:       {nominal[0] - bw/2:.6f}")
    print(f"  freq_high_mhz:      {nominal[-1] + bw/2:.6f}")
    print(f"  reference_freq_mhz: {cfg.ref_freq_mhz:.11f}")
    print(f"  channel_freqs_mhz:")
    for v in nominal:
        print(f"    - {v:.9f}")


if __name__ == "__main__":
    main()
