"""Per-channel flux / noise / S/N statistics: simulations against real data (Stage S, S7.3).

The scintillation model is judged by whether simulated portraits look like real
ones channel by channel. This computes the statistics for simulations from a
config and for real data, either from a directory of real_archive_loader `.pt`
files or from a JSON of previously recorded values, and compares them.

Statistics (definitions fixed; the J1713 GBT targets were measured with them):
  per epoch, across live channels, then the median over epochs
    sd_ln_flux       sd of ln(on-pulse flux)        on-pulse: mean profile > 5% of peak
    sd_ln_noise      sd of ln(off-pulse rms)        the instrument gain alone
    corr_flux_noise  corr(ln flux, ln noise)        ~1 if one gain multiplies both
    sd_ln_snr        sd of ln(flux / noise)         scintillation (signal only)
  pooled, from the flux_per_channel feature (ln, mean removed per epoch, minus
  the median pattern over epochs)
    acf_1..acf_4     correlation between channels 1..4 apart
    skew             skew (the tail of deep fades)
  per epoch, ln flux_per_channel against ln(nu/nu_ref), quantiles over epochs
    slope_p5/p50/p95 power-law slope (p50 also depends on the gamma priors)
    scatter_p50      robust scatter about that power law

Tolerances: relative 15% on spreads and correlations, 0.3 absolute on skew,
15% of the real p95 - p5 range on slope quantiles.

S/N MATTERS for the noise statistics. The off-pulse rms picks up pulse emission
leaking into the "off-pulse" bins (broad components), so at high S/N it tracks
the channel flux: for J1713 at fixed scintillation, corr_flux_noise goes 0.53 ->
0.99 and sd_ln_snr 0.97 -> 0.18 from S/N 150 to 4500. Compare at the real data's
S/N: pin snr to its range (J1713 GBT: SBI medians 106-2843, median 457), e.g.
--pin snr=106:2843.

  # prior predictive of a config against recorded targets
  python -m p2sbi.scint_acceptance --config configs/gbt_J1713_scint.yaml \
      --targets configs/scint_targets_J1713_GBT.json --n 1000
  # against real data
  python -m p2sbi.scint_acceptance --config <cfg> --real_dir <dir of .pt> --n 1000
  # pin inferred parameters (theta units are handled: give LINEAR values)
  ... --pin scint_bw=20 --pin scint_dt=1500
"""
import argparse, glob, json, os, sys
import numpy as np

from .pulsar_config import load_config

STATS = ["sd_ln_flux", "sd_ln_noise", "corr_flux_noise", "sd_ln_snr",
         "acf_1", "acf_2", "acf_3", "acf_4", "skew",
         "slope_p5", "slope_p50", "slope_p95", "scatter_p50"]


def _per_epoch(P):
    """sd ln flux, sd ln noise, corr, sd ln S/N for one portrait (rows = channels)."""
    live = np.any(P != 0, 1)
    prof = P[live].mean(0); on = prof > 0.05 * prof.max()
    sig = P[:, ~on].std(1)
    f = (P[:, on] - np.median(P[:, ~on], 1, keepdims=True)).sum(1)
    ok = live & (f > 0) & (sig > 0)
    lf, ls = np.log(f[ok]), np.log(sig[ok])
    return lf.std(), ls.std(), np.corrcoef(lf, ls)[0, 1], np.log(f[ok] / sig[ok]).std()


def statistics(portraits, fluxes, freqs, ref_mhz):
    x = np.log(np.asarray(freqs) / ref_mhz)
    pe = np.array([_per_epoch(np.asarray(P, float)) for P in portraits])
    L = np.full((len(fluxes), x.size), np.nan)
    sl, sc = [], []
    for i, (P, f) in enumerate(zip(portraits, fluxes)):
        f = np.asarray(f, float)
        ok = np.any(np.asarray(P) != 0, 1) & (f > 0)
        L[i, ok] = np.log(f[ok]) - np.log(f[ok]).mean()
        c = np.polyfit(x[ok], np.log(f[ok]), 1); r = np.log(f[ok]) - np.polyval(c, x[ok])
        sl.append(c[0]); sc.append(1.4826 * np.median(np.abs(r - np.median(r))))
    V = L - np.nanmedian(L, 0)
    v = V[np.isfinite(V)]

    def acf(k):
        a, b = V[:, :-k], V[:, k:]; m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])
    out = dict(zip(STATS[:4], np.median(pe, 0).tolist()))
    out.update({f"acf_{k}": acf(k) for k in (1, 2, 3, 4)})
    out["skew"] = float(np.mean((v - v.mean())**3) / v.std()**3)
    out.update(slope_p5=float(np.quantile(sl, .05)), slope_p50=float(np.quantile(sl, .5)),
               slope_p95=float(np.quantile(sl, .95)), scatter_p50=float(np.median(sc)))
    return out


def compare(sim, ref):
    rng = ref["slope_p95"] - ref["slope_p5"]
    rows, ok_all = [], True
    for k in STATS:
        if k not in ref:
            continue
        s, r = sim[k], ref[k]
        if k.startswith("slope"):
            ok, tol = abs(s - r) <= 0.15 * rng, f"±{0.15 * rng:.2f}"
        elif k == "skew":
            ok, tol = abs(s - r) <= 0.3, "±0.30"
        else:
            ok, tol = abs(s / r - 1) <= 0.15, "±15%"
        info = k == "slope_p50"                       # set by the gamma priors, not scintillation
        ok_all &= ok or info
        rows.append((k, s, r, tol, "info" if info else ("PASS" if ok else "FAIL")))
    return rows, ok_all


def simulate(cfg_path, n, seed, pins):
    from .pulsarDMvary import sim_generator, simulate_from_theta
    cfg = load_config(cfg_path)
    np.random.seed(seed)
    P, F = [], []
    if not pins:
        for s in sim_generator(n, config_path=cfg_path):
            P.append(s["portrait"]); F.append(s["flux_per_channel"])
        return P, F
    unknown = set(pins) - {p.name for p in cfg.inferred}
    if unknown:
        raise SystemExit(f"--pin: {sorted(unknown)} are not inferred parameters of {cfg_path} "
                         f"(only inferred ones can be pinned; nuisance stays drawn from its prior)")
    for i in range(n):
        drawn = cfg.sample()
        for k, v in pins.items():                    # a value, or (lo, hi): log-uniform per simulation
            drawn[k] = float(np.exp(np.random.uniform(np.log(v[0]), np.log(v[1])))) if isinstance(v, tuple) else v
        p, f, _ = simulate_from_theta(cfg.theta_vector(drawn), config_path=cfg_path, seed=seed * 100003 + i)
        P.append(p); F.append(f)
    return P, F


def parse_pins(items):
    """'name=value' or 'name=lo:hi' (log-uniform per simulation)."""
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        out[k] = tuple(float(x) for x in v.split(":")) if ":" in v else float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real_dir", help="directory of real_archive_loader .pt files")
    ap.add_argument("--targets", help="JSON of recorded real statistics (needs 'stats')")
    ap.add_argument("--pin", action="append", default=[],
                    help="name=value or name=lo:hi (log-uniform per simulation), linear units")
    ap.add_argument("--out", help="write the simulated statistics (and the comparison) as JSON")
    a = ap.parse_args()

    cfg = load_config(a.config)
    pins = parse_pins(a.pin)
    P, F = simulate(a.config, a.n, a.seed, pins)
    sim = statistics(P, F, cfg.freqs_mhz, cfg.ref_freq_mhz)

    ref, label = None, None
    if a.real_dir:
        import torch
        files = sorted(glob.glob(os.path.join(a.real_dir, "*.pt")))
        d = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
        ref = statistics([x["portrait"].numpy() for x in d], [x["flux_per_channel"].numpy() for x in d],
                         cfg.freqs_mhz, cfg.ref_freq_mhz)
        label = f"real ({len(files)})"
    elif a.targets:
        t = json.load(open(a.targets)); ref, label = t["stats"], f"recorded ({t.get('n_epochs', '?')})"

    print(f"{os.path.basename(a.config)}  n={a.n}  pins={pins or 'none'}")
    if ref is None:
        for k in STATS:
            print(f"  {k:16s} {sim[k]:+8.3f}")
        result = {"sim": sim}
    else:
        rows, ok = compare(sim, ref)
        print(f"  {'statistic':16s} {'sim':>8s} {label:>16s} {'tol':>7s}")
        for k, s, r, tol, flag in rows:
            print(f"  {k:16s} {s:+8.3f} {r:+16.3f} {tol:>7s}  {flag}")
        print(f"  overall: {'PASS' if ok else 'FAIL'}")
        result = {"sim": sim, "ref": ref, "pass": bool(ok)}
    if a.out:
        json.dump(dict(result, config=a.config, n=a.n, pins=pins), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
