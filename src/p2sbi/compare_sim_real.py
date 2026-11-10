"""
compare_sim_real.py
-------------------
Overlay simulated portraits on a real archive, to see whether the simulator is
producing something that looks like the data it will be asked to interpret.

This is the check that matters before committing to a large simulation run: the
network can only be as good as the agreement between these two.

    conda run -n penv_psr python -m p2sbi.compare_sim_real \
        archives/DE601/*.ar --config configs/german_J2145.yaml --n 10

    # or from a .pt written by real_archive_loader.py (no psrchive needed)
    conda run -n penv python -m p2sbi.compare_sim_real real_data/real_*.pt \
        --config configs/german_J2145.yaml --n 10

Produces  <outdir>/compare_sim_real.png  with four panels:
  1. real frequency-scrunched profile with N simulated ones overlaid
  2. the same, zoomed on the on-pulse region
  3. real portrait vs one simulated portrait
  4. per-channel flux and centroid sweep, real vs the simulated envelope

The simulated S/N is matched to the real profile's measured off-pulse RMS, and
DM and ToA are set to zero, so the comparison is about PROFILE SHAPE and
spectral behaviour rather than about the parameters being inferred.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pulsar_config import get_config, resolve_config_path
from .pulsarDMvary import (PulsarProfileSimulator, build_components,
                          _draw_channel_effects, _finalise_portrait)


def load_real(path, cfg):
    """Return (portrait, freqs) ascending, max-normalised like the pipeline."""
    if path.lower().endswith(".pt"):
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        p = d["portrait"].numpy()
        f = d.get("metadata", {}).get("channel_freqs_mhz")
        return p, (np.asarray(f) if f is not None else cfg.freqs_mhz)

    import psrchive
    a = psrchive.Archive_load(path)
    a.tscrunch(); a.pscrunch(); a.remove_baseline(); a.dedisperse()
    p = a.get_data()[0, 0, :, :]
    sub = a.get_Integration(0)
    f = np.array([sub.get_centre_frequency(i) for i in range(a.get_nchan())])
    if f.size > 1 and f[0] > f[-1]:
        f, p = f[::-1].copy(), p[::-1, :].copy()
    return p / (p.max() + 1e-12), f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("real", help="PSRCHIVE .ar archive or real_*.pt")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=10, help="simulations to overlay")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr", type=float, default=None,
                    help="Override the S/N matched from the real profile")
    args = ap.parse_args()

    cfg = get_config(resolve_config_path(args.config))
    real, rfreq = load_real(args.real, cfg)
    if real.shape != (cfg.n_chan, cfg.n_bins):
        raise SystemExit(f"real portrait is {real.shape} but the config describes "
                         f"({cfg.n_chan}, {cfg.n_bins})")

    prof = real.mean(axis=0)
    prof = prof / prof.max()
    off  = prof < 0.05
    rms  = prof[off].std()
    snr_real = 1.0 / rms
    snr = args.snr if args.snr is not None else snr_real

    print(f"Config        : {os.path.basename(cfg.path)}  ({cfg.fingerprint})")
    print(f"Real archive  : {os.path.basename(args.real)}")
    print(f"  off-pulse RMS {rms:.5f}  -> peak S/N {snr_real:.1f}")
    _sp = next(p for p in cfg.inferred if p.name == cfg.snr_param)
    print(f"  config snr prior U[{_sp.low:g}, {_sp.high:g}]"
          + ("   [!] the real S/N is ABOVE the prior ceiling"
             if snr_real > _sp.high else ""))
    print(f"  simulating {args.n} realisations at S/N {snr:.1f}, DM=0, toa=0")

    # ---- simulate ----
    np.random.seed(args.seed)
    sim = PulsarProfileSimulator.from_config(cfg)
    sims, draws = [], []
    for _ in range(args.n):
        s = cfg.sample()
        s[cfg.dm_param]  = 0.0
        s[cfg.snr_param] = float(snr)
        if "toa" in s:
            s["toa"] = 0.0
        sc = cfg.scattering_values(s)
        r = sim.generate_portrait(
            build_components(cfg, s), epoch_mjd=cfg.epoch_mjd, dm=0.0,
            tau_scatter_ms=sc["tau_ms"], scattering_index=sc["spectral_index"],
            dm_smearing=cfg.dm_smearing, add_noise=cfg.add_noise, snr=float(snr),
            noise_model=cfg.noise_model, npol=cfg.npol, t_obs_s=cfg.t_obs_s,
            **_draw_channel_effects(cfg, s, sim))                    # Stage S; Nones when off
        p, f, c, m = _finalise_portrait(r["portrait"], cfg)
        sims.append((p, f, c)); draws.append(s)

    # ---- align the real profile to the simulated phase ----
    # The simulator puts Main at cfg.reference_phase; a real archive's folding
    # convention need not agree. Cross-correlate once and roll, so the shapes
    # are compared rather than the (degenerate) absolute phase.
    sp   = np.mean([p.mean(axis=0) for p, _, _ in sims], axis=0)
    sp  /= sp.max()
    cc   = np.fft.irfft(np.fft.rfft(prof) * np.conj(np.fft.rfft(sp)), n=prof.size)
    shift = int(np.argmax(cc))
    prof_al = np.roll(prof, -shift)
    real_al = np.roll(real, -shift, axis=1)
    print(f"  real profile rolled by {-shift:+d} bins to match the simulated phase")

    # ---- plot ----
    try:
        from .natastro_style import set_style, NAT_W_DOUBLE
        set_style(); W = NAT_W_DOUBLE * 2.0
    except Exception:
        W = 14.0
    fig, ax = plt.subplots(2, 2, figsize=(W, W * 0.62))

    b = np.arange(cfg.n_bins)
    for i, (p, _, _) in enumerate(sims):
        q = p.mean(axis=0); q /= q.max()
        ax[0, 0].plot(b, q, lw=0.6, alpha=0.45, color="tab:orange",
                      label="simulated" if i == 0 else None)
    ax[0, 0].plot(b, prof_al, "k-", lw=1.3, label="real (aligned)")
    ax[0, 0].set_xlabel("Phase bin"); ax[0, 0].set_ylabel("Normalised amplitude")
    ax[0, 0].set_title(f"Frequency-scrunched profile — {args.n} simulations vs real")
    ax[0, 0].legend(fontsize=7)

    on = np.where(prof_al > 0.03)[0]
    lo, hi = max(0, on.min() - 20), min(cfg.n_bins, on.max() + 20)
    for p, _, _ in sims:
        q = p.mean(axis=0); q /= q.max()
        ax[0, 1].plot(b[lo:hi], q[lo:hi], lw=0.6, alpha=0.45, color="tab:orange")
    ax[0, 1].plot(b[lo:hi], prof_al[lo:hi], "k-", lw=1.3)
    for cc_ in cfg.components:
        ph = (float(cc_.phase_center_ref) * cfg.n_bins)
        if lo <= ph <= hi:
            ax[0, 1].axvline(ph, color="tab:blue", ls=":", lw=0.8)
            ax[0, 1].text(ph, 1.02, cc_.name, rotation=90, fontsize=6,
                          ha="center", va="bottom", color="tab:blue")
    ax[0, 1].set_xlabel("Phase bin"); ax[0, 1].set_ylabel("Normalised amplitude")
    ax[0, 1].set_title("On-pulse zoom, with the config's component phases")

    im = ax[1, 0].imshow(np.vstack([real_al, np.full((3, cfg.n_bins), np.nan),
                                    sims[0][0]]),
                         aspect="auto", origin="lower", cmap="viridis")
    ax[1, 0].axhline(cfg.n_chan + 1, color="w", lw=1.0)
    ax[1, 0].set_xlabel("Phase bin")
    ax[1, 0].set_yticks([cfg.n_chan // 2, cfg.n_chan + 3 + cfg.n_chan // 2])
    ax[1, 0].set_yticklabels(["real", "sim"])
    ax[1, 0].set_title("Portrait: real (below) vs one simulation (above)")
    fig.colorbar(im, ax=ax[1, 0], pad=0.02, fraction=0.046)

    rf = real_al.mean(axis=1); rf /= rf.max()
    F = np.array([f / f.max() for _, f, _ in sims])
    ax[1, 1].fill_between(rfreq, F.min(axis=0), F.max(axis=0), alpha=0.3,
                          color="tab:orange", label="simulated envelope")
    ax[1, 1].plot(rfreq, F.mean(axis=0), color="tab:orange", lw=1.0)
    ax[1, 1].plot(rfreq, rf, "k-", lw=1.3, label="real")
    ax[1, 1].set_xlabel("Frequency (MHz)")
    ax[1, 1].set_ylabel("Flux per channel (peak-normalised)")
    ax[1, 1].set_title("Spectral behaviour: flux vs frequency")
    ax[1, 1].legend(fontsize=7)

    fig.suptitle(f"{os.path.basename(args.real)}   vs   "
                 f"{os.path.basename(cfg.path)} ({cfg.fingerprint})   "
                 f"S/N={snr:.0f}", fontsize=9)
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "compare_sim_real.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved -> {out}")

    # ---- numbers ----
    res = np.array([prof_al - (p.mean(axis=0) / p.mean(axis=0).max())
                    for p, _, _ in sims])
    print(f"\n  real-minus-simulated, on-pulse:")
    print(f"    best  realisation: rms {np.abs(res).std(axis=1).min():.5f}")
    print(f"    worst realisation: rms {np.abs(res).std(axis=1).max():.5f}")
    print(f"    real off-pulse rms {rms:.5f}  (the floor a perfect model would reach)")


if __name__ == "__main__":
    main()
