#!/usr/bin/env python3
"""
simul_valid.py — True-vs-recovered validation grid for the SBI posterior.

Generates N_TEST simulations with parallel workers (same logic as pp_tests.py),
runs the posterior in batched GPU mode, then plots the true-vs-recovered grid
and normalised residuals.

No chunk files are loaded — generation + inference is the fast path.

Outputs
-------
  <outdir>/true_vs_recovered.png     — 2×3 grid, one panel per parameter
  <outdir>/normalised_residuals.png  — (recovered − true) / σ  vs  true
"""

import argparse
import multiprocessing as mp
import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

from .natastro_style import set_style, NAT_W_DOUBLE
from .pulsarDMvary import sim_generator
from .pulsar_config import get_config

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ──────────────────────────────────────────────────────────────────────────────
# Worker — must be at module level for multiprocessing pickling
# ──────────────────────────────────────────────────────────────────────────────
def _sim_worker(task):
    seed, config_path = task
    import torch
    torch.set_num_threads(1)
    np.random.seed(seed)
    return next(sim_generator(1, config_path=config_path))


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="True-vs-recovered validation grid for the SBI posterior."
    )
    parser.add_argument("--dataset",      type=str, required=True,
                        help="Dataset directory (used for posterior and output paths).")
    parser.add_argument("--posterior",    type=str, default=None,
                        help="Path to posterior.pt. Defaults to <dataset>/posterior.pt.")
    parser.add_argument("--n_test",       type=int, default=1000,
                        help="Number of test simulations (default 1000).")
    parser.add_argument("--n_samples",    type=int, default=5000,
                        help="Posterior samples per test case (default 5000).")
    parser.add_argument("--sample_batch", type=int, default=50,
                        help="Test cases per GPU batch (default 50). Reduce if OOM.")
    parser.add_argument("--n_workers",    type=int, default=None,
                        help="Simulation worker processes (default: cpu_count).")
    parser.add_argument("--outdir",       type=str, default=None,
                        help="Output directory for plots (default: same as --dataset).")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed (default 42).")
    parser.add_argument("--just_plot",    action="store_true",
                        help="Skip simulation and inference — load existing simul_valid_summary.csv and replot.")
    parser.add_argument("--config",       type=str, default=None,
                        help="Path to the YAML config. Default: the config recorded in "
                             "the posterior's posterior_meta.pt sidecar.")
    args = parser.parse_args(argv)

    outdir       = args.outdir or args.dataset
    posterior_pt = args.posterior or os.path.join(args.dataset, "posterior.pt")
    n_workers    = args.n_workers or min(os.cpu_count() or 1, args.n_test)
    os.makedirs(outdir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────────
    # Config — labels, prior bounds and display conversions all come from here.
    # This file used to carry the FOURTH copy of PRIOR_LOW/PRIOR_HIGH.
    # ──────────────────────────────────────────────────────────────────────────────
    _meta_path   = os.path.join(os.path.dirname(posterior_pt), "posterior_meta.pt")
    _config_path = args.config
    if _config_path is None:
        if os.path.isfile(_meta_path):
            _config_path = torch.load(_meta_path, map_location="cpu",
                                      weights_only=False).get("config_path")
        if _config_path is None:
            _config_path = os.path.join(args.dataset, "metadata.pt")
            if os.path.isfile(_config_path):
                _config_path = torch.load(_config_path, map_location="cpu",
                                          weights_only=False).get("config_path")
            else:
                _config_path = None
        if _config_path is None:
            raise SystemExit(
                f"No --config given and neither posterior_meta.pt nor metadata.pt "
                f"records one. Pass --config explicitly."
            )

    cfg    = get_config(_config_path)
    P0_S   = cfg.period_s
    _SPECS = cfg.display_specs()
    # Prior bounds in theta space, for the in-bounds mask on posterior samples.
    _PRIOR_LOW  = cfg.prior_low.astype(float)
    _PRIOR_HIGH = cfg.prior_high.astype(float)
    print(f"Config: {os.path.basename(cfg.path)}  (fingerprint {cfg.fingerprint})")

    _usetex = set_style()

    # (label, unit, theta->display fn, display low, display high) per parameter
    PARAMS = [(s["label"], s["unit"], s["fn"], s["low"], s["high"]) for s in _SPECS]
    N_PARAMS = len(PARAMS)

    # Panel grid sized to the parameter count, not fixed at 2x3.
    _NCOLS = min(3, N_PARAMS)
    _NROWS = int(np.ceil(N_PARAMS / _NCOLS))

    # Points are coloured by true S/N — except on the S/N panel itself, where that
    # would be trivial, so it is coloured by DM instead. Indices found by name from
    # the config rather than the hardcoded `k == 4`.
    _INFERRED_NAMES = [p.name for p in cfg.inferred]
    _IDX_SNR = _INFERRED_NAMES.index(cfg.snr_param) if cfg.snr_param in _INFERRED_NAMES else -1
    _IDX_DM  = _INFERRED_NAMES.index(cfg.dm_param)  if cfg.dm_param  in _INFERRED_NAMES else -1

    if _usetex:
        _RESID_YLABEL = r"(Recovered $-$ True) / $\sigma$"
        _CBLABEL_SNR  = r"True S/N"
        _CBLABEL_DM   = r"True $\Delta\mathrm{DM}$ (pc$\,$cm$^{-3}$)"
    else:
        _RESID_YLABEL = "(Recovered − True) / σ"
        _CBLABEL_SNR  = "True S/N"
        _CBLABEL_DM   = "True ΔDM (pc cm⁻³)"


    def _colour_for_panel(k, true_disp, snr_norm):
        """Colour values + colourbar label for panel `k`.

    Colour by true S/N everywhere except the S/N panel itself, where S/N would
    just reproduce the x-axis; there, colour by DM. Falls back to S/N colouring
    if the config does not have a DM parameter in `inferred`.
    """
        if k == _IDX_SNR and _IDX_DM >= 0:
            col = true_disp[:, _IDX_DM]
            rng = col.max() - col.min()
            return (col - col.min()) / (rng + 1e-12), _CBLABEL_DM
        return snr_norm, _CBLABEL_SNR


    # CSV column stems. Derived from the config, so a new parameter set produces
    # matching columns automatically. Note these are the theta-space names
    # ('log_snr', not 'SNR'); the stored VALUES are in display units.
    PARAM_KEYS = [s["name"] for s in _SPECS]

    if args.just_plot:
        # ── Load existing CSV ─────────────────────────────────────────────────────
        csv_path = os.path.join(outdir, "simul_valid_summary.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"simul_valid_summary.csv not found in {outdir}. "
                "Run without --just_plot first."
            )
        _df      = pd.read_csv(csv_path)
        true_disp = np.column_stack([_df[f"{n}_true"].values   for n in PARAM_KEYS])
        med_disp  = np.column_stack([_df[f"{n}_median"].values for n in PARAM_KEYS])
        lo_disp   = np.column_stack([_df[f"{n}_lo"].values     for n in PARAM_KEYS])
        hi_disp   = np.column_stack([_df[f"{n}_hi"].values     for n in PARAM_KEYS])
        print(f"Loaded {len(_df)} test cases from {csv_path}")

    else:
        # ── Device + posterior ────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        posterior = torch.load(posterior_pt, map_location="cpu", weights_only=False)
        posterior.posterior_estimator = posterior.posterior_estimator.to(device)
        emb = posterior.posterior_estimator.embedding_net
        if hasattr(emb, "_mean"):
            emb._mean = emb._mean.to(device)
        if hasattr(emb, "_std"):
            emb._std  = emb._std.to(device)
        print(f"Posterior loaded from {posterior_pt}")

        # ── Generate test simulations in parallel ─────────────────────────────────
        rng   = np.random.default_rng(args.seed)
        seeds = rng.integers(0, 2**31, size=args.n_test).tolist()

        print(f"\nGenerating {args.n_test} test simulations ({n_workers} workers) ...")
        tasks = [(s, cfg.path) for s in seeds]
        try:
            with mp.get_context("fork").Pool(processes=n_workers) as pool:
                all_sims = list(pool.imap(_sim_worker, tasks, chunksize=10))
        except Exception:
            all_sims = [_sim_worker(t) for t in tasks]
        print(f"Generated {len(all_sims)} simulations.")

        def _extract_theta(sim):
            # Already ordered and log-transformed by the config
            return np.asarray(sim["theta"], dtype=np.float32)

        def _build_x(sim):
            t = lambda a: torch.as_tensor(a, dtype=torch.float32)
            return torch.cat([t(sim["portrait"]).flatten(),
                              t(sim["flux_per_channel"]),
                              t(sim["centroids"])], dim=0)

        theta_test = np.array([_extract_theta(s) for s in all_sims])
        x_batch    = torch.stack([_build_x(s) for s in all_sims]).to(device)
        n_test     = len(all_sims)

        # ── Batched GPU inference ─────────────────────────────────────────────────
        print(f"\nRunning posterior on {n_test} test simulations "
              f"({args.n_samples} samples, batch={args.sample_batch}) ...")

        chunks = []
        iterable = range(0, n_test, args.sample_batch)
        if _HAS_TQDM:
            iterable = tqdm(iterable, desc="Inferring", total=(n_test + args.sample_batch - 1) // args.sample_batch)

        for start in iterable:
            end = min(start + args.sample_batch, n_test)
            with torch.no_grad():
                chunk = posterior.posterior_estimator.sample(
                    (args.n_samples,), condition=x_batch[start:end]
                ).cpu().numpy()
            chunks.append(chunk)

        theta_all = np.concatenate(chunks, axis=1)
        print("Inference complete.")

        # ── Per-test statistics with prior-bounds filter ──────────────────────────
        # N_PARAMS, not a literal 6: a config with free scattering has 7 inferred
        # parameters and this silently broadcast-failed at (7,) into (6,).
        rec_median = np.zeros((n_test, N_PARAMS))
        rec_lo     = np.zeros((n_test, N_PARAMS))
        rec_hi     = np.zeros((n_test, N_PARAMS))

        for i in range(n_test):
            s = theta_all[:, i, :]
            in_bounds = ~((s < _PRIOR_LOW) | (s > _PRIOR_HIGH)).any(axis=1)
            if in_bounds.sum() > 20:
                s = s[in_bounds]
            rec_median[i] = np.median(s, axis=0)
            rec_lo[i]     = np.percentile(s, 16, axis=0)
            rec_hi[i]     = np.percentile(s, 84, axis=0)

        # ── Convert to display units ──────────────────────────────────────────────
        def _to_display(raw):
            out = raw.copy()
            for k, (_, _, conv, _, _) in enumerate(PARAMS):
                out[:, k] = conv(raw[:, k])
            return out

        true_disp = _to_display(theta_test)
        med_disp  = _to_display(rec_median)
        lo_disp   = _to_display(rec_lo)
        hi_disp   = _to_display(rec_hi)

        # ── Save summary CSV ──────────────────────────────────────────────────────
        summary_rows = []
        for i in range(n_test):
            row = {"test_case": i}
            for k, name in enumerate(PARAM_KEYS):
                row[f"{name}_true"]   = true_disp[i, k]
                row[f"{name}_median"] = med_disp[i, k]
                row[f"{name}_lo"]     = lo_disp[i, k]
                row[f"{name}_hi"]     = hi_disp[i, k]
            summary_rows.append(row)
        _csv_out = os.path.join(outdir, "simul_valid_summary.csv")
        pd.DataFrame(summary_rows).to_csv(_csv_out, index=False)
        print(f"Saved → {_csv_out}")

    # Column of the S/N parameter, by name from the config rather than index 4.
    # Falls back to the first column so the colour mapping still works for a config
    # with no S/N in `inferred`.
    snr_true = true_disp[:, _IDX_SNR if _IDX_SNR >= 0 else 0]
    snr_norm = (snr_true - snr_true.min()) / (snr_true.max() - snr_true.min() + 1e-12)
    cmap     = cm.viridis


    # ──────────────────────────────────────────────────────────────────────────────
    # Plot 1: True vs Recovered grid
    # ──────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(_NROWS, _NCOLS,
                             figsize=(NAT_W_DOUBLE, 2.5 * _NROWS), squeeze=False)
    axes = axes.flatten()[:N_PARAMS]

    for k, ax in enumerate(axes):
        label, unit, _, _, _ = PARAMS[k]
        ax_label = label + (rf"\,({unit})" if (_usetex and unit) else
                            (f" ({unit})" if unit else ""))

        t = true_disp[:, k]
        m = med_disp[:, k]
        l = lo_disp[:, k]
        h = hi_disp[:, k]

        c_norm, cb_label = _colour_for_panel(k, true_disp, snr_norm)

        ax.errorbar(t, m, yerr=[m - l, h - m],
                    fmt="none", ecolor="grey", alpha=0.25, lw=0.5, zorder=2)
        sc = ax.scatter(t, m, c=c_norm, cmap=cmap, s=6, alpha=0.75, zorder=3, linewidths=0)

        lo_all = min(t.min(), m.min())
        hi_all = max(t.max(), m.max())
        pad    = 0.07 * (hi_all - lo_all + 1e-12)
        lims   = (lo_all - pad, hi_all + pad)
        ax.plot(lims, lims, "k-", lw=1.0, zorder=4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        bias    = np.mean(m - t)
        scatter = np.std(m - t)
        rmse    = np.sqrt(np.mean((m - t) ** 2))
        ax.text(0.04, 0.96,
                f"bias   = {bias:+.4f}\nRMSE  = {rmse:.4f}",
                transform=ax.transAxes, fontsize=4.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                          edgecolor="0.7", alpha=0.9))

        ax.set_xlabel(f"True {ax_label}")
        ax.set_ylabel(f"Recovered {ax_label}")
        ax.set_title(label, pad=3)

        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(cb_label, fontsize=6.5)
        cb.ax.tick_params(labelsize=6.5)

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=1.0)
    out1 = os.path.join(outdir, "true_vs_recovered.png")
    fig.savefig(out1)
    plt.close()
    print(f"Saved → {out1}")


    # ──────────────────────────────────────────────────────────────────────────────
    # Plot 2: Normalised residuals  (recovered − true) / σ  vs  true value
    # ──────────────────────────────────────────────────────────────────────────────
    sigma_disp = (hi_disp - lo_disp) / 2.0

    fig2, axes2 = plt.subplots(_NROWS, _NCOLS,
                               figsize=(NAT_W_DOUBLE, 2.5 * _NROWS), squeeze=False)
    axes2 = axes2.flatten()[:N_PARAMS]

    for k, ax in enumerate(axes2):
        label, unit, _, _, _ = PARAMS[k]
        xlabel = label + (rf"\,({unit})" if (_usetex and unit) else
                          (f" ({unit})" if unit else ""))

        t      = true_disp[:, k]
        resid  = med_disp[:, k] - t
        sigma  = sigma_disp[:, k]
        norm_r = resid / (sigma + 1e-12)

        c_norm, cb_label = _colour_for_panel(k, true_disp, snr_norm)

        ax.scatter(t, norm_r, c=c_norm, cmap=cmap, s=6, alpha=0.75, linewidths=0)
        ax.axhline( 0, color="k",       lw=0.8, ls="-")
        ax.axhline( 1, color="crimson", lw=0.7, ls="--")
        ax.axhline(-1, color="crimson", lw=0.7, ls="--")

        mu_nr  = np.mean(norm_r)
        std_nr = np.std(norm_r)
        ax.text(0.04, 0.96,
                f"mean = {mu_nr:+.2f}\nstd  = {std_nr:.2f}",
                transform=ax.transAxes, fontsize=7, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                          edgecolor="0.7", alpha=0.9))

        ax.set_xlabel(f"True {xlabel}")
        ax.set_ylabel(_RESID_YLABEL)
        ax.set_title(label, pad=3)
        ax.set_ylim(-6, 6)

        cb = fig2.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(cb_label, fontsize=6.5)
        cb.ax.tick_params(labelsize=6.5)

    fig2.tight_layout(pad=0.8, h_pad=1.2, w_pad=1.0)
    out2 = os.path.join(outdir, "normalised_residuals.png")
    fig2.savefig(out2)
    plt.close()
    print(f"Saved → {out2}")


    # ──────────────────────────────────────────────────────────────────────────────
    # Plot 3: Raw DM residual (recovered − true) vs true ΔDM
    # ──────────────────────────────────────────────────────────────────────────────
    dm_true  = true_disp[:, 0]        # pc cm⁻³
    dm_resid = med_disp[:, 0] - dm_true   # recovered − true

    bias_dm  = np.mean(dm_resid)
    rmse_dm  = np.sqrt(np.mean(dm_resid ** 2))
    std_dm   = np.std(dm_resid)

    # Running median in 10 equal-width bins across the prior range
    _n_bins   = 10
    _bin_edges = np.linspace(dm_true.min(), dm_true.max(), _n_bins + 1)
    _bin_cen, _bin_med = [], []
    for _bi in range(_n_bins):
        _mask = (dm_true >= _bin_edges[_bi]) & (dm_true < _bin_edges[_bi + 1])
        if _mask.sum() >= 3:
            _bin_cen.append(0.5 * (_bin_edges[_bi] + _bin_edges[_bi + 1]))
            _bin_med.append(np.median(dm_resid[_mask]))
    _bin_cen = np.array(_bin_cen)
    _bin_med = np.array(_bin_med)

    if _usetex:
        _dm_res_xlabel = r"True $\Delta\mathrm{DM}$ (pc$\,$cm$^{-3}$)"
        _dm_res_ylabel = r"$\Delta\mathrm{DM}_{\rm rec} - \Delta\mathrm{DM}_{\rm true}$ (pc$\,$cm$^{-3}$)"
    else:
        _dm_res_xlabel = "True ΔDM (pc cm⁻³)"
        _dm_res_ylabel = "ΔDM recovered − ΔDM true (pc cm⁻³)"

    from .natastro_style import NAT_W_SINGLE
    fig3, ax3 = plt.subplots(figsize=(NAT_W_SINGLE * 1.5, NAT_W_SINGLE * 1.3))

    sc3 = ax3.scatter(dm_true, dm_resid, c=snr_norm, cmap=cmap,
                      s=6, alpha=0.65, linewidths=0, zorder=2)
    if len(_bin_cen):
        ax3.plot(_bin_cen, _bin_med, color="crimson", lw=1.2, ls="-",
                 marker="o", markersize=3, zorder=4, label="bin median")
        ax3.legend(fontsize=7, loc="upper right")

    ax3.axhline(0,        color="k",   lw=0.9, ls="--", zorder=3)
    ax3.axhline( std_dm,  color="0.5", lw=0.7, ls=":",  zorder=3)
    ax3.axhline(-std_dm,  color="0.5", lw=0.7, ls=":",  zorder=3)

    ax3.set_xlabel(_dm_res_xlabel)
    ax3.set_ylabel(_dm_res_ylabel)

    ax3.text(0.04, 0.96,
             f"bias = {bias_dm:+.5f}\nRMSE = {rmse_dm:.5f}\nstd  = {std_dm:.5f}",
             transform=ax3.transAxes, fontsize=7, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                       edgecolor="0.7", alpha=0.9))

    cb3 = fig3.colorbar(sc3, ax=ax3, pad=0.02, fraction=0.046)
    cb3.set_label(_CBLABEL_SNR, fontsize=7)
    cb3.ax.tick_params(labelsize=6.5)

    fig3.tight_layout(pad=0.5)
    out3 = os.path.join(outdir, "dm_bias.png")
    fig3.savefig(out3)
    plt.close()
    print(f"Saved → {out3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
