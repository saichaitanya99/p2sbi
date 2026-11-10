import threading
import torch
import os
import torch.nn as nn
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
import corner
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde

from .pulsarDMvary import get_device
from .pulsar_config import get_config, recorded_config_path
from .natastro_style import set_style, NAT_W_DOUBLE


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the DM scattering time sampling step using a trained SBI model and a test simulation.")
    parser.add_argument('--posterior_tensor', type=str, default="pulsar_simulation_posterior.pt", help='Path to the trained posterior tensor file for sampling.')
    parser.add_argument('--test_simulation', type=str, default="test_simulation.pt", help='Path to the test simulation tensor file for sampling.')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for sampling results. If not set, it will use the same directory as the posterior tensor file.')
    parser.add_argument('--config', type=str, default=None,
                        help="Path to the YAML config. Default: the config recorded in "
                             "the posterior's posterior_meta.pt sidecar.")
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Posterior samples to draw (default 10000).')
    args = parser.parse_args(argv)

    if args.outdir is None:
        outdir = os.path.dirname(args.posterior_tensor)
    else:
        outdir = args.outdir

    # --------------------------------------------------
    # Config: parameter names, labels, prior bounds and display conversions.
    # This file used to hold the FIFTH copy of the prior bounds.
    # --------------------------------------------------
    _meta_dir  = os.path.dirname(args.posterior_tensor)
    _meta_path = os.path.join(_meta_dir, "posterior_meta.pt")
    _meta = (torch.load(_meta_path, map_location="cpu", weights_only=False)
             if os.path.isfile(_meta_path) else {})
    _config_path = recorded_config_path(_meta, _meta_dir, args.config)

    cfg    = get_config(_config_path)
    _SPECS = cfg.display_specs()
    P0_S   = cfg.period_s
    print(f"Config: {os.path.basename(cfg.path)}  (fingerprint {cfg.fingerprint})")
    print(f"Parameters: {', '.join(cfg.param_names)}")


    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print("Using device:", device)

    posterior = torch.load(args.posterior_tensor,map_location=device, weights_only=False)

    posterior.posterior_estimator = posterior.posterior_estimator.to(device)
    #posterior.posterior_sampler = posterior.posterior_sampler.to(device)

    emb = posterior.posterior_estimator.embedding_net
    if hasattr(emb, "_mean"):
        emb._mean = emb._mean.to(device)
    if hasattr(emb, "_std"):
        emb._std = emb._std.to(device)


    test_sim = torch.load(args.test_simulation, map_location=device, weights_only=False)

    portrait           = test_sim["portrait"]             # (40, 512)
    flux               = test_sim["flux_per_channel"]     # (40,)
    centroids          = test_sim["centroids"]            # (40,)
    #centroid_residuals = test_sim["centroid_residuals"]   # (40,)
    true_params        = test_sim["true_params"]          # None for real data, dict for simulations
    is_real_data = true_params is None
    dm_header = test_sim["metadata"]["dm_header"] if is_real_data else None

    x0 = torch.cat([portrait.flatten(), flux, centroids,
                     # mean_profile removed: spectral evolution contaminates profile shape
                     # centroid_residuals removed: scatter_branch dropped from architecture
                     ], dim=0).unsqueeze(0).to(device)


    # shape: (1, 1, 32, 1024)


    num_samples = args.num_samples
    REJECTION_TIMEOUT = 20  # seconds before giving up on rejection sampling

    # Prior bounds in theta space, from the config the posterior was trained with.
    prior_low  = torch.from_numpy(cfg.prior_low)
    prior_high = torch.from_numpy(cfg.prior_high)

    # Try rejection sampling first (fast for well-conditioned posteriors).
    # If it stalls (low acceptance rate, common when posterior concentrates near a prior boundary),
    # fall back to masked direct sampling: draw a large batch from the flow and discard samples
    # outside prior bounds. This is fast (~seconds) and enforces prior support without MCMC overhead.
    _result = [None]
    _error  = [None]

    def _rejection_worker():
        try:
            _result[0] = posterior.sample(
                (num_samples,),
                x=x0,
                max_sampling_batch_size=50000,
                show_progress_bars=True,
            )
        except Exception as e:
            _error[0] = e

    print("Attempting rejection sampling ...")
    _thread = threading.Thread(target=_rejection_worker, daemon=True)
    _thread.start()
    _thread.join(timeout=REJECTION_TIMEOUT)

    if _thread.is_alive():
        print(f"\nRejection sampling did not complete within {REJECTION_TIMEOUT}s. "
              "Switching to masked direct sampling ...")
        n_draw = num_samples * 200  # oversample; acceptance fraction is diagnostic
        with torch.no_grad():
            raw = posterior.posterior_estimator.sample((n_draw,), condition=x0)
        raw = raw.reshape(n_draw, -1).cpu()
        in_bounds = ((raw >= prior_low) & (raw <= prior_high)).all(dim=1)
        accepted = raw[in_bounds]
        acceptance_pct = 100.0 * in_bounds.float().mean().item()
        print(f"  Prior acceptance: {acceptance_pct:.1f}%  ({len(accepted)} / {n_draw} samples kept)")
        if len(accepted) < num_samples:
            print(f"  Warning: only {len(accepted)} in-bounds samples — posterior may sit near a prior boundary.")
        samples_log = accepted[:num_samples]
    elif _error[0] is not None:
        raise _error[0]
    else:
        print("Rejection sampling succeeded.")
        samples_log = _result[0]

    if is_real_data:
        print("Real data: true parameters unknown.")
    else:
        print(f"The true parameters are ({', '.join(cfg.param_names)}):")
        for p in cfg.inferred:
            if p.name in true_params:
                print(f"    {p.name:<16} = {true_params[p.name]:.6g}")

    set_style()

    # Labels, units and the theta -> display conversion all come from the config.
    param_labels = [
        (s["label"] + (f" ({s['unit']})" if s["unit"] else "")) for s in _SPECS
    ]

    samples_np = samples_log.detach().cpu().numpy()

    # Display copy: each column mapped through its config-declared conversion
    # (exp for log params, phase->us for ToA, identity otherwise).
    samples_plot = np.column_stack([
        np.asarray(s["fn"](samples_np[:, i]), dtype=float)
        for i, s in enumerate(_SPECS)
    ])


    def _fmt_for(values):
        """Decimal places from the spread of the samples, not a hardcoded table.

    The old _FMT list was fixed at 6 entries in a fixed order, so it silently
    mis-formatted (or IndexError'd) under any other parameter set.
    """
        spread = np.std(values)
        if spread <= 0 or not np.isfinite(spread):
            return ".4f"
        digits = int(np.clip(2 - np.floor(np.log10(spread)), 1, 6))
        return f".{digits}f"


    _FMT = [_fmt_for(samples_plot[:, i]) for i in range(samples_plot.shape[1])]

    if is_real_data:
        truths      = None
        truths_plot = None
    else:
        # true_params is keyed by the config's parameter names, in linear units.
        _missing = [p.name for p in cfg.inferred if p.name not in true_params]
        if _missing:
            raise SystemExit(
                f"Test simulation {args.test_simulation} is missing true values for "
                f"{_missing}, which config {os.path.basename(cfg.path)} declares as "
                f"inferred. Was it generated with a different config?"
            )
        # theta space first, then through the same display conversion as the samples
        truths      = [p.to_theta(true_params[p.name]) for p in cfg.inferred]
        truths_plot = [float(s["fn"](t)) for s, t in zip(_SPECS, truths)]

    # 0.5th–99.5th percentile ranges to clip extreme flow tails
    ranges_plot = [
        np.quantile(samples_plot[:, i], [0.005, 0.995])
        for i in range(samples_plot.shape[1])
    ]

    # ── Corner plot ───────────────────────────────────────────────────────────────
    ndim = samples_plot.shape[1]
    fig  = plt.figure(figsize=(NAT_W_DOUBLE * 1.15, NAT_W_DOUBLE * 1.15))

    figcorner = corner.corner(
        samples_plot,
        labels=param_labels,
        truths=truths_plot,
        range=ranges_plot,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,          # titles set manually below for better formatting
        color="#1F3A5F",
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        levels=(0.68, 0.95),        # 1σ and 2σ filled contours
        smooth=1.0,
        smooth1d=1.0,
        bins=40,
        truth_color="#C0392B",
        hist_kwargs={"linewidth": 1.2},
        contour_kwargs={"linewidths": 0.8},
        contourf_kwargs={"alpha": 0.45},
        label_kwargs={"fontsize": 9},
        fig=fig,
    )

    # ── Diagonal panel titles: parameter name + median ± 1σ ──────────────────────
    axes_arr = np.array(figcorner.get_axes()).reshape(ndim, ndim)
    for i in range(ndim):
        ax = axes_arr[i, i]
        q16, q50, q84 = np.quantile(samples_plot[:, i], [0.16, 0.5, 0.84])
        err_lo = q50 - q16
        err_hi = q84 - q50
        fmt    = _FMT[i]
        ax.set_title(
            f"{param_labels[i]}\n"
            f"${q50:{fmt}}^{{+{err_hi:{fmt}}}}_{{-{err_lo:{fmt}}}}$",
            fontsize=8, pad=4,
        )

    # ── Apply inward ticks and axis label padding to all corner axes ──────────────
    # Only iterate the lower triangle — upper-triangle axes are invisible spacers
    # and calling set_major_locator on them makes them reappear as blank panels.
    for row in range(ndim):
        for col in range(row + 1):
            ax = axes_arr[row, col]
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.tick_params(direction="in", which="major", top=True, right=True,
                           width=0.5, length=3.5)
            ax.tick_params(direction="in", which="minor", top=True, right=True,
                           width=0.4, length=1.8)
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(15)
                lbl.set_ha("center")
            ax.xaxis.labelpad = 8
            ax.yaxis.labelpad = 6

    figcorner.tight_layout(pad=0.3)
    figcorner.subplots_adjust(bottom=0.09, left=0.09)

    # ── Save ──────────────────────────────────────────────────────────────────────
    sim_name  = os.path.basename(args.test_simulation)
    unique_id = sim_name.split("_")[-1].replace(".pt", "")
    outfile   = f"{outdir}/DM_P0_snr_corner_plot_{unique_id}.png"
    figcorner.savefig(outfile)
    plt.close()
    print(f"Saved corner plot → {outfile}")

    def marginal_map(samples_col):
        if len(samples_col) < 2:
            return float(samples_col[0]) if len(samples_col) == 1 else np.nan
        kde = gaussian_kde(samples_col)
        x   = np.linspace(samples_col.min(), samples_col.max(), 1000)
        return x[np.argmax(kde(x))]

    # Stats in display units (ToA in μs, SNR linear — same as corner plot)
    medians = np.median(samples_plot, axis=0)
    stds    = np.std(samples_plot,    axis=0)
    maps    = np.array([marginal_map(samples_plot[:, i]) for i in range(samples_plot.shape[1])])

    # Index of the DM parameter, by name from the config rather than assuming 0.
    _INFERRED_NAMES = [p.name for p in cfg.inferred]
    _IDX_DM = _INFERRED_NAMES.index(cfg.dm_param) if cfg.dm_param in _INFERRED_NAMES else -1

    # Terminal table uses the plain parameter names; param_labels carries LaTeX
    # for the plots and renders as raw markup in a terminal.
    _table_names = [
        (s["name"] + (f" [{s['unit']}]" if s["unit"] else "")) for s in _SPECS
    ]
    _table_names = [n.replace("$", "").replace("\\mu", "u") for n in _table_names]

    print(f"\n{'Parameter':<28} {'Median':>12} {'MAP':>12} {'Std':>12}")
    print("-" * 68)
    for i, name in enumerate(_table_names):
        if is_real_data:
            print(f"{name:<28} {medians[i]:>12.5f} {maps[i]:>12.5f} {stds[i]:>12.5f}")
            if i == _IDX_DM and dm_header is not None:
                print(f"  {'DM_header':<26} {dm_header:>12.5f} {'':>12} {'':>12} pc/cm³")
                print(f"  {'DM_total (header+offset)':<26} {dm_header+medians[i]:>12.5f} {dm_header+maps[i]:>12.5f} {stds[i]:>12.5f} pc/cm³")
        else:
            truth_display = truths_plot[i]
            print(f"{name:<28} {medians[i]:>12.5f} {maps[i]:>12.5f} {stds[i]:>12.5f}  (Truth: {truth_display:.5f})")


    # figcorner.savefig(f"{outdir}/DM_scattering_time_P0_snr_corner_plot.png")
    # print(f"Saved corner plot to {outdir}/DM_scattering_time_P0_snr_corner_plot.png")


if __name__ == "__main__":
    main()
