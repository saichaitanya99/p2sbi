import argparse
import multiprocessing as mp
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import torch

from .pulsarDMvary import sim_generator
from .pulsar_config import get_config, recorded_config_path
from .natastro_style import set_style, NAT_W_DOUBLE, NAT_W_SINGLE


# Must be at module level for multiprocessing pickling.
# config_path is a plain string so it pickles into the worker; load_config is
# lru_cached, so each worker parses the YAML once.
def _pp_sim_worker(task):
    seed, config_path = task
    import torch
    torch.set_num_threads(1)
    np.random.seed(seed)
    return next(sim_generator(1, config_path=config_path))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PP (probability-probability) calibration tests for the trained SBI posterior."
    )
    parser.add_argument("--n_test_cases", type=int, default=100,
                        help="Number of independent test simulations to generate (default: 100).")
    parser.add_argument("--n_posterior_samples", type=int, default=2000,
                        help="Posterior samples drawn per test case (default: 2000).")
    parser.add_argument("--n_sims", type=int, default=250000,
                        help="Training-set size used to locate the default posterior path (default: 250000).")
    parser.add_argument("--posterior_tensor", type=str, default=None,
                        help="Path to trained posterior .pt file. Default: SBI_prods_{n_sims}/posterior.pt")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for plots, CSV, and test simulations. Default: same folder as posterior.")
    parser.add_argument("--sample_batch", type=int, default=50,
                        help="Number of test cases to sample on GPU simultaneously (default: 50). Reduce if OOM.")
    parser.add_argument("--just_plot", action="store_true",
                        help="Skip simulation and sampling — load existing pp_test_summary.csv and replot.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the YAML config. Default: the config recorded in "
                             "the posterior's posterior_meta.pt sidecar.")
    args = parser.parse_args(argv)

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    posterior_path = args.posterior_tensor or f"SBI_prods_{args.n_sims}/posterior.pt"
    outdir = args.outdir or os.path.dirname(posterior_path)
    pp_test_dir = os.path.join(outdir, f"PP_test_dir_{args.n_test_cases}")

    os.makedirs(outdir, exist_ok=True)

    # --------------------------------------------------
    # Parameter metadata — from the config the posterior was trained with, so the
    # theta ordering here cannot drift from the ordering the network learned.
    # --------------------------------------------------
    _meta_dir  = os.path.dirname(posterior_path)
    _meta_path = os.path.join(_meta_dir, "posterior_meta.pt")
    _meta = (torch.load(_meta_path, map_location="cpu", weights_only=False)
             if os.path.isfile(_meta_path) else {})
    _config_path = recorded_config_path(_meta, _meta_dir, args.config)

    cfg = get_config(_config_path)
    PARAM_NAMES = cfg.param_names          # e.g. ['DM', ..., 'log_snr', 'amp_ref']
    PARAM_LABELS = cfg.param_labels
    N_PARAMS = len(PARAM_NAMES)
    print(f"Config: {os.path.basename(cfg.path)}  (fingerprint {cfg.fingerprint})")
    print(f"Parameters ({N_PARAMS}): {', '.join(PARAM_NAMES)}")

    if args.just_plot:
        # Load existing CSV — search pp_test_dir first, then outdir
        _csv_candidates = [
            os.path.join(pp_test_dir, "pp_test_summary.csv"),
            os.path.join(outdir,      "pp_test_summary.csv"),
        ]
        _csv_found = next((p for p in _csv_candidates if os.path.isfile(p)), None)
        if _csv_found is None:
            raise FileNotFoundError(
                f"pp_test_summary.csv not found in {pp_test_dir} or {outdir}. "
                "Run without --just_plot first."
            )
        print(f"Loading existing CSV → {_csv_found}")
        _df = pd.read_csv(_csv_found)
        _df = _df[_df["test_case"] != "KS_pvalue"]   # drop the KS summary row
        n_valid = len(_df)
        all_cls = np.column_stack([_df[f"{n}_CL"].values.astype(float) for n in PARAM_NAMES])
        os.makedirs(pp_test_dir, exist_ok=True)
    else:
        if os.path.exists(pp_test_dir):
            print(f"Removing existing {pp_test_dir}")
            shutil.rmtree(pp_test_dir)
        os.makedirs(pp_test_dir)

        # --------------------------------------------------
        # Device + posterior
        # --------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print(f"Loading posterior from {posterior_path}")
        posterior = torch.load(posterior_path, map_location=device, weights_only=False)
        posterior.posterior_estimator = posterior.posterior_estimator.to(device)
        emb = posterior.posterior_estimator.embedding_net
        if hasattr(emb, "_mean"):
            emb._mean = emb._mean.to(device)
        if hasattr(emb, "_std"):
            emb._std = emb._std.to(device)


    def build_x0(sim):
        def t(x):
            return torch.as_tensor(x, dtype=torch.float32)
        return torch.cat([
            t(sim["portrait"]).flatten(),
            t(sim["flux_per_channel"]),
            t(sim["centroids"]),
            # mean_profile removed: spectral evolution contaminates profile shape
            # t(sim["centroid_residuals"]),  — scatter_branch dropped from architecture
        ], dim=0)


    def extract_true_theta(sim):
        """Ground-truth theta, already ordered and log-transformed by the config.

    sim_generator yields this directly now, so there is no per-parameter list
    here to fall out of step with cfg.param_names.
    """
        return np.asarray(sim["theta"], dtype=float)


    if not args.just_plot:
        # --------------------------------------------------
        # Step 1: Generate all test simulations in parallel
        # --------------------------------------------------
        n_cpu = min(os.cpu_count() or 1, args.n_test_cases)
        rng = np.random.default_rng(42)
        seeds = rng.integers(0, 2**31, size=args.n_test_cases).tolist()

        print(f"\nGenerating {args.n_test_cases} test simulations ({n_cpu} workers) ...")
        tasks = [(s, cfg.path) for s in seeds]
        if n_cpu > 1:
            with mp.get_context('fork').Pool(processes=n_cpu) as pool:
                all_sims = list(pool.imap(_pp_sim_worker, tasks, chunksize=10))
        else:
            all_sims = [_pp_sim_worker(t) for t in tasks]
        print(f"Generated {len(all_sims)} simulations.")

        # --------------------------------------------------
        # Step 2: Build batched observation tensor on GPU
        # --------------------------------------------------
        x_batch = torch.stack([build_x0(sim) for sim in all_sims]).to(device)

        # --------------------------------------------------
        # Step 3: Batched GPU sampling in chunks to avoid OOM
        # --------------------------------------------------
        print(f"Sampling {args.n_posterior_samples} × {args.n_test_cases} posteriors "
              f"on {device} (batch={args.sample_batch}) ...")

        theta_chunks = []
        for start in range(0, len(all_sims), args.sample_batch):
            end = min(start + args.sample_batch, len(all_sims))
            with torch.no_grad():
                chunk = posterior.posterior_estimator.sample(
                    (args.n_posterior_samples,),
                    condition=x_batch[start:end],
                ).cpu().numpy()
            theta_chunks.append(chunk)
            print(f"  Sampled {end}/{len(all_sims)} ...", end="\r")
        print()

        theta_batch = np.concatenate(theta_chunks, axis=1)

        # --------------------------------------------------
        # Step 4: Compute credible levels
        # --------------------------------------------------
        records = []
        _cls_list = []
        for i, sim in enumerate(all_sims):
            theta_true    = extract_true_theta(sim)
            theta_samples = theta_batch[:, i, :]
            cls = np.array([
                np.sum(theta_samples[:, j] < theta_true[j]) / args.n_posterior_samples
                for j in range(N_PARAMS)
            ])
            _cls_list.append(cls)
            record = {"test_case": i, "sim_file": f"in_memory_{i}"}
            for j, name in enumerate(PARAM_NAMES):
                record[f"{name}_true"] = float(theta_true[j])
                record[f"{name}_CL"]   = float(cls[j])
            records.append(record)

        n_valid = len(records)
        print(f"\nCompleted {n_valid} test cases.\n")
        all_cls = np.array(_cls_list)   # (n_valid, N_PARAMS)

        # --------------------------------------------------
        # Summary CSV  (last row = KS p-values)
        # --------------------------------------------------
        df = pd.DataFrame(records)
        ks_row = {"test_case": "KS_pvalue", "sim_file": ""}
        for j, name in enumerate(PARAM_NAMES):
            pval = scipy.stats.kstest(all_cls[:, j], "uniform").pvalue
            ks_row[f"{name}_true"] = np.nan
            ks_row[f"{name}_CL"]   = pval
        df = pd.concat([df, pd.DataFrame([ks_row])], ignore_index=True)
        csv_path = os.path.join(outdir, "pp_test_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved summary CSV  →  {csv_path}")

    # --------------------------------------------------
    # Plot helpers
    # --------------------------------------------------
    _usetex = set_style()

    # PARAM_LABELS comes from the config (set above) — the two hardcoded length-6
    # lists that used to live here shadowed it and would silently mislabel every
    # panel under any config with a different parameter set.
    _ci_xlabel   = "Credible interval"
    _ci_ylabel   = "Fraction in C.I."
    _cl_xlabel   = "Credible level"
    _dens_ylabel = "Density"
    _unif_label  = "Uniform"

    COLORS = plt.cm.tab10(np.linspace(0, 0.9, N_PARAMS))

    # Panel grid sized to the parameter count rather than fixed at 2x3.
    _NCOLS = min(3, N_PARAMS)
    _NROWS = int(np.ceil(N_PARAMS / _NCOLS))


    def init_pp_axes(ax, n_posteriors, confidence_intervals=(0.68, 0.95, 0.997)):
        x = np.linspace(0, 1, 1001)
        for ci in confidence_intervals:
            edge = (1.0 - ci) / 2.0
            lo = scipy.stats.binom.ppf(1 - edge, n_posteriors, x) / n_posteriors
            hi = scipy.stats.binom.ppf(edge, n_posteriors, x) / n_posteriors
            lo[0] = 0.0
            hi[0] = 0.0
            ax.fill_between(x, lo, hi, alpha=0.10, color="k")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)


    def add_pp_curve(ax, credible_levels, label, color=None):
        x = np.linspace(0, 1, 1001)
        pp = np.array([np.sum(credible_levels < v) / len(credible_levels) for v in x])
        pval = scipy.stats.kstest(credible_levels, "uniform").pvalue
        p_str = r"$<0.01$" if (_usetex and pval < 0.01) else ("<0.01" if pval < 0.01 else f"{pval:.2f}")
        kw = dict(label=f"{label}  (p={p_str})")
        if color is not None:
            kw["color"] = color
        ax.plot(x, pp, **kw)
        return pval


    # --------------------------------------------------
    # Plot 1 — Individual PP plot for each parameter
    # --------------------------------------------------
    fig, axes = plt.subplots(_NROWS, _NCOLS,
                             figsize=(NAT_W_DOUBLE, 2.5 * _NROWS),
                             squeeze=False)
    axes_flat = axes.flatten()

    for j, (name, label) in enumerate(zip(PARAM_NAMES, PARAM_LABELS)):
        ax = axes_flat[j]
        init_pp_axes(ax, n_valid)
        add_pp_curve(ax, all_cls[:, j], label=label, color=COLORS[j])
        ax.set_title(label, pad=3)
        ax.set_xlabel(_ci_xlabel)
        ax.set_ylabel(_ci_ylabel)
        ax.legend(loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=1.0)
    fig.savefig(os.path.join(outdir, "pp_test_individual.png"))
    plt.close()
    print("Saved pp_test_individual.png")

    # --------------------------------------------------
    # Plot 2 — Combined PP plot (all parameters)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(NAT_W_SINGLE * 1.6, 3.4))
    init_pp_axes(ax, n_valid)

    for j, (name, label) in enumerate(zip(PARAM_NAMES, PARAM_LABELS)):
        add_pp_curve(ax, all_cls[:, j], label=label, color=COLORS[j])

    ax.set_xlabel(_ci_xlabel)
    ax.set_ylabel(_ci_ylabel)
    ax.legend(loc="upper left", ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(outdir, "pp_test_combined.png"))
    plt.close()
    print("Saved pp_test_combined.png")

    # --------------------------------------------------
    # Plot 3 — Credible-level histograms
    # --------------------------------------------------
    fig, axes = plt.subplots(_NROWS, _NCOLS,
                             figsize=(NAT_W_DOUBLE, 2.5 * _NROWS),
                             squeeze=False)
    axes_flat = axes.flatten()

    for j, (name, label) in enumerate(zip(PARAM_NAMES, PARAM_LABELS)):
        ax = axes_flat[j]
        ax.hist(all_cls[:, j], bins=20, range=(0, 1), density=True,
                color=COLORS[j], edgecolor="k", linewidth=0.5, alpha=0.75)
        ax.axhline(1.0, color="k", lw=0.8, ls="--", label=_unif_label)
        pval = scipy.stats.kstest(all_cls[:, j], "uniform").pvalue
        p_str = "<0.01" if pval < 0.01 else f"{pval:.2f}"
        ax.set_title(f"{label}  ($p={p_str}$)" if _usetex else f"{label}  (p={p_str})", pad=3)
        ax.set_xlabel(_cl_xlabel)
        ax.set_ylabel(_dens_ylabel)
        ax.set_xlim(0, 1)
        ax.legend()

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=1.0)
    fig.savefig(os.path.join(outdir, "pp_test_cl_histograms.png"))
    plt.close()
    print("Saved pp_test_cl_histograms.png")

    # --------------------------------------------------
    # Console KS summary
    # --------------------------------------------------
    print("\n--- KS Test Summary ---")
    print(f"{'Parameter':<22} {'p-value':>10}  {'Status':>6}")
    print("-" * 44)
    for j, name in enumerate(PARAM_NAMES):
        pval = scipy.stats.kstest(all_cls[:, j], "uniform").pvalue
        status = "PASS" if pval > 0.01 else "FAIL"
        print(f"{name:<22} {pval:>10.4f}  {status:>6}")

    #Move all the pp_tests* to the pp_test_dir
    for fname in os.listdir(outdir):
        if fname.startswith("pp_test_") and fname.endswith((".png", ".csv")):
            shutil.move(os.path.join(outdir, fname), os.path.join(pp_test_dir, fname))
    print(f"\nMoved all PP test outputs to {pp_test_dir}")


if __name__ == "__main__":
    main()
