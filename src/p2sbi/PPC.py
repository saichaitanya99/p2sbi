import threading
import multiprocessing as mp
import torch
import os
import torch.nn as nn
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
import corner
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .pulsarDMvary import simulate_from_theta
from .pulsar_config import get_config, recorded_config_path
import argparse
import glob
import shutil

from .natastro_style import set_style, NAT_W_SINGLE


def _ppc_sim_worker(task):
    """Re-simulate one posterior sample.

    Was a full duplicate of the nuisance priors, the instrument setup, the
    epoch, sigma_bandpass and the normalisation — a fifth place the same
    numbers lived. Now everything comes from the config, and the portrait
    post-processing is the same _finalise_portrait() the training simulator
    uses, so a PPC can no longer silently test against a different forward
    model than the one that was trained.
    """
    seed, theta, config_path = task
    import torch
    torch.set_num_threads(1)

    portrait, _flux, _cent = simulate_from_theta(
        theta, config_path=config_path, seed=seed
    )
    return portrait   # numpy (n_chan, n_bins)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the PPC checks using a trained SBI model and a test simulation. Only running for 100 samples for quick testing, but can be increased as needed.")
    parser.add_argument('--n_sims', type=int, default=100, help='Number of simulations in the training model.')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of posterior samples to draw for PPC. Default is 100.')
    parser.add_argument('--posterior_tensor', type=str, help='Path to the trained posterior tensor file for sampling. Default is SBI_prods_{n_sims}/posterior.pt')
    parser.add_argument('--test_simulation', type=str, help='Path to the test simulation tensor file for sampling. Default is SBI_prods_{n_sims}/<recent_test_simulation.pt>')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for sampling results. If not set, it will use the same directory as the posterior tensor file.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the YAML config. Default: the config recorded in '
                             "the posterior's posterior_meta.pt sidecar.")
    args = parser.parse_args(argv)

    if args.n_sims is None:
        print("Number of simulations (n_sims) is required. Please provide it using --n_sims.")
        exit(1)

    n_samples = args.n_samples


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    posterior_path = args.posterior_tensor if args.posterior_tensor else f"SBI_prods_{args.n_sims}/posterior.pt"
    dataset_dir = args.outdir if args.outdir else os.path.dirname(posterior_path)
    test_simulation = args.test_simulation if args.test_simulation else max(glob.glob(os.path.join(dataset_dir, "sim*.pt")),key=os.path.getmtime)

    # --------------------------------------------------
    # Config — the forward model used for the predictive check MUST be the one the
    # posterior was trained under, so it defaults to the recorded sidecar.
    # --------------------------------------------------
    _meta_dir  = os.path.dirname(posterior_path)
    _meta_path = os.path.join(_meta_dir, "posterior_meta.pt")
    _meta = (torch.load(_meta_path, map_location="cpu", weights_only=False)
             if os.path.isfile(_meta_path) else {})
    _config_path = recorded_config_path(_meta, _meta_dir, args.config)
    cfg = get_config(_config_path)
    print(f"Config: {os.path.basename(cfg.path)}  (fingerprint {cfg.fingerprint})")

    posterior = torch.load(posterior_path, map_location=device, weights_only=False)
    test_simulation = torch.load(test_simulation, map_location=device, weights_only=False)

    portrait           = test_simulation["portrait"]
    flux               = test_simulation["flux_per_channel"]
    centroids          = test_simulation["centroids"]
    x0 = torch.cat([portrait.flatten(), flux, centroids,
                     # mean_profile removed: spectral evolution contaminates profile shape
                     # centroid_residuals removed: scatter_branch dropped from architecture
                     ], dim=0).unsqueeze(0).to(device)
    # theta_true unused in PPC logic; skipped here to support real data (true_params=None)
    # theta_true = np.array([test_simulation["true_params"]["DM"], ...])

    # print("portrait shape:", portrait.shape)
    # print("theta_true shape:", theta_true.shape)

    REJECTION_TIMEOUT = 10  # seconds before giving up on rejection sampling

    _result = [None]
    _error  = [None]

    def _rejection_worker():
        try:
            _result[0] = posterior.sample(
                (n_samples,),
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
              "Switching to direct neural flow sampling ...")
        with torch.no_grad():
            theta_samples = posterior.posterior_estimator.sample((n_samples,), condition=x0)
        theta_samples = theta_samples.reshape(n_samples, -1).cpu().numpy()
    elif _error[0] is not None:
        raise _error[0]
    else:
        print("Rejection sampling succeeded.")
        theta_samples = _result[0].cpu().numpy()

    #exit(0)
    PPCdir = os.path.join(dataset_dir, "PPC_dir")
    if os.path.exists(PPCdir):
        print(f"Directory {PPCdir} already exists. Removing it and its contents.")
        shutil.rmtree(PPCdir)
    os.makedirs(PPCdir, exist_ok=True)

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=n_samples).tolist()
    worker_args = [(seeds[i], theta_samples[i], cfg.path) for i in range(n_samples)]
    n_cpu = min(os.cpu_count() or 1, n_samples)

    print(f"Generating {n_samples} PPC simulations ({n_cpu} workers) ...")
    if n_cpu > 1:
        with mp.get_context('fork').Pool(processes=n_cpu) as pool:
            portraits_list = list(pool.imap(_ppc_sim_worker, worker_args, chunksize=10))
    else:
        portraits_list = [_ppc_sim_worker(a) for a in worker_args]

    x_preds = [p[np.newaxis] for p in portraits_list]  # list of (1, 40, 512)

    x_preds = np.array(x_preds)  # Shape: (N, 1, 400, 512)
    #print(f"Generated {len(x_preds)} PPC predictions in {dataset_dir}/PPC_dir")

    # Remove channel dimension
    # x_preds: (N, 1, 400, 512) → (N, 400, 512)
    # print(f"x_preds shape before removing channel dimension: {x_preds.shape}")
    x_preds_2d = x_preds[:, 0]

    # x_obs: (1, 400, 512) → (400, 512)
    x_obs_2d = portrait.cpu().numpy()

    # Posterior predictive summaries
    # mean = np.mean(x_preds_2d, axis=0)
    # std  = np.std(x_preds_2d, axis=0)
    residuals = x_preds_2d - x_obs_2d
    mean = np.mean(residuals, axis=0)
    std  = np.std(residuals, axis=0)


    lower = np.quantile(x_preds_2d, 0.025, axis=0)
    upper = np.quantile(x_preds_2d, 0.975, axis=0)

    #residual = mean - x_obs_2d
    zscore   = mean / (std + 1e-8)

    # ---------------------------------------------------
    # Plots
    # ---------------------------------------------------
    _usetex = set_style()
    _PBIN  = "Phase bin"
    _FCHAN = "Frequency channel"
    _INTENS = "Intensity"
    _FIGSIZE = (NAT_W_SINGLE * 1.35, NAT_W_SINGLE * 1.1)

    def _imshow_fig(data, cmap="viridis", clabel=_INTENS):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xlabel(_PBIN)
        ax.set_ylabel(_FCHAN)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(clabel)
        fig.tight_layout(pad=0.5)
        return fig

    # Plot 1: Posterior Predictive overlaid profiles
    fig = _imshow_fig(x_preds_2d[-1])   # last sample as representative
    for i in range(len(x_preds_2d) - 1):
        fig.axes[0].imshow(x_preds_2d[i], aspect="auto", origin="lower", alpha=0.3)
    fig.savefig(os.path.join(dataset_dir, "PPC_dir", "posterior_predictive_profiles.png"))
    plt.close()

    # Plot 2: Observed portrait
    fig = _imshow_fig(x_obs_2d)
    fig.savefig(os.path.join(dataset_dir, "PPC_dir", "observed_portrait.png"))
    plt.close()

    # Plot 3: Residual mean (predicted − observed)
    _rlabel = r"Residual" if not _usetex else r"Residual"
    fig = _imshow_fig(mean, cmap="coolwarm", clabel=_rlabel)
    fig.savefig(os.path.join(dataset_dir, "PPC_dir", "residual_mean_observed.png"))
    plt.close()

    # Plot 4: Z-score map
    _zlabel = r"$z$-score" if _usetex else "z-score"
    fig = _imshow_fig(zscore, cmap="coolwarm", clabel=_zlabel)
    fig.savefig(os.path.join(dataset_dir, "PPC_dir", "posterior_predictive_zscore.png"))
    plt.close()


if __name__ == "__main__":
    main()
