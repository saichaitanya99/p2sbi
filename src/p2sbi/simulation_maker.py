"""
simulation_maker.py
-------------------
Generates ONE test simulation (portrait + .pt + diagnostic plots) from a YAML
config. This is the "ground truth known" observation fed to DM_sampling.py and
PPC.py to check that the trained posterior recovers what it should.

Every parameter defaults to a draw from its prior in the config. Any parameter
can be pinned from the command line: the CLI flags are GENERATED from the
config, so `--DM`, `--gamma_main`, `--conc_main` ... exist because
configs/ilofar_J2145.yaml declares them. Point it at a different config and the
flags follow that config's parameter names.

  conda run -n penv python -m p2sbi.simulation_maker --outdir out --verbose
  conda run -n penv python -m p2sbi.simulation_maker --DM 0.01 --snr 40 --outdir out
  conda run -n penv python -m p2sbi.simulation_maker --justplot --outdir out

Note: the old flags --conc_ref_main / --conc_ref_lead are now --conc_main /
--conc_lead, matching the parameter names in the YAML.
"""

import argparse
import os
import sys
import glob
import textwrap
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pulsar_config import get_config, resolve_config_path
from .pulsarDMvary import (
    PulsarProfileSimulator,
    ProfileComponent,
    build_components,
    _draw_channel_effects,
    _finalise_portrait,
    plot_portrait,
)

# --------------------------------------------------
# Two-stage argument parsing: we need --config before we can generate the
# per-parameter flags, because the flags ARE the config's parameter names.
# --------------------------------------------------


def main(argv=None):
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=None)
    _known, _ = _pre.parse_known_args(argv)
    cfg = get_config(resolve_config_path(_known.config))

    parser = argparse.ArgumentParser(
        description=f"Generate a single test simulation for {cfg.name} "
                    f"@ {cfg.obs_name}. Parameter flags below are generated from "
                    f"{os.path.basename(cfg.path)}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the YAML config")
    parser.add_argument("--mjd", type=float, default=None,
                        help="MJD epoch. Default: the config's simulation.epoch_mjd")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed, for a reproducible test simulation")
    parser.add_argument("--nopt", action="store_true",
                        help="Skip the .pt output, save only plots")
    parser.add_argument("--noplot", action="store_true",
                        help="Skip plots, save only the .pt")
    parser.add_argument("--outdir", type=str, default="sim_outputs",
                        help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the simulation parameters and output paths")
    parser.add_argument("--justplot", action="store_true",
                        help="Load the most recent sim*.pt in --outdir and regenerate "
                             "its diagnostic plots. No new simulation is generated.")
    parser.add_argument("--sim", type=str, default=None,
                        help="Specific .pt to plot (with --justplot). "
                             "Default: most recent sim*.pt in --outdir")

    # ── One flag per config parameter, inferred and nuisance alike ──────────────
    _param_group = parser.add_argument_group(
        "simulation parameters",
        f"Generated from {os.path.basename(cfg.path)}. Unset parameters are drawn "
        f"from their prior."
    )
    _all_params = list(cfg.inferred) + list(cfg.nuisance)
    for _p in _all_params:
        _kind = "inferred" if _p in cfg.inferred else "nuisance"
        _param_group.add_argument(
            f"--{_p.name}", type=float, default=None, dest=f"p_{_p.name}",
            help=f"[{_kind}] prior U[{_p.low:g}, {_p.high:g}]"
                 + (" (log-uniform)" if _p.log else ""),
        )

    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    P0 = cfg.period_ms
    FREQS = cfg.freqs_mhz
    N_CHAN, N_BINS = cfg.n_chan, cfg.n_bins


    def _freq_ticks():
        """Evenly spaced y-tick indices/labels for the frequency axis."""
        idx = np.unique(np.linspace(0, N_CHAN - 1, min(5, N_CHAN)).astype(int)).tolist()
        return idx, [f"{FREQS[i]:.0f}" for i in idx]


    def _diagnostic_figure(portrait, flux, centroids, mean_profile, toa, title):
        """The 4-panel diagnostic: portrait | flux | centroids | mean profile."""
        from .natastro_style import set_style, NAT_W_DOUBLE
        set_style()

        tick_idx, tick_lbls = _freq_ticks()
        fig, ax = plt.subplots(1, 4, figsize=(NAT_W_DOUBLE * 2.2, 3.8))

        # Dead (RFI-zapped) channels: exactly-zero rows, the same signature the
        # archive loader produces. Read from the portrait itself rather than passed
        # in, so a real portrait and a simulated one are marked by the same rule.
        dead   = ~np.any(portrait != 0.0, axis=1)
        n_dead = int(dead.sum())

        im = ax[0].imshow(portrait, aspect="auto", origin="lower", cmap="viridis")
        for i in np.flatnonzero(dead):
            ax[0].axhspan(i - 0.5, i + 0.5, facecolor="0.55", edgecolor="0.25",
                          hatch="////", lw=0.0, zorder=2)
        ax[0].set_title("Portrait (normalised, bandpass included)")
        ax[0].set_xlabel("Phase bin")
        ax[0].set_yticks(tick_idx)
        ax[0].set_yticklabels(tick_lbls)
        ax[0].set_ylabel("Frequency (MHz)")
        fig.colorbar(im, ax=ax[0], pad=0.02, fraction=0.046)

        ax[1].plot(flux, FREQS, color="#1F3A5F", lw=1.2)
        ax[1].set_title("Flux per channel")
        ax[1].set_xlabel("Mean flux")
        ax[1].set_ylabel("Frequency (MHz)")

        for a, vals in ((ax[1], flux), (ax[2], centroids)):
            if n_dead:
                a.scatter(vals[dead], FREQS[dead], marker="x", s=14, color="0.35",
                          zorder=3, label=f"dead ({n_dead})")
                a.legend(loc="lower right", fontsize=6)

        ax[2].plot(centroids, FREQS, color="#8E44AD", lw=1.2)
        ax[2].set_title("Centroids (DM sweep)")
        ax[2].set_xlabel("Flux-weighted centroid (phase bin)")
        ax[2].set_ylabel("Frequency (MHz)")

        phase_ms = np.arange(portrait.shape[1]) * (P0 / portrait.shape[1])
        ax[3].plot(phase_ms, mean_profile, color="#C0392B", lw=1.2)
        ax[3].set_title("Mean profile (freq-averaged)")
        ax[3].set_xlabel("Phase (ms)")
        ax[3].set_ylabel("Normalised intensity")
        if toa is not None and not np.isnan(toa):
            # Reference component's phase, shifted by the ToA offset
            ax[3].axvline((cfg.reference_phase + toa) * P0,
                          color="#E67E22", lw=0.9, ls="--",
                          label=f"ToA = {toa * P0 * 1000:.0f} $\\mu$s")
            ax[3].legend()

        if cfg.dead_channels is not None or n_dead:
            title = f"{title}   |   n_dead = {n_dead}"
        # Wrapped: with ~40 parameters (J1713) one line made the saved figure
        # ~11000 px wide under bbox_inches="tight".
        title = textwrap.fill(title, width=230, break_on_hyphens=False)
        n_lines = title.count("\n") + 1
        fig.suptitle(title, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.022 * n_lines))
        return fig


    def _params_title(values):
        """One-line summary of every parameter, whatever the config declares."""
        return "  ".join(f"{p.name}={values[p.name]:.4g}" for p in _all_params)


    # ============================================================
    # --justplot: regenerate plots from an existing sim*.pt
    # ============================================================
    if args.justplot:
        sim_file = args.sim
        if sim_file is None:
            candidates = glob.glob(os.path.join(args.outdir, "sim*.pt"))
            if not candidates:
                print(f"ERROR: no sim*.pt files found in {args.outdir}")
                sys.exit(1)
            sim_file = max(candidates, key=os.path.getmtime)
        elif not os.path.isfile(sim_file):
            print(f"ERROR: {sim_file} not found")
            sys.exit(1)

        print(f"Plotting existing simulation: {sim_file}")
        data = torch.load(sim_file, map_location="cpu", weights_only=False)

        portrait     = data["portrait"].numpy()
        flux         = data["flux_per_channel"].numpy()
        centroids    = data["centroids"].numpy()
        mean_profile = data.get("mean_profile", torch.zeros(portrait.shape[1])).numpy()

        # true_params is keyed by the config's own parameter names
        tp  = data.get("true_params") or {}
        toa = tp.get("toa")
        base = os.path.splitext(os.path.basename(sim_file))[0]

        if args.verbose and tp:
            print("True parameters from loaded simulation:")
            for k, v in tp.items():
                print(f"  {k:<16} = {v:.6g}")

        # Reconstruct components for the portrait breakdown, if every parameter the
        # config needs was stored. Nothing is invented: if a value is missing we
        # simply skip the component overlay rather than substituting a guess.
        _comps = None
        _missing = [p.name for p in _all_params if p.name not in tp]
        if _missing:
            print(f"  (component overlay skipped — .pt lacks {_missing})")
        else:
            _comps = build_components(cfg, tp)

        try:
            fig_portrait = plot_portrait({"portrait": portrait}, components=_comps)
            portrait_path = os.path.join(args.outdir, f"{base}.png")
            fig_portrait.savefig(portrait_path)
            plt.close(fig_portrait)
            print(f"Portrait plot saved -> {portrait_path}")
        except Exception as e:
            print(f"WARNING: plot_portrait failed ({e}) — skipping portrait plot.")

        fig_diag = _diagnostic_figure(
            portrait, flux, centroids, mean_profile, toa,
            _params_title(tp) if not _missing else base,
        )
        diag_path = os.path.join(args.outdir, f"{base}_diagnostic.png")
        fig_diag.savefig(diag_path)
        plt.close(fig_diag)
        print(f"Diagnostic plot saved -> {diag_path}")
        sys.exit(0)


    # ============================================================
    # Generate a new simulation
    # ============================================================
    if args.seed is not None:
        np.random.seed(args.seed)

    # Draw every parameter from its prior, then apply any CLI overrides.
    # Draw first so a pinned parameter does not shift the RNG stream for the rest.
    values = cfg.sample()
    overridden = []
    for p in _all_params:
        v = getattr(args, f"p_{p.name}")
        if v is not None:
            values[p.name] = float(v)
            overridden.append(p.name)

    mjd = args.mjd if args.mjd is not None else cfg.epoch_mjd

    if args.verbose:
        print(f"Simulating {cfg.name} @ {cfg.obs_name}")
        print(f"  config      : {os.path.basename(cfg.path)}  ({cfg.fingerprint})")
        print(f"  P0          : {P0:.6f} ms")
        print(f"  instrument  : {N_CHAN} ch x {N_BINS} bins  "
              f"({cfg.freq_low_mhz:.2f}-{cfg.freq_high_mhz:.2f} MHz)")
        print(f"  MJD epoch   : {mjd}")
        if overridden:
            print(f"  pinned      : {', '.join(overridden)}")
        print("  parameters  :")
        for p in _all_params:
            tag = " (pinned)" if p.name in overridden else ""
            print(f"    {p.name:<16} = {values[p.name]:.6g}{tag}")

    sim        = PulsarProfileSimulator.from_config(cfg)
    components = build_components(cfg, values)
    scattering = cfg.scattering_values(values)
    chan_fx    = _draw_channel_effects(cfg, values, sim)     # Stage S; Nones when off

    result = sim.generate_portrait(
        components,
        epoch_mjd        = mjd,
        dm               = values[cfg.dm_param],
        tau_scatter_ms   = scattering["tau_ms"],
        scattering_index = scattering["spectral_index"],
        dm_smearing      = cfg.dm_smearing,
        add_noise        = cfg.add_noise,
        snr              = values[cfg.snr_param],
        noise_model      = cfg.noise_model,
        npol             = cfg.npol,
        t_obs_s          = cfg.t_obs_s,
        **chan_fx,
    )

    # Same post-processing as the training simulator — shared function, so traps 3
    # and 4 (centroid sentinel, max-normalisation) cannot diverge from training.
    portrait, flux, centroids, mean_profile = _finalise_portrait(result["portrait"], cfg)

    # --------------------------------------------------
    # Naming
    # --------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid       = uuid.uuid4().hex[:6]
    base_name = (f"sim_{cfg.name}_P0-{P0:.3f}"
                 f"_DM-{values[cfg.dm_param] + cfg.dm_header:.4f}_{uid}")
    pt_path   = os.path.join(args.outdir, f"{base_name}.pt")
    png_path  = os.path.join(args.outdir, f"{base_name}.png")

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    if not args.noplot:
        fig = plot_portrait(result, title=f"{cfg.name} Simulation", components=components)
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig_diag = _diagnostic_figure(
            portrait, flux, centroids, mean_profile,
            values.get("toa"), _params_title(values),
        )
        diag_path = os.path.join(args.outdir, f"{base_name}_diagnostic.png")
        fig_diag.savefig(diag_path, bbox_inches="tight")
        plt.close(fig_diag)

    # --------------------------------------------------
    # Save tensor + parameters
    #
    # true_params holds EVERY parameter the config declares, keyed by its own name.
    # The old version stored a fixed six plus four renamed concentrations, so
    # --justplot had to guess at values it did not save.
    # --------------------------------------------------
    if not args.nopt:
        torch.save(
            {
                "portrait":         torch.tensor(portrait,     dtype=torch.float32),
                "portrait_clean":   torch.tensor(result["portrait_clean"], dtype=torch.float32),
                "components":       torch.tensor(np.asarray(result["component_portraits"]),
                                                 dtype=torch.float32),
                "flux_per_channel": torch.tensor(flux,         dtype=torch.float32),
                "centroids":        torch.tensor(centroids,    dtype=torch.float32),
                "mean_profile":     torch.tensor(mean_profile, dtype=torch.float32),
                "true_params":      {p.name: float(values[p.name]) for p in _all_params},
                "true_theta":       cfg.theta_vector(values),   # ordered, log applied
                "param_names":      cfg.param_names,
                "metadata": {
                    "timestamp_utc":      timestamp,
                    "uid":                uid,
                    "mjd":                mjd,
                    "seed":               args.seed,
                    "config_path":        cfg.path,
                    "config_fingerprint": cfg.fingerprint,
                    "pinned_params":      overridden,
                },
            },
            pt_path,
        )

    if args.verbose:
        print("\nSaved simulation:")
        if not args.nopt:
            print(f"  {pt_path}")
        if not args.noplot:
            print(f"  {png_path}")


if __name__ == "__main__":
    main()
