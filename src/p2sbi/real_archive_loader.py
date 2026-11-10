import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import matplotlib.pyplot as plt



from .pulsar_config import get_config, resolve_config_path
from .features import build_features

# --------------------------------------------------
# Argument parsing
# --------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Load a PSRCHIVE archive, validate against the training config, "
                    "scrunch if needed, and produce SBI data products."
    )
    parser.add_argument("archive", type=str, help="Path to the .ar / .calib PSRCHIVE file")
    parser.add_argument("--outdir", type=str, default="real_data_outputs",
                        help="Directory for .pt and .png outputs (default: real_data_outputs)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the YAML config describing the instrument the model "
                             "was trained on. Required unless $SBI_PULSAR_CONFIG is set.")
    parser.add_argument("--noplot", action="store_true", help="Skip saving the diagnostic plot")
    parser.add_argument("--verbose", action="store_true", help="Print extra archive metadata")
    args = parser.parse_args(argv)

    try:
        import psrchive
    except ImportError:
        print("ERROR: psrchive Python bindings not installed in this environment.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # --------------------------------------------------
    # Training configuration — read from the SAME config the simulator used.
    #
    # These were hardcoded constants that had to be kept in sync with the simulator
    # by hand. MAIN_PHASE_BIN in particular is trap 3: if the sentinel here and in
    # the simulator diverge, the network meets a centroid value at inference that it
    # never saw in training, and nothing errors.
    # --------------------------------------------------
    cfg = get_config(resolve_config_path(args.config))

    TRAIN_N_CHAN    = cfg.n_chan
    TRAIN_N_BINS    = cfg.n_bins
    TRAIN_FREQ_LOW  = cfg.freq_low_mhz
    TRAIN_FREQ_HIGH = cfg.freq_high_mhz
    FREQ_TOL_MHZ    = cfg.freq_tol_mhz
    MAIN_PHASE_BIN  = cfg.reference_phase_bin     # 297 at 512 bins for J2145

    print(f"\nConfig: {cfg.path}")
    print(f"  {cfg.name} @ {cfg.obs_name}   fingerprint {cfg.fingerprint}")
    print(f"  Expecting {TRAIN_N_CHAN} ch x {TRAIN_N_BINS} bins, "
          f"{TRAIN_FREQ_LOW:.3f}-{TRAIN_FREQ_HIGH:.3f} MHz (tol {FREQ_TOL_MHZ} MHz)")
    print(f"  Dead-channel centroid sentinel: bin {MAIN_PHASE_BIN} "
          f"(reference component '{cfg.reference_component}')")

    # --------------------------------------------------
    # Load archive
    # --------------------------------------------------
    print(f"\nLoading: {args.archive}")
    arch = psrchive.Archive_load(args.archive)

    n_sub       = arch.get_nsubint()
    n_pol       = arch.get_npol()
    n_chan_orig  = arch.get_nchan()
    n_bin        = arch.get_nbin()
    center_freq  = arch.get_centre_frequency()        # MHz
    bw           = arch.get_bandwidth()               # MHz (negative for LSB)
    freq_low_orig  = center_freq - abs(bw) / 2.0
    freq_high_orig = center_freq + abs(bw) / 2.0
    mjd          = arch.start_time().in_days()        # MJD of observation start

    print(f"\n{'─'*52}")
    print(f"  Archive info")
    print(f"{'─'*52}")
    print(f"  Sub-integrations  : {n_sub}")
    print(f"  Polarisations     : {n_pol}")
    print(f"  Channels          : {n_chan_orig}")
    print(f"  Phase bins        : {n_bin}")
    print(f"  MJD               : {mjd:.6f}")
    print(f"  Centre frequency  : {center_freq:.4f} MHz")
    print(f"  Bandwidth         : {bw:.4f} MHz")
    print(f"  Frequency range   : {freq_low_orig:.3f} – {freq_high_orig:.3f} MHz")

    if args.verbose:
        print(f"  Source            : {arch.get_source()}")
        print(f"  Telescope         : {arch.get_telescope()}")
        print(f"  DM (header)       : {arch.get_dispersion_measure():.6f} pc/cm³")

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    print(f"\n{'─'*52}")
    print(f"  Validation  (training target: {TRAIN_N_CHAN} ch × {TRAIN_N_BINS} bins, "
          f"{TRAIN_FREQ_LOW:.0f}–{TRAIN_FREQ_HIGH:.0f} MHz)")
    print(f"{'─'*52}")

    failures = []
    scrunch_factor = 1  # default — updated below if needed

    # 1. Phase bins
    if n_bin != TRAIN_N_BINS:
        failures.append(
            f"Phase bins: archive has {n_bin}, training expects {TRAIN_N_BINS}."
        )
        print(f"  [FAIL] Phase bins : {n_bin}  (expected {TRAIN_N_BINS})")
    else:
        print(f"  [PASS] Phase bins : {n_bin}")

    # 2. Channel count must be an exact integer multiple of TRAIN_N_CHAN
    if n_chan_orig < TRAIN_N_CHAN:
        failures.append(
            f"Archive has {n_chan_orig} channels, fewer than training target {TRAIN_N_CHAN}."
        )
        print(f"  [FAIL] Channels   : {n_chan_orig} < {TRAIN_N_CHAN} (cannot upsample)")
    elif n_chan_orig % TRAIN_N_CHAN != 0:
        failures.append(
            f"Channel count {n_chan_orig} is not divisible by {TRAIN_N_CHAN}. "
            f"Remainder: {n_chan_orig % TRAIN_N_CHAN} channels."
        )
        print(f"  [FAIL] Channels   : {n_chan_orig} is not divisible by {TRAIN_N_CHAN}")
    else:
        scrunch_factor = n_chan_orig // TRAIN_N_CHAN
        if scrunch_factor == 1:
            print(f"  [PASS] Channels   : {n_chan_orig} — already at {TRAIN_N_CHAN}, no scrunch needed")
        else:
            print(f"  [PASS] Channels   : {n_chan_orig} ÷ {scrunch_factor} → {TRAIN_N_CHAN} channels")

    # 3. Low-frequency edge
    low_diff = abs(freq_low_orig - TRAIN_FREQ_LOW)
    if low_diff > FREQ_TOL_MHZ:
        failures.append(
            f"Low-freq edge: archive {freq_low_orig:.3f} MHz vs "
            f"training {TRAIN_FREQ_LOW:.3f} MHz (diff {low_diff:.2f} MHz, tol {FREQ_TOL_MHZ} MHz)."
        )
        print(f"  [FAIL] Low freq   : {freq_low_orig:.3f} MHz  "
              f"(training: {TRAIN_FREQ_LOW:.3f}, diff: {low_diff:.2f} MHz)")
    else:
        print(f"  [PASS] Low freq   : {freq_low_orig:.3f} MHz  (diff: {low_diff:.2f} MHz)")

    # 4. High-frequency edge
    high_diff = abs(freq_high_orig - TRAIN_FREQ_HIGH)
    if high_diff > FREQ_TOL_MHZ:
        failures.append(
            f"High-freq edge: archive {freq_high_orig:.3f} MHz vs "
            f"training {TRAIN_FREQ_HIGH:.3f} MHz (diff {high_diff:.2f} MHz, tol {FREQ_TOL_MHZ} MHz)."
        )
        print(f"  [FAIL] High freq  : {freq_high_orig:.3f} MHz  "
              f"(training: {TRAIN_FREQ_HIGH:.3f}, diff: {high_diff:.2f} MHz)")
    else:
        print(f"  [PASS] High freq  : {freq_high_orig:.3f} MHz  (diff: {high_diff:.2f} MHz)")

    if failures:
        print(f"\n{'─'*52}")
        print("  VALIDATION FAILED — cannot proceed:")
        for f in failures:
            print(f"    • {f}")
        print(f"{'─'*52}\n")
        sys.exit(1)

    print(f"{'─'*52}")
    print("  All checks passed.")

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    print(f"\n{'─'*52}")
    print("  Preprocessing")
    print(f"{'─'*52}")

    if n_sub > 1:
        print(f"  Time-averaging {n_sub} sub-integrations → 1")
        arch.tscrunch()

    if n_pol > 1:
        print(f"  Folding {n_pol} polarisations → Stokes I")
        arch.pscrunch()

    arch.remove_baseline()
    arch.dedisperse()
    print(f"  Baseline removed and archive dedispersed")

    if scrunch_factor > 1:
        print(f"  Frequency scrunching ×{scrunch_factor}: {n_chan_orig} → {TRAIN_N_CHAN} channels")
        arch.fscrunch(scrunch_factor)

    data = arch.get_data()     # (1, 1, TRAIN_N_CHAN, TRAIN_N_BINS)
    print(f"  Data shape after preprocessing: {data.shape}")

    portrait = data[0, 0, :, :]   # (40, 512) numpy array

    # --------------------------------------------------
    # Frequency ordering check
    # Simulations use ASCENDING order: row 0 = lowest freq (112 MHz),
    # row 39 = highest freq (190 MHz).
    # PSRCHIVE / LOFAR archives are often stored in DESCENDING order
    # (channel 0 = highest freq). Flip if needed so the ordering matches training.
    # --------------------------------------------------
    subint    = arch.get_Integration(0)
    freq_ch0  = subint.get_centre_frequency(0)
    freq_chN  = subint.get_centre_frequency(arch.get_nchan() - 1)
    print(f"\n  Frequency ordering check")
    print(f"    Channel 0   : {freq_ch0:.3f} MHz")
    print(f"    Channel {arch.get_nchan()-1:>2d}  : {freq_chN:.3f} MHz")

    if freq_ch0 > freq_chN:
        portrait = portrait[::-1, :].copy()
        print(f"  [FLIP] Descending order detected → flipped to ascending "
              f"(row 0 = {freq_chN:.1f} MHz, row {TRAIN_N_CHAN-1} = {freq_ch0:.1f} MHz)")
    else:
        print(f"  [OK]   Ascending order confirmed — matches training convention")

    # --------------------------------------------------
    # This observation's TRUE channel grid, vs the grid the model was trained on
    #
    # Stations sharing a nominal band still differ in their exact channel centres,
    # and so do epochs at one station. PSRCHIVE dedisperses using the true
    # frequencies, so the residual sweep in the portrait follows them, while the
    # network reads it against the config's grid. The mismatch is a SCALE error on
    # ΔDM — invisible downstream unless it is measured here.
    #
    # The grid is recorded in the .pt so channel_grid_report.py can quantify it
    # across stations and epochs.
    # --------------------------------------------------
    chan_freqs = np.array([subint.get_centre_frequency(i)
                           for i in range(arch.get_nchan())], dtype=float)
    if chan_freqs.size > 1 and chan_freqs[0] > chan_freqs[-1]:
        chan_freqs = chan_freqs[::-1].copy()     # match the portrait's ascending order

    cfg_freqs = cfg.freqs_mhz
    print(f"\n  Channel grid vs config")
    if chan_freqs.size != cfg_freqs.size:
        print(f"  [FAIL] archive has {chan_freqs.size} channels after scrunching but the "
              f"config declares {cfg_freqs.size}.")
        sys.exit(1)
    _dev = chan_freqs - cfg_freqs
    print(f"    max |Δν| vs config : {np.abs(_dev).max():.4f} MHz "
          f"(rms {_dev.std():.4f})")
    if not cfg.has_explicit_freqs:
        print(f"    [NOTE] the config has no channel_freqs_mhz, so the comparison is "
              f"against a uniform linspace. Real LOFAR channels are not evenly spaced; "
              f"run channel_grid_report.py and put the true grid in the config.")

    # Fractional ΔDM scale error this mismatch induces (see channel_grid_report.py)
    _u, _x = 1.0 / chan_freqs**2, 1.0 / cfg_freqs**2
    _xc    = _x - _x.mean()
    _frac  = float(np.dot(_u - _u.mean(), _xc) / np.dot(_xc, _xc)) - 1.0
    _dm_edge = max(abs(p.low) for p in cfg.inferred if p.name == cfg.dm_param)
    print(f"    induced ΔDM scale error: {_frac:+.2e}  "
          f"(|ΔDM| ≤ {abs(_frac)*_dm_edge:.2e} pc/cm³ at the prior edge)")
    if abs(_frac) * _dm_edge > 1e-4:
        print(f"    [WARN] that exceeds a typical 1e-4 pc/cm³ epoch-wise errorbar. This "
              f"observation's channel grid differs enough from the training grid to "
              f"bias DM. See channel_grid_report.py.")

    # --------------------------------------------------
    # Data products
    #
    # Normalisation (trap 4) and the per-channel centroid (trap 3) come from
    # features.py — the SAME function the simulator calls. These used to be a second
    # implementation here, kept in step with sim_generator by hand; that is what
    # CLAUDE.md warns about, and it is now structurally impossible to diverge.
    # --------------------------------------------------
    print(f"\n{'─'*52}")
    print("  Data products")
    print(f"{'─'*52}")
    print(f"  centroid_mode: {cfg.centroid_mode}")

    portrait, flux_per_channel, centroids = build_features(portrait, cfg)

    # Dead / RFI-flagged channels.
    #
    # In 'linear' mode near-zero channel flux drives the centroid ratio out of range
    # (values like -1463 on real data), so that IS the detector. In 'circular' mode
    # atan2 is bounded and features.py already sets a flux-starved channel to offset
    # 0, so out-of-range values cannot occur and the detector has to be the flux
    # itself. Either way the portrait row and flux are zeroed so the channel
    # contributes nothing.
    if cfg.centroid_mode == "linear":
        bad_mask = (centroids < 0.0) | (centroids >= TRAIN_N_BINS)
        _why = f"centroid out of [0, {TRAIN_N_BINS})"
        _fill = f"MAIN_PHASE_BIN = {MAIN_PHASE_BIN}"
    else:
        _tot = np.clip(portrait, 0.0, None).sum(axis=1)
        bad_mask = _tot <= 1e-3 * max(float(_tot.max()), 1e-30)
        _why = "channel flux below 1e-3 of the strongest channel"
        _fill = "centroid offset 0"

    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        bad_idx = np.where(bad_mask)[0].tolist()
        print(f"\n  [WARN] {n_bad} dead/flagged channel(s) — {_why}: {bad_idx}")
        print(f"         Zeroing portrait rows and flux; centroid set to {_fill}")
        portrait[bad_mask, :]      = 0.0
        flux_per_channel[bad_mask] = 0.0
        if cfg.centroid_mode == "linear":
            centroids[bad_mask] = float(MAIN_PHASE_BIN)
        else:
            centroids[bad_mask] = 0.0

    mean_profile = portrait.mean(axis=0)                        # (512,) frequency-averaged profile
    mean_profile = mean_profile / (mean_profile.max() + 1e-12)  # unit-peak normalised — decouples from SNR

    # centroid_residuals removed: scatter_branch dropped from architecture
    # freqs_mhz_train = np.linspace(TRAIN_FREQ_LOW, TRAIN_FREQ_HIGH, TRAIN_N_CHAN)
    # inv_nu2         = 1.0 / freqs_mhz_train**2
    # dm_slope, dm_intercept = np.polyfit(inv_nu2, centroids, 1)
    # centroid_residuals = centroids - (dm_slope * inv_nu2 + dm_intercept)

    print(f"  portrait          {portrait.shape}    min={portrait.min():.3f}  max={portrait.max():.3f}")
    print(f"  flux_per_channel  {flux_per_channel.shape}")
    print(f"  centroids         {centroids.shape}     median={np.median(centroids):.1f} bins  bad_channels={n_bad}")
    print(f"  mean_profile      {mean_profile.shape}")

    # --------------------------------------------------
    # Save .pt   (identical key set to simulation_maker.py;
    # true_params = None since ground truth is unknown)
    # --------------------------------------------------
    timestamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_stem = os.path.splitext(os.path.basename(args.archive))[0]
    pt_name      = f"real_{archive_stem}_{timestamp}.pt"
    pt_path      = os.path.join(args.outdir, pt_name)

    torch.save(
        {
            "portrait":            torch.tensor(portrait,            dtype=torch.float32),
            "flux_per_channel":    torch.tensor(flux_per_channel,    dtype=torch.float32),
            "centroids":           torch.tensor(centroids,           dtype=torch.float32),
            "mean_profile":        torch.tensor(mean_profile,        dtype=torch.float32),
            # "centroid_residuals": removed — scatter_branch dropped from architecture
            "true_params":         None,   # real data — ground truth unknown
            "metadata": {
                "archive":            args.archive,
                "n_chan_orig":        n_chan_orig,
                "scrunch_factor":     scrunch_factor,
                "freq_low_mhz":       freq_low_orig,
                "freq_high_mhz":      freq_high_orig,
                "center_freq_mhz":    center_freq,
                "dm_header":          arch.get_dispersion_measure(),
                "mjd":                mjd,
                "timestamp_utc":      timestamp,
                # Which config this observation was validated and prepared against,
                # so DM_sampling.py / timeseries.py can check it matches the model.
                "config_path":        cfg.path,
                "config_fingerprint": cfg.fingerprint,
                # This observation's TRUE channel centres, ascending, matching the
                # portrait's row order. Consumed by channel_grid_report.py to decide
                # whether stations/epochs can share one trained model.
                "channel_freqs_mhz":  chan_freqs.tolist(),
                "dm_scale_error":     _frac,
            },
        },
        pt_path,
    )
    print(f"\n  Saved .pt  →  {pt_path}")

    # --------------------------------------------------
    # Diagnostic plot
    # --------------------------------------------------
    if not args.noplot:
        png_path = os.path.join(args.outdir, pt_name.replace(".pt", ".png"))

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        freqs_mhz = np.linspace(freq_low_orig, freq_high_orig, TRAIN_N_CHAN)
        # Evenly spaced ticks — the old [0, 10, 20, 30, 39] assumed exactly 40 channels
        tick_idx  = np.unique(
            np.linspace(0, TRAIN_N_CHAN - 1, min(5, TRAIN_N_CHAN)).astype(int)
        ).tolist()
        tick_lbls = [f"{freqs_mhz[i]:.0f}" for i in tick_idx]

        axes[0].imshow(portrait, aspect="auto", origin="lower")
        axes[0].set_title("Portrait (normalised)", fontsize=10)
        axes[0].set_xlabel("Phase bin")
        axes[0].set_yticks(tick_idx)
        axes[0].set_yticklabels(tick_lbls)
        axes[0].set_ylabel("Frequency (MHz)")

        axes[1].plot(flux_per_channel, freqs_mhz, color="steelblue")
        axes[1].set_title("Flux per channel", fontsize=10)
        axes[1].set_xlabel("Mean flux")
        axes[1].set_ylabel("Frequency (MHz)")

        axes[2].plot(centroids, freqs_mhz, color="purple")
        axes[2].set_title("Centroids (DM sweep)", fontsize=10)
        axes[2].set_xlabel("Flux-weighted centroid (phase bin)")
        axes[2].set_ylabel("Frequency (MHz)")

        # Panel 4: mean profile (frequency-averaged) — no ToA reference line for real data
        P0_ms = cfg.period_ms
        phase_ms = np.arange(TRAIN_N_BINS) * (P0_ms / TRAIN_N_BINS)
        axes[3].plot(phase_ms, mean_profile, color="darkorange", lw=1.2)
        axes[3].set_title("Mean profile (freq-averaged)", fontsize=10)
        axes[3].set_xlabel("Phase (ms)")
        axes[3].set_ylabel("Normalised intensity")

        # Panel 4 (old): centroid vs ν^-2 — removed: scatter_branch dropped, centroid_residuals no longer computed
        # centroid_fit_line = dm_slope * inv_nu2 + dm_intercept
        # axes[3].scatter(inv_nu2, centroids, ...)
        # axes[3].plot(inv_nu2, centroid_fit_line, ...)
        # axes[3].scatter(inv_nu2, centroid_residuals + centroid_fit_line.mean(), ...)

        fig.suptitle(
            f"{os.path.basename(args.archive)}\n"
            f"Freq: {freq_low_orig:.1f}–{freq_high_orig:.1f} MHz  |  "
            f"{n_chan_orig} ch ÷ {scrunch_factor} → {TRAIN_N_CHAN} ch  |  {n_bin} bins",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot →  {png_path}")

    print(f"\n{'─'*52}")
    print(f"  Done. Pass the .pt to DM_sampling.py:")
    print(f"    python -m p2sbi.DM_sampling \\")
    print(f"      --test_simulation {pt_path} \\")
    print(f"      --posterior_tensor <path/to/posterior.pt>")
    print(f"{'─'*52}\n")


if __name__ == "__main__":
    main()
