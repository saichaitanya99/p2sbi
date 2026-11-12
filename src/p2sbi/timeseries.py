import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import scipy.stats as _stats
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin as _argrelmin


from astropy.utils.iers import conf as _iers_conf
_iers_conf.auto_max_age = None

from .natastro_style import set_style, NAT_W_DOUBLE


def marginal_map(samples_col):
    if len(samples_col) < 2:
        return float(samples_col[0]) if len(samples_col) == 1 else np.nan
    kde = gaussian_kde(samples_col)
    x   = np.linspace(samples_col.min(), samples_col.max(), 1000)
    return x[np.argmax(kde(x))]


from .pulsar_config import get_config, resolve_config_path, check_fingerprint


def _idx_of(pname, what):
    if pname not in _INFERRED_NAMES:
        raise SystemExit(
            f"timeseries.py needs '{pname}' ({what}) among the inferred "
            f"parameters of {os.path.basename(cfg.path)}, which declares "
            f"{_INFERRED_NAMES}."
        )
    return _INFERRED_NAMES.index(pname)



# Indices of every log-sampled parameter, so the exp() below is not tied to
# 'log_snr happens to sit at column 4'.

def doppler_dm_correction(mjd_array, dm_ref=None, factor=1.0):
    if dm_ref is None:
        dm_ref = SBI_DM_REF

    try:
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        import astropy.units as u
    except ImportError:
        print("WARNING: astropy not installed — Doppler DM correction skipped (returning zeros).")
        return np.zeros(len(np.atleast_1d(mjd_array)))

    from astropy.coordinates import EarthLocation
    ilofar = EarthLocation.from_geodetic(lon=_GEO["lon_deg"] * u.deg,
                                         lat=_GEO["lat_deg"] * u.deg,
                                         height=_GEO["height_m"] * u.m)
    J2145  = SkyCoord(ra=raj, dec=decj, frame='icrs')
    t      = Time(np.asarray(mjd_array, dtype=float), format='mjd', scale='tdb')
    # radial_velocity_correction returns the topo→bary Doppler velocity;
    # positive = Earth moving toward the source.
    v_corr = J2145.radial_velocity_correction(kind='barycentric', obstime=t,
                                               location=ilofar)
    # astropy: v_corr > 0 when Earth moves TOWARD the source (blueshift convention:
    # v_bary = v_topo + v_corr, so correction is positive when observatory approaches).
    v_r    = v_corr.to(u.km / u.s).value          # km/s, positive = Earth toward
    return factor * dm_ref * v_r / 299792.458      # pc/cm³  (positive → subtract to lower SBI DM)


def solar_elongation_deg(mjd_array):
    """Solar elongation of J2145-0750 in degrees at each MJD."""
    try:
        from astropy.coordinates import SkyCoord, get_sun
        from astropy.time import Time
    except ImportError:
        print("WARNING: astropy not installed — solar elongation skipped (returning NaN).")
        return np.full(len(np.atleast_1d(mjd_array)), np.nan)

    J2145 = SkyCoord(ra=raj, dec=decj, frame='icrs')
    t     = Time(np.asarray(mjd_array, dtype=float), format='mjd', scale='utc')
    sep   = get_sun(t).separation(J2145)
    return np.atleast_1d(sep.deg)


def main(argv=None):
    global SBI_DM_REF, _GEO, _INFERRED_NAMES, cfg, decj, raj
    parser = argparse.ArgumentParser(
        description="Batch inference over real_*.pt files → timeseries CSV + plots."
    )
    parser.add_argument("--outdir",           type=str, required=True,
                        help="Directory containing real_*.pt files from real_archive_loader.py")
    parser.add_argument("--config",           type=str, default=None,
                        help="Required (or set $SBI_PULSAR_CONFIG): the YAML config the "
                             "posterior was trained with. Supplies the pulsar period, "
                             "reference DMs, coordinates and the OBSERVATORY geodetic "
                             "position used for the Doppler correction.")
    parser.add_argument("--posterior_tensor", type=str, required=True,
                        help="Path to trained posterior .pt")
    parser.add_argument("--n_samples",        type=int, default=10000,
                        help="Posterior samples per epoch (default: 10000)")
    parser.add_argument("--just_plot",       action="store_true",
                        help="Skip inference and just plot timeseries.csv if it exists")
    parser.add_argument("--conventional_dm", type=str, default=None,
                        help="Path to conventional timing .dm file (cols: MJD DM error solar_angle) "
                             "for comparison plot")
    parser.add_argument("--offset_method", type=str, default="inv_var",
                        choices=["inv_var", "ml_pairs"],
                        help="How to align SBI and epochwise DM series: "
                             "'inv_var' (default) = difference of separate inverse-variance weighted means; "
                             "'ml_pairs' = ML constant from matched epoch pairs.")
    parser.add_argument("--toa_ref_mhz", type=float, default=None,
                        help="Report toa_ref_us at this frequency (MHz). Default: each "
                             "archive's centre frequency, where it was dedispersed and folded.")
    parser.add_argument("--doppler_dm_factor", type=float, default=1.0,
                        help="Scale factor on the annual Doppler DM correction (see "
                             "doppler_dm_correction()'s docstring): ΔDM_bias = factor × DM_ref × "
                             "v_r/c. Theoretical full bias is factor=2.0; empirically the German "
                             "CNN run saw only 1.0 (the default, preserving existing German "
                             "results). NOT universal across configs/encoders -- regress raw "
                             "(uncorrected) DM against factor=1.0's own correction array to find "
                             "the right value for a new one. Calibrated for J1713/GBT "
                             "(ChannelSetEncoder): 1.904 (from 134 real epochs).")
    parser.add_argument("--force", action="store_true",
                        help="Continue even if the posterior was trained under a config "
                             "with a different fingerprint. Only use deliberately.")
    args = parser.parse_args(argv)
    try:
        cfg = get_config(resolve_config_path(args.config))
    except ValueError as e:
        parser.error(str(e))
    P0_S = cfg.period_s
    SBI_DM_REF  = cfg.dm_header   # DM of the reference epoch used to derive SBI templates
    CONV_DM_REF = cfg.dm_conv     # Reference DM used in conventional timing
    K_DISP      = 4148.808        # μs GHz² (pc/cm³)⁻¹ — dispersion delay constant
    F_TOP_MHZ   = float(np.max(cfg.freqs_mhz))   # the simulator's dispersion reference
    PRIOR_LOW  = cfg.prior_low.astype(float)
    PRIOR_HIGH = cfg.prior_high.astype(float)
    raj  = cfg.raj
    decj = cfg.decj
    _GEO = cfg.geodetic   # lon_deg / lat_deg / height_m
    _INFERRED_NAMES = [p.name for p in cfg.inferred]
    _PARAM_LABELS   = [p.label or p.name for p in cfg.inferred]
    _IDX_DM  = _idx_of(cfg.dm_param,  "dispersion measure")
    _IDX_SNR = _idx_of(cfg.snr_param, "signal-to-noise")
    _IDX_TOA = _idx_of("toa",         "time of arrival")
    _LOG_IDX = [i for i, p in enumerate(cfg.inferred) if p.log]


    if not args.just_plot:
        # --------------------------------------------------
        # Device + posterior
        # --------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # The posterior must come from this config's fingerprint (observatory
        # excluded, so a sibling station's config passes). Otherwise the theta
        # indices and prior bounds above belong to a different parameter set.
        _meta_path = os.path.join(os.path.dirname(args.posterior_tensor), "posterior_meta.pt")
        if os.path.isfile(_meta_path):
            check_fingerprint(cfg, torch.load(_meta_path, map_location="cpu", weights_only=False)
                              .get("config_fingerprint"), what=_meta_path, force=args.force)
        else:
            print(f"WARNING: no posterior_meta.pt beside {args.posterior_tensor}; cannot "
                  f"check it was trained under {os.path.basename(cfg.path)}.")

        posterior = torch.load(args.posterior_tensor, map_location=device, weights_only=False)
        posterior.posterior_estimator = posterior.posterior_estimator.to(device)
        emb = posterior.posterior_estimator.embedding_net
        if hasattr(emb, "_mean"):
            emb._mean = emb._mean.to(device)
        if hasattr(emb, "_std"):
            emb._std = emb._std.to(device)

        # --------------------------------------------------
        # Find all real data .pt files
        # --------------------------------------------------
        pt_files = sorted(glob.glob(os.path.join(args.outdir, "real_*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"No real_*.pt files found in {args.outdir}")
        print(f"\nFound {len(pt_files)} archive(s) to process.\n")

        # --------------------------------------------------
        # Loop over epochs
        # --------------------------------------------------
        records = []

        for pt_file in pt_files:
            fname = os.path.basename(pt_file)
            sim   = torch.load(pt_file, map_location="cpu", weights_only=False)

            mjd       = sim["metadata"]["mjd"]
            dm_header = sim["metadata"]["dm_header"]

            x0 = torch.cat([
                torch.as_tensor(sim["portrait"],           dtype=torch.float32).flatten(),
                torch.as_tensor(sim["flux_per_channel"],   dtype=torch.float32),
                torch.as_tensor(sim["centroids"],          dtype=torch.float32),
                # mean_profile removed: spectral evolution contaminates profile shape
                # torch.as_tensor(sim["centroid_residuals"], dtype=torch.float32),  — scatter_branch removed
            ], dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                samples = posterior.posterior_estimator.sample(
                    (args.n_samples,), condition=x0
                ).reshape(args.n_samples, -1).cpu().numpy()
            if samples.shape[1] != len(_INFERRED_NAMES):
                raise SystemExit(f"The posterior returns {samples.shape[1]} parameters but "
                                 f"{os.path.basename(cfg.path)} infers {len(_INFERRED_NAMES)} "
                                 f"({', '.join(_INFERRED_NAMES)}). Wrong --config for this posterior.")

            # Enforce prior bounds
            param_oob      = (samples < PRIOR_LOW) | (samples > PRIOR_HIGH)   # (n_samples, n_theta)
            in_bounds      = ~param_oob.any(axis=1)
            n_accepted     = int(in_bounds.sum())
            acceptance_pct = 100.0 * n_accepted / args.n_samples
            flagged        = acceptance_pct < 20.0

            samples = samples[in_bounds] if n_accepted > 0 else samples

            for _i in _LOG_IDX:
                samples[:, _i] = np.exp(samples[:, _i])   # log-sampled -> linear

            med = np.median(samples, axis=0)
            std = np.std(samples,    axis=0)
            mp  = np.array([marginal_map(samples[:, i]) for i in range(samples.shape[1])])

            # toa is the arrival at the top channel; move each sample, with its own ΔDM,
            # to the frequency the archive was dedispersed at (Stage F).
            f_ref   = args.toa_ref_mhz or float(sim["metadata"]["center_freq_mhz"])
            toa_ref = (samples[:, _IDX_TOA] * P0_S * 1e6
                       + K_DISP * samples[:, _IDX_DM] * ((1e3 / f_ref)**2 - (1e3 / F_TOP_MHZ)**2))

            flag_str = f"  [FLAGGED — {acceptance_pct:.1f}% in-bounds]" if flagged else ""
            print(f"  MJD {mjd:.5f} | "
                f"DM_total={dm_header+med[_IDX_DM]:.5f} ± {std[_IDX_DM]:.5f} pc/cm³ | "
                f"ToA@{f_ref:.0f}MHz={np.median(toa_ref):.2f} ± {np.std(toa_ref):.2f} μs | "
                f"acceptance={acceptance_pct:.1f}%{flag_str}")

            if flagged:
                PARAM_NAMES = cfg.param_names
                oob_rates   = 100.0 * param_oob.mean(axis=0)
                parts       = "  ".join(f"{n}={r:.1f}%" for n, r in zip(PARAM_NAMES, oob_rates))
                print(f"    Out-of-bounds rates: {parts}")

            record = {
                "filename":              fname,
                "mjd":                   mjd,
                "flagged":               flagged,
                "acceptance_pct":        round(acceptance_pct, 2),
                "center_freq_mhz":       sim["metadata"]["center_freq_mhz"],
                "dm_header":             dm_header,
                "dm_residual":           med[_IDX_DM],
                "dm_residual_map":       mp[_IDX_DM],
                "dm_residual_std":       std[_IDX_DM],
                "dm_total":              dm_header + med[_IDX_DM],
                "dm_total_map":          dm_header + mp[_IDX_DM],
                "dm_total_std":          std[_IDX_DM],
                "toa_phase":             med[_IDX_TOA],
                "toa_phase_map":         mp[_IDX_TOA],
                "toa_phase_std":         std[_IDX_TOA],
                "toa_us":                med[_IDX_TOA] * P0_S * 1e6,
                "toa_us_map":            mp[_IDX_TOA]  * P0_S * 1e6,
                "toa_us_std":            std[_IDX_TOA] * P0_S * 1e6,
                "toa_ref_freq_mhz":      f_ref,
                "toa_ref_us":            np.median(toa_ref),
                "toa_ref_us_map":        marginal_map(toa_ref),
                "toa_ref_us_std":        np.std(toa_ref),
                "corr_toa_dm":           np.corrcoef(samples[:, _IDX_TOA], samples[:, _IDX_DM])[0, 1],
            }
            # Every other inferred parameter, by name (log-sampled ones are already
            # linear here). These used to be med[1]..med[5] by position, which only
            # matched the J2145 layout; for J2145 the column names are unchanged.
            for _i, _name in enumerate(_INFERRED_NAMES):
                if _i not in (_IDX_DM, _IDX_TOA):
                    record[_name]          = med[_i]
                    record[f"{_name}_map"] = mp[_i]
                    record[f"{_name}_std"] = std[_i]
            records.append(record)

        if not records:
            raise RuntimeError("No valid results produced.")

        # --------------------------------------------------
        # CSV
        # --------------------------------------------------
        df = pd.DataFrame(records).sort_values("mjd").reset_index(drop=True)

        # ── Doppler DM correction ─────────────────────────────────────────────────
        _dcorr = doppler_dm_correction(df["mjd"].values, factor=args.doppler_dm_factor)
        df["doppler_corr"]          = _dcorr
        df["dm_total_dcorr"]        = df["dm_total"]        - _dcorr
        df["dm_total_map_dcorr"]    = df["dm_total_map"]     - _dcorr
        df["dm_residual_dcorr"]     = df["dm_residual"]      - _dcorr
        df["dm_residual_map_dcorr"] = df["dm_residual_map"]  - _dcorr

        df["solar_elongation"] = solar_elongation_deg(df["mjd"].values)

        csv_path = os.path.join(args.outdir, "timeseries.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV → {csv_path}")

    if args.just_plot:
        csv_path = os.path.join(args.outdir, "timeseries.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"timeseries.csv not found in {args.outdir}. Run without --just_plot first.")
        df = pd.read_csv(csv_path)
        print(f"Loaded timeseries CSV → {csv_path}")
        _dcorr = doppler_dm_correction(df["mjd"].values, factor=args.doppler_dm_factor)
        df["doppler_corr"]          = _dcorr
        df["dm_total_dcorr"]        = df["dm_total"]        - _dcorr
        df["dm_total_map_dcorr"]    = df["dm_total_map"]     - _dcorr
        df["dm_residual_dcorr"]     = df["dm_residual"]      - _dcorr
        df["dm_residual_map_dcorr"] = df["dm_residual_map"]  - _dcorr
        df.to_csv(csv_path, index=False)
        print(f"Re-saved CSV with doppler_dm_factor={args.doppler_dm_factor} -> {csv_path}")

        if "solar_elongation" not in df.columns:
            df["solar_elongation"] = solar_elongation_deg(df["mjd"].values)
        if "toa_ref_us" not in df.columns:
            print("NOTE: this CSV predates toa_ref_us; plotting toa_us (top channel). "
                  "Rerun without --just_plot for the ToA at the archive's frequency.")

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    _usetex = set_style()

    ok  = df[~df["flagged"]]
    bad = df[ df["flagged"]]

    if _usetex:
        _dm_ylabel  = r"$\Delta\mathrm{DM}$ (pc$\,$cm$^{-3}$)"
        _toa_ylabel = r"ToA residual ($\mu$s)"
        _cmp_ylabel = r"DM $-$ reference (pc$\,$cm$^{{-3}}$)"
        _mjd_xlabel = "MJD"
    else:
        _dm_ylabel  = "ΔDM (pc cm⁻³)"
        _toa_ylabel = "ToA residual (μs)"
        _cmp_ylabel = "DM − reference (pc cm⁻³)"
        _mjd_xlabel = "MJD"

    # 2-panel figure: DM | ToA residuals (no polynomial fit — real timing fits
    # belong in tempo2/PINT against a proper par file, not a quadratic in this
    # script. This only applies the Doppler DM correction and shows the ToAs
    # at their reference frequency as measured.)
    fig, axes = plt.subplots(2, 1, figsize=(NAT_W_DOUBLE, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08})

    def plot_panel(ax, col_med, col_std, ylabel, color, hline=None, label=None):
        ref = SBI_DM_REF if col_med in ("dm_total", "dm_total_dcorr") else 0.0
        if len(ok):
            ax.errorbar(ok["mjd"], ok[col_med] - ref, yerr=ok[col_std],
                        fmt="o", color=color, capsize=2, markersize=3, label=label)
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.8, ls="--")
        ax.set_ylabel(ylabel)
        if label is not None:
            ax.legend(fontsize=6.5, loc="upper right", frameon=False)

    # Panel 1: DM timeseries — SBI ΔDM, Doppler-corrected, no header DM
    _dm_label = f"DM (Doppler factor={args.doppler_dm_factor:.3f})"
    plot_panel(axes[0], "dm_total_dcorr", "dm_total_std", _dm_ylabel, "steelblue", label=_dm_label)

    # ------------------------------------------------------------------
    # Panel 2: ToA residuals at the reference frequency, undetrended.
    # ------------------------------------------------------------------
    if len(ok) > 0:
        _col = "toa_ref_us" if "toa_ref_us" in ok.columns else "toa_us"
        toa_vals     = ok[_col].values.copy()
        toa_err_vals = ok[f"{_col}_std"].values
        mjd_ok_vals  = ok["mjd"].values
        if _col == "toa_ref_us":
            _fr = ok["toa_ref_freq_mhz"]
            _toa_label = (f"ToA at {_fr.median():.0f} MHz" if _fr.max() - _fr.min() < 1.0 else
                          f"ToA at the archive centre ({_fr.min():.0f}–{_fr.max():.0f} MHz)")
        else:
            _toa_label = f"ToA at the top channel ({F_TOP_MHZ:.0f} MHz)"

        # Nothing is excluded from the plot or the stats below -- >3sigma is used
        # ONLY to decide which points get drawn lighter (alpha reduced by 0.2),
        # as a visual flag, not a removal criterion.
        med_toa  = np.median(toa_vals)
        std_toa  = np.std(toa_vals)
        _is_out  = np.abs(toa_vals - med_toa) > 3.0 * std_toa
        n_total  = len(toa_vals)
        n_out    = int(_is_out.sum())
        rms_toa  = np.sqrt(np.mean(toa_vals ** 2))

        print(f"\n  ToA residuals (no fit applied): RMS = {rms_toa:.2f} μs, n={n_total} "
              f"({n_out} drawn lighter, >3σ from median)")

        _ALPHA_REGULAR = 1.0
        _ALPHA_OUTLIER = 0.4
        axes[1].errorbar(mjd_ok_vals[~_is_out], toa_vals[~_is_out], yerr=toa_err_vals[~_is_out],
                         fmt="o", color="darkorange", capsize=2, markersize=3, zorder=3,
                         alpha=_ALPHA_REGULAR, label=_toa_label)
        if n_out > 0:
            axes[1].errorbar(mjd_ok_vals[_is_out], toa_vals[_is_out], yerr=toa_err_vals[_is_out],
                             fmt="o", color="darkorange", capsize=2, markersize=3, zorder=3,
                             alpha=_ALPHA_OUTLIER, label="_nolegend_")
        axes[1].legend(fontsize=6.5, loc="upper right", frameon=False)

        if _usetex:
            _info = rf"RMS $={rms_toa:.2f}$\,$\mu$s$\quad n={n_total}$"
        else:
            _info = f"RMS={rms_toa:.2f} μs    n={n_total}"
        axes[1].text(0.02, 0.97, _info, transform=axes[1].transAxes,
                     fontsize=6.5, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.7", alpha=0.9))


        from astropy.timeseries import LombScargle as _LSG
        _ls_toa = _LSG(mjd_ok_vals, toa_vals, toa_err_vals)
        _freq_toa, _pow_toa = _ls_toa.autopower(
            minimum_frequency=1.0 / 800.0,
            maximum_frequency=1.0 / 10.0,
            samples_per_peak=10,
        )
        _per_toa   = 1.0 / _freq_toa
        _best_toa  = _per_toa[np.argmax(_pow_toa)]
        _fap_toa   = _ls_toa.false_alarm_level([0.001, 0.01, 0.1])
        _ann_pow   = _pow_toa[np.argmin(np.abs(_per_toa - 365.25))]
        print(f"\n  ToA residuals LS periodogram ({_toa_label}):")
        print(f"    Best period       : {_best_toa:.1f} days")
        print(f"    Peak power        : {_pow_toa.max():.4f}")
        print(f"    Power at 365 d    : {_ann_pow:.4f}")
        print(f"    FAP levels        : 0.1%={_fap_toa[0]:.4f}  1%={_fap_toa[1]:.4f}  10%={_fap_toa[2]:.4f}")

        from .natastro_style import NAT_W_DOUBLE as _NWD
        _fig_ls_toa, _ax_ls_toa = plt.subplots(figsize=(_NWD, 2.8))
        _ax_ls_toa.plot(_per_toa, _pow_toa, color="#1F3A5F", lw=0.9, zorder=2)
        _fap_toa_styles = [("0.1%", _fap_toa[0], "crimson",   "--"),
                           ("1%",   _fap_toa[1], "darkorange", "-."),
                           ("10%",  _fap_toa[2], "0.55",       ":")]
        for _fl, _flv, _flc, _fls in _fap_toa_styles:
            _ax_ls_toa.axhline(_flv, color=_flc, lw=0.8, ls=_fls, label=f"FAP {_fl}", zorder=1)
        _ax_ls_toa.axvline(365.25, color="seagreen", lw=0.9, ls="--", alpha=0.7,
                           label="365 d", zorder=1)
        _ax_ls_toa.axvline(_best_toa, color="#1F3A5F", lw=0.7, ls=":", alpha=0.6, zorder=1)
        _ax_ls_toa.text(_best_toa, _pow_toa.max(),
                        f" {_best_toa:.0f} d", fontsize=7, color="#1F3A5F", va="top", clip_on=True)
        _ax_ls_toa.set_title(f"ToA residuals — LS periodogram  ({_toa_label})",
                              fontsize=8, loc="left")
        _ax_ls_toa.set_xlabel("Period (days)")
        _ax_ls_toa.set_ylabel("LS power")
        _ax_ls_toa.set_xscale("log")
        _ax_ls_toa.legend(fontsize=7, loc="upper left", ncol=2)
        _fig_ls_toa.tight_layout(pad=0.5)
        _toa_ls_path = os.path.join(args.outdir, "timeseries_toa_periodogram.png")
        _fig_ls_toa.savefig(_toa_ls_path)
        plt.close()
        print(f"  Saved ToA periodogram → {_toa_ls_path}")

    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel(_toa_ylabel)
    axes[1].set_xlabel(_mjd_xlabel)

    fig.tight_layout(pad=0.5, h_pad=0.5)
    plot_path = os.path.join(args.outdir, "timeseries.png")
    fig.savefig(plot_path)
    plt.close()
    print(f"Saved plot  → {plot_path}")

    # --------------------------------------------------
    # Comparison plot: SBI vs Epochwise DM timeseries
    # --------------------------------------------------
    if args.conventional_dm is not None:
        conv = np.loadtxt(args.conventional_dm)
        if conv.ndim == 1:
            conv = conv[np.newaxis, :]
        conv_mjd = conv[:, 0]
        conv_dm  = conv[:, 1]
        conv_err = conv[:, 2]

        sbi_ok  = df[~df["flagged"]]
        sbi_bad = df[ df["flagged"]]

        # ── Subtract header DMs from both series ─────────────────────────────────
        # SBI: dm_residual_dcorr already = dm_total_dcorr − dm_header (per-epoch)
        sbi_res     = sbi_ok["dm_total_dcorr"].values
        sbi_res_err = sbi_ok["dm_total_std"].values

        # Epochwise: subtract fixed reference DM
        ep_res     = conv_dm - CONV_DM_REF
        ep_res_err = conv_err

        # ── Matching tolerance (used by both offset methods and the diff panel) ─────
        _TOL = 2.0

        # ── Compute offset and align SBI to epochwise ─────────────────────────────
        if args.offset_method == "ml_pairs":
            _sbi_mjd_arr = sbi_ok["mjd"].values
            _pairs = []
            for _j, _em in enumerate(conv_mjd):
                _n = np.argmin(np.abs(_sbi_mjd_arr - _em))
                if np.abs(_sbi_mjd_arr[_n] - _em) <= _TOL:
                    _pairs.append((_sbi_mjd_arr[_n], sbi_res[_n], sbi_res_err[_n],
                                    ep_res[_j], ep_res_err[_j]))
            if len(_pairs) < 2:
                raise RuntimeError(
                    f"ml_pairs: only {len(_pairs)} matched pair(s) within {_TOL} d — "
                    "lower _TOL or use --offset_method inv_var."
                )
            _, _s, _se, _e, _ee = map(np.asarray, zip(*_pairs))
            _w        = 1.0 / (_se ** 2 + _ee ** 2)
            offset    = np.sum(_w * (_e - _s)) / np.sum(_w)
            offset_err = 1.0 / np.sqrt(np.sum(_w))
            _chi2r = np.sum(_w * (_e - _s - offset) ** 2) / (len(_s) - 1)
            print(f"  ML offset  = {offset:+.5e} ± {offset_err:.2e} pc/cm³  "
                  f"(n={len(_s)}, χ²_r={_chi2r:.2f})")
        else:
            sbi_wmean = np.average(sbi_res, weights=1.0 / sbi_res_err ** 2)
            ep_wmean  = np.average(ep_res,  weights=1.0 / ep_res_err  ** 2)
            offset    = ep_wmean - sbi_wmean
            print(f"  inv_var offset = {offset:+.5e} pc/cm³")

        sbi_aligned = sbi_res + offset

        if _usetex:
            _sbi_label  = rf"SBI (offset $= {offset+SBI_DM_REF:+.5f}$ pc cm$^{{-3}}$)"
            _ep_label   = "Epochwise"
            _res_ylabel = r"DM residual (pc cm$^{-3}$)"
        else:
            _sbi_label  = f"SBI (offset = {offset+SBI_DM_REF:+.5f} pc cm⁻³)"
            _ep_label   = "Epochwise"
            _res_ylabel = "DM residual (pc cm⁻³)"

        # ── Epoch-matched difference for residual panel ───────────────────────────
        # For each Epochwise MJD find the nearest SBI epoch within _TOL days
        sbi_mjd_arr = sbi_ok["mjd"].values
        # _match_* arrays are also used by the consistency figure further below
        diff_mjd, diff_val, diff_err = [], [], []
        _match_sbi_vals, _match_ep_vals = [], []
        _match_sbi_errs, _match_ep_errs = [], []
        for j, em in enumerate(conv_mjd):
            if len(sbi_mjd_arr) == 0:
                break
            nearest = np.argmin(np.abs(sbi_mjd_arr - em))
            if np.abs(sbi_mjd_arr[nearest] - em) <= _TOL:
                diff_mjd.append(em)
                diff_val.append(sbi_aligned[nearest] - ep_res[j])
                diff_err.append(np.sqrt(sbi_res_err[nearest] ** 2 + ep_res_err[j] ** 2))
                _match_sbi_vals.append(sbi_aligned[nearest])
                _match_ep_vals.append(ep_res[j])
                _match_sbi_errs.append(sbi_res_err[nearest])
                _match_ep_errs.append(ep_res_err[j])

        diff_mjd = np.array(diff_mjd)
        diff_val = np.array(diff_val)
        diff_err = np.array(diff_err)
        _match_sbi = np.array(_match_sbi_vals)
        _match_ep  = np.array(_match_ep_vals)
        _match_se  = np.array(_match_sbi_errs)
        _match_ee  = np.array(_match_ep_errs)
        rms_diff = float(np.sqrt(np.mean(diff_val ** 2))) if len(diff_val) else float("nan")

        # ── Solar conjunction MJDs: local minima of solar elongation ─────────────
        # Use a daily grid over the full observation span so the minima are
        # accurately located regardless of actual epoch cadence.
        _mjd_min = min(sbi_ok["mjd"].min(), conv_mjd.min())
        _mjd_max = max(sbi_ok["mjd"].max(), conv_mjd.max())
        _mjd_grid = np.arange(_mjd_min - 10, _mjd_max + 10, 1.0)
        _sol_grid = solar_elongation_deg(_mjd_grid)
        # argrelmin with order=50 requires each minimum to be the smallest value
        # within a ±50-day window — robust against day-to-day noise.
        _conj_idx = _argrelmin(_sol_grid, order=50)[0]
        _conj_mjd = _mjd_grid[_conj_idx]
        print(f"  Solar conjunctions in data span: {[f'{m:.1f}' for m in _conj_mjd]}")

        # ── Figure: 4:1 height ratio, shared x-axis ──────────────────────────────
        fig2, (ax2, ax2_res) = plt.subplots(
            2, 1, figsize=(NAT_W_DOUBLE, 4.2), sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.06},
        )

        # Main panel — SBI first (large errorbars), Epochwise on top
        if len(sbi_ok):
            ax2.errorbar(sbi_ok["mjd"], sbi_aligned, yerr=sbi_res_err,
                         fmt="o", color="steelblue", capsize=2, markersize=3,
                         label=_sbi_label, zorder=2)
        ax2.errorbar(conv_mjd, ep_res, yerr=ep_res_err,
                     fmt="s", color="darkorange", capsize=2, markersize=3,
                     label=_ep_label, zorder=3)
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.set_ylabel(_res_ylabel)
        ax2.legend(loc="upper left")

        # Residual panel — SBI − Epochwise at matched epochs
        if len(diff_mjd):
            ax2_res.errorbar(diff_mjd, diff_val, yerr=diff_err,
                             fmt="o", color="steelblue", capsize=2, markersize=2.5,
                             zorder=2)
        ax2_res.axhline(0,        color="k",       lw=0.8, ls="--", zorder=1)
        ax2_res.axhline( rms_diff, color="0.5", lw=0.7, ls=":", zorder=1)
        ax2_res.axhline(-rms_diff, color="0.5", lw=0.7, ls=":", zorder=1)
        ax2_res.set_ylabel("SBI − EW", fontsize=7)
        ax2_res.set_xlabel(_mjd_xlabel)
        if not np.isnan(rms_diff):
            ax2_res.text(0.98, 0.10,
                         f"RMS = {rms_diff:.5f} pc cm⁻³",
                         transform=ax2_res.transAxes, fontsize=6.5,
                         ha="right", va="bottom")

        # Solar conjunction lines on both panels
        for _ci, _cm in enumerate(_conj_mjd):
            ax2.axvline(_cm, color="red", lw=0.8, ls="--", alpha=0.8, zorder=0,
                        label="Solar conjunction" if _ci == 0 else None)
            ax2_res.axvline(_cm, color="red", lw=0.8, ls="--", alpha=0.8, zorder=0)
        if len(_conj_mjd):
            ax2.legend(loc="upper left")   # refresh legend to include conjunction entry

        fig2.tight_layout(pad=0.5)
        cmp_path = os.path.join(args.outdir, "timeseries_dm_comparison.png")
        fig2.savefig(cmp_path)
        plt.close()
        print(f"Saved comparison plot → {cmp_path}")

        # ── (SBI − Epochwise) ΔDM vs solar elongation ────────────────────────────
        if len(diff_mjd):
            diff_sol = solar_elongation_deg(diff_mjd)

            if _usetex:
                _sol_xlabel = "Solar elongation (deg)"
                _sol_ylabel = r"$\Delta\mathrm{DM}_\mathrm{SBI} - \Delta\mathrm{DM}_\mathrm{EW}$ (pc cm$^{-3}$)"
            else:
                _sol_xlabel = "Solar elongation (deg)"
                _sol_ylabel = "ΔDM SBI − ΔDM Epochwise (pc cm⁻³)"

            from .natastro_style import NAT_W_SINGLE
            fig3, ax3 = plt.subplots(figsize=(NAT_W_SINGLE * 1.35, NAT_W_SINGLE * 1.2))

            ax3.errorbar(
                diff_sol, diff_val, yerr=diff_err,
                fmt="o", color="steelblue", capsize=2, markersize=3, zorder=2,
            )
            ax3.axhline(0,        color="k",   lw=0.8, ls="--", zorder=1)
            ax3.axhline( rms_diff, color="0.5", lw=0.7, ls=":",  zorder=1)
            ax3.axhline(-rms_diff, color="0.5", lw=0.7, ls=":",  zorder=1)
            ax3.set_xlabel(_sol_xlabel)
            ax3.set_ylabel(_sol_ylabel)

            fig3.tight_layout(pad=0.5)
            sol_path = os.path.join(args.outdir, "timeseries_dm_solar.png")
            fig3.savefig(sol_path)
            plt.close()
            print(f"Saved solar elongation plot → {sol_path}")

            # ── Lomb-Scargle periodogram of SBI − Epochwise differences ──────────
            from astropy.timeseries import LombScargle

            ls = LombScargle(diff_mjd, diff_val, diff_err)
            # Period range: 10 days → 800 days (covers annual and sub-annual signals)
            freq, power = ls.autopower(
                minimum_frequency=1.0 / 800.0,
                maximum_frequency=1.0 / 10.0,
                samples_per_peak=10,
            )
            periods = 1.0 / freq
            best_period = periods[np.argmax(power)]
            fap_levels = ls.false_alarm_level([0.001, 0.01, 0.1])   # 0.1%, 1%, 10% FAP

            print(f"\nLomb-Scargle periodogram:")
            print(f"  Best period : {best_period:.1f} days")
            print(f"  Peak power  : {power.max():.4f}")
            print(f"  FAP levels  : 0.1%={fap_levels[0]:.4f}  1%={fap_levels[1]:.4f}  10%={fap_levels[2]:.4f}")

            fig4, ax4 = plt.subplots(figsize=(NAT_W_DOUBLE, 2.8))

            ax4.plot(periods, power, color="#1F3A5F", lw=0.9, zorder=2)

            # False alarm probability reference lines
            _fap_styles = [("0.1%", fap_levels[0], "crimson",  "--"),
                           ("1%",   fap_levels[1], "darkorange","-."),
                           ("10%",  fap_levels[2], "0.55",      ":")]
            for label, lvl, col, ls_style in _fap_styles:
                ax4.axhline(lvl, color=col, lw=0.8, ls=ls_style,
                            label=f"FAP {label}", zorder=1)

            # Mark the annual period
            ax4.axvline(365.25, color="seagreen", lw=0.9, ls="--", alpha=0.7,
                        label="365 d", zorder=1)

            # Annotate best period
            ax4.axvline(best_period, color="#1F3A5F", lw=0.7, ls=":", alpha=0.6, zorder=1)
            ax4.text(best_period, ax4.get_ylim()[1] if ax4.get_ylim()[1] > 0 else power.max(),
                     f" {best_period:.0f} d", fontsize=7, color="#1F3A5F",
                     va="top", clip_on=True)

            ax4.set_xlabel("Period (days)")
            ax4.set_ylabel("LS power")
            ax4.set_xscale("log")
            ax4.legend(fontsize=7, loc="upper left", ncol=2)

            fig4.tight_layout(pad=0.5)
            ls_path = os.path.join(args.outdir, "timeseries_dm_periodogram.png")
            fig4.savefig(ls_path)
            plt.close()
            print(f"Saved periodogram → {ls_path}")

            _z   = diff_val / diff_err          # (SBI_aligned - EW) / combined sigma
            _n_m = len(_z)


            def _odr_slope(e, ee, s, se):
                try:
                    from scipy.odr import ODR, Model, RealData
                    _out = ODR(
                        RealData(e, s, sx=ee, sy=se),
                        Model(lambda B, x: B[0] * x + B[1]),
                        beta0=[1.0, 0.0],
                    ).run()
                    return float(_out.beta[0]), float(_out.sd_beta[0]), float(_out.beta[1])
                except Exception:
                    # Weighted LS fallback
                    _ww  = 1.0 / (se ** 2 + ee ** 2)
                    _A   = np.vstack([e, np.ones_like(e)]).T
                    _cov = np.linalg.inv(_A.T @ np.diag(_ww) @ _A)
                    _b   = _cov @ _A.T @ (_ww * s)
                    return float(_b[0]), float(np.sqrt(_cov[0, 0])), float(_b[1])

            _slope, _slope_err, _icept = _odr_slope(_match_ep, _match_ee,
                                                     _match_sbi, _match_se)
            _ks    = _stats.kstest(_z, "norm")
            _sw    = _stats.shapiro(_z)
            _r     = float(np.corrcoef(_match_sbi, _match_ep)[0, 1])
            _rms   = float(np.sqrt(np.mean(diff_val ** 2)))
            _chi2r = float(np.mean(_z ** 2))

            # Print all consistency statistics
            print(f"\n  ── DM consistency  (n={_n_m} matched pairs, tol={_TOL} d) ──")
            print(f"  RMS(SBI − EW)              = {_rms:.4e} pc/cm³")
            print(f"  reduced χ²                 = {_chi2r:.3f} ± {np.sqrt(2/_n_m):.3f}")
            print(f"  norm. residual mean        = {_z.mean():+.3f}  (s.e. {1/np.sqrt(_n_m):.3f})")
            print(f"  norm. residual std         = {_z.std(ddof=1):.3f}  (s.e. {1/np.sqrt(2*_n_m):.3f})")
            print(f"  within 1/2/3 σ             = "
                  f"{np.mean(np.abs(_z)<1)*100:.0f}% / "
                  f"{np.mean(np.abs(_z)<2)*100:.0f}% / "
                  f"{np.mean(np.abs(_z)<3)*100:.0f}%  (expect 68/95/99.7)")
            print(f"  KS vs N(0,1)               : D={_ks.statistic:.3f}, p={_ks.pvalue:.3f}")
            print(f"  Shapiro-Wilk               : W={_sw.statistic:.3f}, p={_sw.pvalue:.3f}")
            print(f"  Pearson r(SBI, EW)         = {_r:.3f}")
            print(f"  ODR slope                  = {_slope:.3f} ± {_slope_err:.3f}  "
                  f"({abs(_slope - 1) / _slope_err:.1f}σ from unity)")
            print(f"  mean σ_SBI / σ_EW          = {_match_se.mean() / _match_ee.mean():.2f}")

            # Labels (usetex vs plain unicode)
            if _usetex:
                _cx  = r"Epoch-wise $\Delta\mathrm{DM}$ (pc cm$^{-3}$)"
                _cy  = r"SBI $\Delta\mathrm{DM}$ (pc cm$^{-3}$)"
                _czx = (r"$(\mathrm{SBI}-\mathrm{EW})\,/\,"
                        r"\sqrt{\sigma_\mathrm{SBI}^2+\sigma_\mathrm{EW}^2}$")
                _cnl = r"$\mathcal{N}(0,1)$"
                _csl = rf"ODR slope $={_slope:.2f}\pm{_slope_err:.2f}$"
                _cst = (rf"mean $={_z.mean():+.2f}$" + "\n"
                        + rf"std $={_z.std(ddof=1):.2f}$" + "\n"
                        + rf"KS $p={_ks.pvalue:.2f}$")
            else:
                _cx  = "Epoch-wise ΔDM (pc cm⁻³)"
                _cy  = "SBI ΔDM (pc cm⁻³)"
                _czx = "(SBI − EW) / sqrt(σ_SBI² + σ_EW²)"
                _cnl = "N(0,1)"
                _csl = f"ODR slope = {_slope:.2f} ± {_slope_err:.2f}"
                _cst = (f"mean = {_z.mean():+.2f}\n"
                        f"std  = {_z.std(ddof=1):.2f}\n"
                        f"KS p = {_ks.pvalue:.2f}")

            # Panel (a): SBI vs Epochwise scatter with 1:1 line and ODR fit
            _fig_cons, (_ac1, _ac2) = plt.subplots(1, 2, figsize=(NAT_W_DOUBLE, 4.0))

            _pad = 0.08 * (max(_match_ep.max(), _match_sbi.max())
                           - min(_match_ep.min(), _match_sbi.min()) + 1e-12)
            _lim = np.array([min(_match_ep.min(), _match_sbi.min()) - _pad,
                             max(_match_ep.max(), _match_sbi.max()) + _pad])
            _ac1.plot(_lim, _lim, "k--", lw=1, zorder=1, label="1:1")
            _ac1.errorbar(
                _match_ep, _match_sbi,
                xerr=_match_ee, yerr=_match_se,
                fmt="o", ms=3.5, color="steelblue",
                ecolor="0.7", elinewidth=0.8, capsize=0, zorder=3,
            )
            _ac1.plot(_lim, _slope * _lim + _icept, "-", color="firebrick",
                      lw=1.3, zorder=2, label=_csl)
            _ac1.set_xlim(*_lim)
            _ac1.set_ylim(*_lim)
            _ac1.set_aspect("equal")
            _ac1.set_xlabel(_cx)
            _ac1.set_ylabel(_cy)
            for lbl in _ac1.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_horizontalalignment("right")
            _ac1.legend(fontsize=8, frameon=False, loc="upper left")
            _ac1.set_title("Epoch-by-epoch agreement", loc="center",
                           fontsize=9, fontweight="bold")

            # Panel (b): histogram of normalised residuals vs N(0,1)
            _ac2.hist(_z, bins=np.linspace(-3.2, 3.2, 17), density=True,
                      color="steelblue", alpha=0.75, edgecolor="w")
            _xg = np.linspace(-3.4, 3.4, 300)
            _ac2.plot(_xg, _stats.norm.pdf(_xg), "k-", lw=1.6, label=_cnl)
            _ac2.axvline(0, color="0.4", ls=":", lw=1)
            _ac2.set_xlabel(_czx)
            _ac2.set_ylabel("Probability density")
            _ac2.legend(fontsize=8, frameon=False, loc="upper left")
            _ac2.set_title("Normalised residuals", loc="center",
                           fontsize=9, fontweight="bold")
            # Stats annotation: mean, std, KS p-value
            _ac2.text(0.97, 0.85, _cst, transform=_ac2.transAxes, ha="right",
                      fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.7"))

            _fig_cons.tight_layout(pad=0.8)
            _cons_path = os.path.join(args.outdir, "timeseries_dm_consistency.png")
            _fig_cons.savefig(_cons_path)
            plt.close()
            print(f"Saved consistency figure → {_cons_path}")


if __name__ == "__main__":
    main()
