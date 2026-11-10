import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as _stats

from .natastro_style import set_style, NAT_W_DOUBLE


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile evolution timeseries: γ_lead, γ_main, amp_ref vs MJD."
    )
    parser.add_argument("--outdir", type=str, required=True,
                        help="Directory containing timeseries.csv")
    parser.add_argument("--include_flagged", action="store_true",
                        help="Include flagged epochs (default: exclude)")
    parser.add_argument("--periodogram", action="store_true",
                        help="Also save a Lomb-Scargle periodogram for each parameter")
    parser.add_argument("--event_mjd", type=float, default=None, nargs="+",
                        help="Mark one or more reference epochs (e.g. a documented event) with "
                             "a vertical dashed line across all panels. Purely a display aid -- "
                             "does not affect the fit.")
    parser.add_argument("--event_label", type=str, default=None, nargs="+",
                        help="Label(s) for --event_mjd, same order/count. Default: unlabeled lines.")
    args = parser.parse_args(argv)
    if args.event_label and (not args.event_mjd or len(args.event_label) != len(args.event_mjd)):
        parser.error("--event_label must match --event_mjd in count if given.")

    csv_path = os.path.join(args.outdir, "timeseries.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"timeseries.csv not found in {args.outdir}. Run timeseries.py first."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} epochs from {csv_path}")

    if not args.include_flagged:
        n_all = len(df)
        df = df[~df["flagged"]].reset_index(drop=True)
        print(f"  Excluded {n_all - len(df)} flagged epochs  →  {len(df)} clean")

    _usetex = set_style()

    # --------------------------------------------------
    # Parameter definitions
    # (col_med, col_std, y-axis label, colour)
    # --------------------------------------------------
    PARAMS = [
        ("gamma_leading", "gamma_leading_std",
         r"$\gamma_{\mathrm{lead}}$" if _usetex else r"$\gamma_{\mathrm{lead}}$",
         "steelblue"),
        ("gamma_main",    "gamma_main_std",
         r"$\gamma_{\mathrm{main}}$" if _usetex else r"$\gamma_{\mathrm{main}}$",
         "darkorange"),
        ("amp_ref",       "amp_ref_std",
         r"$A_{\mathrm{ref}}$"       if _usetex else r"$A_{\mathrm{ref}}$",
         "seagreen"),
    ]

    _mjd_xlabel = "MJD"


    # --------------------------------------------------
    # Weighted linear fit helper
    # --------------------------------------------------
    def weighted_linear_fit(mjd, vals, errs):
        """Fit y = C1*(t - PEPOCH) + C0 by analytical weighted least squares.

    Returns a results dict, or None if fewer than 3 valid points.
    Formal uncertainties on coefficients come from error propagation
    (not from chi² rescaling), so chi²_r diagnoses error-bar calibration
    independently.
    """
        good = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
        if good.sum() < 3:
            return None

        t  = mjd[good]
        y  = vals[good]
        e  = errs[good]
        w  = 1.0 / e**2

        PEPOCH = 0.5 * (t.min() + t.max())
        t_rel  = t - PEPOCH

        # Design matrix [t_rel | 1], solve via normal equations
        A   = np.column_stack([t_rel, np.ones_like(t_rel)])
        cov = np.linalg.inv(A.T @ (w[:, None] * A))
        b   = cov @ (A.T @ (w * y))
        C1, C0   = float(b[0]), float(b[1])
        sig_C1   = float(np.sqrt(cov[0, 0]))
        sig_C0   = float(np.sqrt(cov[1, 1]))

        model     = C1 * t_rel + C0
        residuals = y - model
        chi2r     = float(np.sum((residuals / e)**2) / max(good.sum() - 2, 1))
        rms       = float(np.sqrt(np.mean(residuals**2)))

        return {
            "PEPOCH":       PEPOCH,
            "C0":           C0,    "sig_C0": sig_C0,
            "C1":           C1,    "sig_C1": sig_C1,
            "significance": abs(C1) / sig_C1,
            "chi2r":        chi2r, "rms": rms,
            "good":         good,
            "t":            t,     "t_rel": t_rel,
            "model":        model, "residuals": residuals,
            "n":            int(good.sum()),
        }


    # --------------------------------------------------
    # Run fits and print console summary
    # --------------------------------------------------
    mjd_arr = df["mjd"].values

    print(f"\n{'─'*76}")
    print(f"  Profile evolution  ({len(df)} clean epochs)")
    print(f"{'─'*76}")
    print(f"  {'Parameter':<14} {'C1 (per day)':>16}  {'σ(C1)':>12}  "
          f"{'Signif.':>8}  {'χ²_r':>7}  {'RMS':>10}")
    print(f"{'─'*76}")

    fit_results = {}
    for col_med, col_std, label, color in PARAMS:
        if col_med not in df.columns:
            print(f"  {col_med}: column missing in CSV — skipping")
            continue
        res = weighted_linear_fit(mjd_arr, df[col_med].values, df[col_std].values)
        if res is None:
            print(f"  {label:<14}  too few valid points")
            continue
        fit_results[col_med] = res
        print(f"  {label:<14}  {res['C1']:>+16.4e}  {res['sig_C1']:>12.4e}  "
              f"{res['significance']:>7.1f}σ  {res['chi2r']:>7.2f}  {res['rms']:>10.4f}")

    print(f"{'─'*76}\n")


    # --------------------------------------------------
    # Figure: stacked timeseries panels
    # --------------------------------------------------
    valid_params = [(c, s, l, col) for c, s, l, col in PARAMS if c in fit_results]
    n_panels     = len(valid_params)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(NAT_W_DOUBLE, 1.9 * n_panels + 0.5),
        sharex=True,
        gridspec_kw={"hspace": 0.07},
    )
    if n_panels == 1:
        axes = [axes]

    for _ax_i, (ax, (col_med, col_std, label, color)) in enumerate(zip(axes, valid_params)):
        res  = fit_results[col_med]
        vals = df[col_med].values
        errs = df[col_std].values

        # Data
        ax.errorbar(mjd_arr, vals, yerr=errs,
                    fmt="o", color=color, capsize=2, markersize=3,
                    elinewidth=0.7, zorder=3, label="_nolegend_")

        # Optional reference epoch(s) -- e.g. a documented astrophysical event --
        # marked across every panel, purely for visual context.
        if args.event_mjd:
            for _j, _emjd in enumerate(args.event_mjd):
                ax.axvline(_emjd, color="firebrick", lw=1.0, ls="--", alpha=0.7, zorder=2)
                if args.event_label and _ax_i == 0:
                    ax.text(_emjd, ax.get_ylim()[1], f" {args.event_label[_j]}",
                            fontsize=6.5, color="firebrick", va="top", ha="left",
                            rotation=0, clip_on=False)

        # Linear fit across full span
        # _t_span = np.linspace(mjd_arr.min(), mjd_arr.max(), 500)
        # _fit    = res["C1"] * (_t_span - res["PEPOCH"]) + res["C0"]
        # _sig    = res["significance"]
        # _ls     = "--" if _sig < 2.0 else "-"
        # _lw     = 0.9  if _sig < 2.0 else 1.3

        # if _usetex:
        #     _slope_str = (rf"slope $= {res['C1']:+.3e}$\,d$^{{-1}}$"
        #                   rf"$\quad({_sig:.1f}\sigma)$")
        # else:
        #     _slope_str = f"slope = {res['C1']:+.3e} d⁻¹  ({_sig:.1f}σ)"

        # ax.plot(_t_span, _fit, color=color, lw=_lw, ls=_ls,
        #         alpha=0.85, zorder=4, label=_slope_str)

        # Weighted mean reference line
        ax.axhline(res["C0"], color="k", lw=0.6, ls=":", alpha=0.4, zorder=1)

        ax.set_ylabel(label)
        #ax.legend(fontsize=7, loc="upper right",
                  #frameon=True, framealpha=0.85, edgecolor="0.8")

        # # χ²_r + n annotation
        # if _usetex:
        #     _ann = rf"$\chi^2_r = {res['chi2r']:.2f}$,  $n = {res['n']}$"
        # else:
        #     _ann = f"χ²_r = {res['chi2r']:.2f},  n = {res['n']}"
        # ax.text(0.02, 0.97, _ann, transform=ax.transAxes,
        #         fontsize=6.5, va="top",
        #         bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow",
        #                   ec="0.7", alpha=0.9))

    axes[-1].set_xlabel(_mjd_xlabel)
    fig.tight_layout(pad=0.5)

    out_path = os.path.join(args.outdir, "profile_evolution.png")
    fig.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved  →  {out_path}")


    # --------------------------------------------------
    # Optional Lomb-Scargle periodogram on detrended residuals
    # --------------------------------------------------
    if args.periodogram:
        from astropy.timeseries import LombScargle

        fig_ls, axes_ls = plt.subplots(
            n_panels, 1,
            figsize=(NAT_W_DOUBLE, 2.2 * n_panels),
            sharex=True,
            gridspec_kw={"hspace": 0.08},
        )
        if n_panels == 1:
            axes_ls = [axes_ls]

        for ax_ls, (col_med, col_std, label, color) in zip(axes_ls, valid_params):
            res  = fit_results[col_med]
            errs = df[col_std].values[res["good"]]

            ls       = LombScargle(res["t"], res["residuals"], errs)
            freq, power = ls.autopower(
                minimum_frequency=1.0 / 800.0,
                maximum_frequency=1.0 / 10.0,
                samples_per_peak=10,
            )
            periods  = 1.0 / freq
            fap      = ls.false_alarm_level([0.001, 0.01, 0.1])
            best_per = periods[np.argmax(power)]
            ann_pow  = power[np.argmin(np.abs(periods - 365.25))]

            ax_ls.plot(periods, power, color=color, lw=0.9, zorder=2)

            for fl_label, flv, flc, fls in [
                ("0.1%", fap[0], "crimson",    "--"),
                ("1%",   fap[1], "darkorange", "-."),
                ("10%",  fap[2], "0.55",       ":"),
            ]:
                ax_ls.axhline(flv, color=flc, lw=0.8, ls=fls,
                              label=f"FAP {fl_label}", zorder=1)

            ax_ls.axvline(365.25, color="seagreen", lw=0.8, ls="--",
                          alpha=0.7, label="365 d", zorder=1)
            ax_ls.axvline(best_per, color=color, lw=0.7, ls=":", alpha=0.6, zorder=1)
            ax_ls.text(best_per, power.max(),
                       f" {best_per:.0f} d", fontsize=7, color=color,
                       va="top", clip_on=True)

            ax_ls.set_ylabel(f"{label}\nLS power", fontsize=8)
            ax_ls.set_xscale("log")
            ax_ls.legend(fontsize=6.5, ncol=2, loc="upper left")

            print(f"  LS {col_med}: best period = {best_per:.1f} d, "
                  f"power at 365 d = {ann_pow:.4f}, "
                  f"FAP 1% = {fap[1]:.4f}")

        axes_ls[-1].set_xlabel("Period (days)")
        fig_ls.tight_layout(pad=0.5)
        ls_path = os.path.join(args.outdir, "profile_evolution_periodogram.png")
        fig_ls.savefig(ls_path, dpi=200)
        plt.close()
        print(f"Saved  →  {ls_path}")


if __name__ == "__main__":
    main()
