#!/usr/bin/env python3
"""
j2145_toa_diagnostics.py — diagnose the excess ToA scatter in timeseries.csv
 
Context
-------
The per-epoch SBI pipeline reports a ToA (phase offset) and its posterior
std for each of ~56 real J2145-0750 I-LOFAR epochs. The empirical scatter
of ToA medians across epochs is larger than the model's own reported
per-epoch uncertainty would predict, and a weighted polynomial (secular /
F1-like) fit in time does not explain it. This script runs a battery of
diagnostics on timeseries.csv to help distinguish between the live
hypotheses:
 
  (a) marginalised nuisance shape parameters (kappa_main/lead/premain/inter,
      amp_premain/inter) biasing ToA via an unmodelled degeneracy,
  (b) intrinsic per-epoch scatter (jitter, real small profile changes),
  (c) model misspecification / real archive systematics not covered by the
      training prior (RFI, bandpass, low acceptance epochs),
  (d) a genuine secular timing-model (F0/F1) signal,
  (e) an environmental systematic (solar elongation / solar wind) leaking
      into ToA the way it already does into DM.
 
Nothing here retrains the network or touches the simulator. It only
re-analyses timeseries.csv. Run it, read the printed report, and look at
the saved plots in --outdir before deciding which of the above to chase.
 
Usage
-----
    python j2145_toa_diagnostics.py --csv timeseries.csv --outdir diag_out
 
Optional (only meaningful once you have the real par file):
    --pepoch <MJD>            reference epoch for polynomial/F1 fits
                               (default: mean MJD of the clean sample —
                               a model-free choice, fine for detecting
                               curvature, NOT for absolute F0/F1 values)
    --f0 <Hz>                 spin frequency, used only for the rough,
                               heavily-caveated Delta-F1 estimate
    --orbital-period-days <P> mark this period (and its low harmonics) on
                               the periodograms
 
Dependencies: numpy, pandas, scipy, matplotlib. astropy is optional
(used for Lomb-Scargle); if missing, that section is skipped with a
warning, matching the fallback style already used in the project's own
timeseries.py.
"""
 
import argparse
import os
import sys
 
import numpy as np
import pandas as pd
from scipy import stats
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
try:
    from astropy.timeseries import LombScargle
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def hr(title=""):
    line = "─" * 78
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)
 
 
def load_data(csv_path):
    df = pd.read_csv(csv_path)
 
    # 'flagged' sometimes round-trips as the strings "True"/"False" rather
    # than a real bool dtype, depending on how the CSV was written/read.
    if df["flagged"].dtype != bool:
        df["flagged"] = df["flagged"].astype(str).str.strip().str.lower() == "true"

    if "toa_ref_us" in df.columns:
        df["toa_us"], df["toa_us_std"] = df["toa_ref_us"], df["toa_ref_us_std"]
        print(f"Using toa_ref_us, the ToA at {df.toa_ref_freq_mhz.median():.0f} MHz.")
    else:
        print("This CSV has no toa_ref_us; using toa_us (top channel).")

    df = df.sort_values("mjd").reset_index(drop=True)
    return df
 
 
def clip_3sigma(x):
    """Return a boolean mask keeping points within 3 sigma of the median."""
    med = np.median(x)
    std = np.std(x)
    if std == 0:
        return np.ones_like(x, dtype=bool)
    return np.abs(x - med) <= 3.0 * std
 
 
def weighted_ols(x_powers, y, w):
    """Weighted least squares. x_powers: (n, k) design matrix. Returns beta, cov, chi2."""
    X = x_powers
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ (XtW @ y)
    resid = y - X @ beta
    chi2 = np.sum(w * resid ** 2)
    return beta, cov, chi2, resid
 
 
def lrt_pvalue(chi2_simple, chi2_complex, dof_diff):
    """Likelihood-ratio-style test: chi2 improvement ~ chi2(dof_diff) under H0."""
    delta = chi2_simple - chi2_complex
    if delta < 0 or dof_diff <= 0:
        return delta, np.nan
    p = stats.chi2.sf(delta, dof_diff)
    return delta, p
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Flag / outlier bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
 
def section_flags_and_outliers(df, top_n, outdir):
    hr("1. Flag summary and outlier ranking")
 
    n_total = len(df)
    n_flagged = int(df["flagged"].sum())
    print(f"  Total epochs        : {n_total}")
    print(f"  Flagged (low accept): {n_flagged}")
    if n_flagged:
        print(f"\n  Flagged epochs:")
        cols = ["filename", "mjd", "acceptance_pct", "toa_us", "toa_us_std"]
        print(df.loc[df["flagged"], cols].to_string(index=False))
 
    ok = df[~df["flagged"]].copy()
    mask = clip_3sigma(ok["toa_us"].values)
    ok = ok[mask].copy()
    ok["toa_resid"] = ok["toa_us"] - np.average(ok["toa_us"], weights=1.0 / ok["toa_us_std"] ** 2)
    ok["abs_pull"] = np.abs(ok["toa_resid"]) / ok["toa_us_std"]
 
    print(f"\n  Top {top_n} epochs by |ToA residual| (largest first):")
    cols = ["filename", "mjd", "toa_resid", "toa_us_std", "abs_pull", "acceptance_pct", "snr"]
    print(ok.reindex(ok["toa_resid"].abs().sort_values(ascending=False).index)[cols]
          .head(top_n).to_string(index=False))
 
    print(f"\n  Top {top_n} epochs by toa_us_std (least precise):")
    print(ok.sort_values("toa_us_std", ascending=False)[cols].head(top_n).to_string(index=False))
 
    print(f"\n  Top {top_n} epochs by absolute pull (residual / own sigma — most anomalous \n"
          f"  *given their own claimed precision*; inspect these archives first):")
    print(ok.sort_values("abs_pull", ascending=False)[cols].head(top_n).to_string(index=False))
 
    print("\n  NOTE: this script only sees summary statistics (median/MAP/std per epoch).\n"
          "  It cannot check within-epoch posterior correlations (e.g. a ToA-DM\n"
          "  degeneracy visible in a single epoch's corner plot) — that needs the raw\n"
          "  posterior samples, not this CSV. Pull up the corner plots for the epochs\n"
          "  ranked above and check that by eye as a complementary, free diagnostic.")
 
    return ok  # cleaned, unflagged, with toa_resid/abs_pull columns
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Uncertainty calibration (reduced chi-square)
# ─────────────────────────────────────────────────────────────────────────────
 
def section_calibration(ok, outdir):
    hr("2. Uncertainty calibration: reduced chi-square vs reported sigma")
 
    def report(name, resid, sigma):
        n = len(resid)
        rms = np.sqrt(np.mean(resid ** 2))
        chi2 = np.sum((resid / sigma) ** 2)
        dof = n - 1
        red_chi2 = chi2 / dof
        print(f"  {name}")
        print(f"    n epochs                 : {n}")
        print(f"    RMS of residual          : {rms:.4g}")
        print(f"    median reported sigma    : {np.median(sigma):.4g}")
        print(f"    mean reported sigma      : {np.mean(sigma):.4g}")
        print(f"    chi2 / dof               : {chi2:.1f} / {dof} = {red_chi2:.3f}")
        print(f"    implied under-dispersion : sigma too small by factor "
              f"{np.sqrt(red_chi2):.2f}x  (if red_chi2 > 1)")
        print()
        return red_chi2
 
    toa_resid = ok["toa_resid"].values
    toa_sigma = ok["toa_us_std"].values
    red_chi2_toa = report("ToA (us)", toa_resid, toa_sigma)
 
    dm_resid = (ok["dm_residual_dcorr"] - np.average(ok["dm_residual_dcorr"],
                weights=1.0 / ok["dm_residual_std"].values ** 2)).values
    dm_sigma = ok["dm_residual_std"].values
    red_chi2_dm = report("DM residual (Doppler-corrected, pc/cm^3)", dm_resid, dm_sigma)
 
    # Pull histogram (should look like a standard normal if calibration is honest)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, resid, sigma, name in [
        (axes[0], toa_resid, toa_sigma, "ToA"),
        (axes[1], dm_resid, dm_sigma, "DM residual"),
    ]:
        pulls = resid / sigma
        x = np.linspace(-4, 4, 200)
        ax.hist(pulls, bins=15, density=True, color="#1F3A5F", alpha=0.75)
        ax.plot(x, stats.norm.pdf(x), "k--", lw=1)
        ax.set_title(f"{name} pulls (std normal if calibrated)")
        ax.set_xlabel("residual / reported sigma")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pull_histograms.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved pull_histograms.png")
 
    return red_chi2_toa, red_chi2_dm
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Weighted polynomial trend fits (secular / F1-like)
# ─────────────────────────────────────────────────────────────────────────────
 
def section_polynomial_trend(ok, colname, sigma_col, label, pepoch, f0_hz, outdir):
    hr(f"3. Weighted polynomial trend fit: {label}")
 
    t = ok["mjd"].values
    y = ok[colname].values
    sigma = ok[sigma_col].values
    w = 1.0 / sigma ** 2
    x = t - pepoch
 
    results = {}
    for order in [0, 1, 2, 3]:
        X = np.vander(x, order + 1, increasing=True)
        beta, cov, chi2, resid = weighted_ols(X, y, w)
        dof = len(y) - (order + 1)
        results[order] = dict(beta=beta, cov=cov, chi2=chi2, dof=dof, resid=resid)
        rms = np.sqrt(np.mean(resid ** 2))
        print(f"  order {order}: chi2={chi2:8.1f}  dof={dof:3d}  "
              f"red_chi2={chi2/dof:6.3f}  rms_resid={rms:8.3f}")
 
    print()
    for lo, hi in [(0, 1), (1, 2), (2, 3)]:
        delta, p = lrt_pvalue(results[lo]["chi2"], results[hi]["chi2"], hi - lo)
        verdict = "SIGNIFICANT" if (not np.isnan(p) and p < 0.01) else "not significant"
        print(f"  order {lo} -> {hi}:  delta_chi2 = {delta:6.2f}  p = {p:.4f}   [{verdict}]")
 
    # Rough, heavily caveated Delta-F1 estimate from the quadratic term
    if f0_hz is not None and colname == "toa_us":
        c_us_per_day2 = results[2]["beta"][2]
        c_sec_per_day2 = c_us_per_day2 * 1e-6
        c_sec_per_s2 = c_sec_per_day2 / (86400.0 ** 2)
        # phase_resid(t) [cycles] = -toa_resid(t)[s] * F0  =>  0.5*dF1*t^2 = -c_sec_per_s2 * t^2 * F0
        delta_f1 = -2.0 * c_sec_per_s2 * f0_hz
        c_err = np.sqrt(results[2]["cov"][2, 2]) * 1e-6 / (86400.0 ** 2)
        delta_f1_err = 2.0 * c_err * f0_hz
        print(f"\n  Rough Delta-F1 estimate from quadratic term: "
              f"{delta_f1:.3e} +/- {delta_f1_err:.3e} Hz/s")
        print("  CAVEAT: this treats the quadratic coefficient as if it were purely a\n"
              "  spin-frequency-derivative signal. It is completely degenerate with any\n"
              "  uncorrected long-term DM trend, red/timing noise, or an incorrect PEPOCH/\n"
              "  reference F0/F1 baked into the archive fold. Do not quote this number\n"
              "  without cross-checking against the actual par file's own F1 uncertainty,\n"
              "  and without first ruling out DM-trend and profile-drift explanations.")
 
    # Plot: data + order-0,1,2 fits
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(t, y, yerr=sigma, fmt="o", color="darkorange", capsize=2, markersize=3, zorder=2)
    xs = np.linspace(x.min(), x.max(), 200)
    for order, style in [(0, "k--"), (1, "steelblue"), (2, "crimson")]:
        beta = results[order]["beta"]
        ys = np.polyval(beta[::-1], xs)
        ax.plot(xs + pepoch, ys, style, lw=1.2, label=f"order {order}", alpha=0.8)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlabel("MJD")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fname = f"trend_fit_{colname}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)
    print(f"\n  Saved {fname}")
 
    return results
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Lomb-Scargle periodograms
# ─────────────────────────────────────────────────────────────────────────────
 
def section_periodogram(ok, colname, sigma_col, label, orbital_period_days, outdir):
    hr(f"4. Lomb-Scargle periodogram: {label}")
 
    if not _HAS_ASTROPY:
        print("  WARNING: astropy not installed — periodogram skipped.")
        return
 
    t = ok["mjd"].values
    y = ok[colname].values
    dy = ok[sigma_col].values
    y = y - np.average(y, weights=1.0 / dy ** 2)
 
    ls = LombScargle(t, y, dy)
    freq, power = ls.autopower(minimum_frequency=1.0 / 800.0, maximum_frequency=1.0 / 3.0,
                                samples_per_peak=10)
    periods = 1.0 / freq
    best_period = periods[np.argmax(power)]
    fap_levels = ls.false_alarm_level([0.001, 0.01, 0.1])
 
    print(f"  Best period : {best_period:.2f} days   (peak power = {power.max():.4f})")
    print(f"  FAP levels  : 0.1%={fap_levels[0]:.4f}  1%={fap_levels[1]:.4f}  10%={fap_levels[2]:.4f}")
    if power.max() > fap_levels[1]:
        print(f"  -> peak clears the 1% false-alarm level. Worth a closer look.")
    else:
        print(f"  -> no peak clears the 1% false-alarm level.")
 
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(periods, power, color="#1F3A5F", lw=0.9)
    for lvl, name, col, ls_style in [
        (fap_levels[0], "0.1%", "crimson", "--"),
        (fap_levels[1], "1%", "darkorange", "-."),
        (fap_levels[2], "10%", "0.55", ":"),
    ]:
        ax.axhline(lvl, color=col, lw=0.8, ls=ls_style, label=f"FAP {name}")
    ax.axvline(365.25, color="seagreen", lw=0.9, ls="--", alpha=0.7, label="365 d")
    if orbital_period_days is not None:
        for n, harm in enumerate([1, 2, 3], start=1):
            ax.axvline(orbital_period_days * harm, color="purple", lw=0.8, ls=":",
                       alpha=0.6, label=f"Porb x{harm}" if n == 1 else None)
    ax.set_xscale("log")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("LS power")
    ax.set_title(label)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fname = f"periodogram_{colname}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Correlation diagnostics
# ─────────────────────────────────────────────────────────────────────────────
 
def section_correlations(ok, target_col, target_label, covariates, outdir):
    hr(f"5. Correlations: {target_label} vs covariates")
 
    y = ok[target_col].values
    rows = []
    for cov in covariates:
        if cov not in ok.columns or cov == target_col:
            continue
        x = ok[cov].values
        if np.std(x) == 0:
            continue
        r, p = stats.pearsonr(x, y)
        rows.append((cov, r, p))
 
    rows.sort(key=lambda r: -abs(r[1]))
    print(f"  {'covariate':20s} {'r':>8s} {'p-value':>10s}")
    for cov, r, p in rows:
        flag = "  <-- p<0.05" if p < 0.05 else ""
        print(f"  {cov:20s} {r:8.3f} {p:10.4f}{flag}")
 
    return rows
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. Multivariate weighted regression
# ─────────────────────────────────────────────────────────────────────────────
 
def section_multivariate(ok, outdir):
    hr("6. Multivariate weighted regression: ToA residual ~ DM + shape params + SNR")
 
    predictors = ["dm_residual_dcorr", "gamma_leading", "gamma_main", "amp_ref", "snr"]
    y = ok["toa_resid"].values
    w = 1.0 / ok["toa_us_std"].values ** 2
 
    X = np.column_stack([ok[p].values for p in predictors])
    X = np.column_stack([np.ones(len(y)), X])  # intercept
    names = ["intercept"] + predictors
 
    beta, cov, chi2, resid = weighted_ols(X, y, w)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), df=len(y) - len(names))
 
    # Weighted R^2 relative to weighted-mean-only model
    y_wmean = np.average(y, weights=w)
    ss_res = np.sum(w * resid ** 2)
    ss_tot = np.sum(w * (y - y_wmean) ** 2)
    r2 = 1 - ss_res / ss_tot
 
    print(f"  Weighted R^2 = {r2:.3f}  (fraction of weighted ToA-residual variance jointly\n"
          f"  explained by DM residual + gamma_leading + gamma_main + amp_ref + SNR)")
    print(f"\n  {'term':20s} {'coef':>12s} {'std err':>12s} {'t':>8s} {'p':>8s}")
    for name, b, s, tv, pv in zip(names, beta, se, tvals, pvals):
        flag = "  <-- p<0.05" if pv < 0.05 else ""
        print(f"  {name:20s} {b:12.4g} {s:12.4g} {tv:8.2f} {pv:8.4f}{flag}")
 
    print("\n  If R^2 here is much bigger than any single pairwise |r| in section 5,\n"
          "  the ToA scatter is being driven by a *combination* of these parameters\n"
          "  rather than any one of them alone — that favours hypothesis (a), the\n"
          "  marginalised-nuisance-parameter degeneracy, since it would predict exactly\n"
          "  this kind of multivariate structure.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. SNR scaling of reported uncertainty
# ─────────────────────────────────────────────────────────────────────────────
 
def section_snr_scaling(ok, outdir):
    hr("7. SNR scaling of reported toa_us_std")
 
    snr = ok["snr"].values
    sigma = ok["toa_us_std"].values
    log_snr = np.log(snr)
    log_sigma = np.log(sigma)
 
    slope, intercept, r, p, se = stats.linregress(log_snr, log_sigma)
    print(f"  log(toa_us_std) = {intercept:.3f} + {slope:.3f} * log(snr)   "
          f"(r={r:.3f}, p={p:.4g}, slope_err={se:.3f})")
    print(f"  Expected slope for pure Fisher/template-fitting noise: close to -1.")
    if slope > -0.6:
        print(f"  -> slope is much flatter than -1: reported uncertainty is not simply\n"
              f"     SNR-limited. Consistent with an unmodelled noise floor (jitter, real\n"
              f"     profile variability, or archive-level systematics) dominating over\n"
              f"     template-fitting precision at higher SNR.")
    else:
        print(f"  -> slope is close to the Fisher/SNR-limited expectation.")
 
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(snr, sigma, s=14, color="#1F3A5F")
    xs = np.linspace(snr.min(), snr.max(), 100)
    ax.plot(xs, np.exp(intercept) * xs ** slope, "crimson", lw=1.2,
            label=f"fit slope={slope:.2f}")
    ax.plot(xs, sigma[np.argmax(snr)] * (xs / snr.max()) ** -1, "k--", lw=1,
            alpha=0.6, label="slope = -1 (Fisher expectation)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel("toa_us_std")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "snr_scaling.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved snr_scaling.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 8. Solar elongation check
# ─────────────────────────────────────────────────────────────────────────────
 
def section_solar(ok, outdir):
    hr("8. Solar elongation check (ToA)")
 
    if "solar_elongation" not in ok.columns:
        print("  'solar_elongation' column not found — skipping.")
        return
 
    elong = ok["solar_elongation"].values
    resid = ok["toa_resid"].values
    r, p = stats.pearsonr(elong, resid)
    print(f"  corr(toa_resid, solar_elongation) = {r:+.3f}  (p={p:.4f})")
 
    near = elong < 30.0
    if near.sum() >= 3 and (~near).sum() >= 3:
        rms_near = np.sqrt(np.mean(resid[near] ** 2))
        rms_far = np.sqrt(np.mean(resid[~near] ** 2))
        print(f"  RMS |ToA residual|, elongation < 30 deg (n={near.sum()}): {rms_near:.1f} us")
        print(f"  RMS |ToA residual|, elongation >= 30 deg (n={(~near).sum()}): {rms_far:.1f} us")
        if rms_near > 1.3 * rms_far:
            print("  -> near-conjunction epochs show noticeably larger scatter. The\n"
                 "     existing pipeline only Doppler-corrects DM, not ToA — a solar-wind\n"
                 "     or barycentring systematic leaking into ToA would show up exactly\n"
                 "     like this.")
    else:
        print("  Not enough epochs on one side of the 30 deg cut to compare usefully.")
 
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(elong, resid, s=14, color="#1F3A5F")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(30, color="crimson", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Solar elongation (deg)")
    ax.set_ylabel("ToA residual (us)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "solar_elongation_toa.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved solar_elongation_toa.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 9. Flagged vs unflagged / low-acceptance comparison
# ─────────────────────────────────────────────────────────────────────────────
 
def section_flagged_comparison(df, outdir, acceptance_threshold=95.0):
    hr("9. Flagged / low-acceptance vs clean-epoch comparison")
 
    borderline = df[(~df["flagged"]) & (df["acceptance_pct"] < acceptance_threshold)]
    clean = df[(~df["flagged"]) & (df["acceptance_pct"] >= acceptance_threshold)]
 
    print(f"  Clean epochs (acceptance >= {acceptance_threshold}%)      : {len(clean)}")
    print(f"  Borderline epochs (unflagged, acceptance < {acceptance_threshold}%): {len(borderline)}")
    print(f"  Flagged epochs                                : {int(df['flagged'].sum())}")
 
    if len(borderline) >= 2:
        toa_c = clean["toa_us"].values
        toa_b = borderline["toa_us"].values
        offset = np.average(clean["toa_us"], weights=1.0 / clean["toa_us_std"].values ** 2)
        rms_clean = np.sqrt(np.mean((toa_c - offset) ** 2))
        rms_border = np.sqrt(np.mean((toa_b - offset) ** 2))
        print(f"\n  RMS ToA residual, clean epochs      : {rms_clean:.1f} us")
        print(f"  RMS ToA residual, borderline epochs : {rms_border:.1f} us")
        if rms_border > 1.5 * rms_clean:
            print("  -> borderline/low-acceptance epochs carry disproportionately more\n"
                  "     scatter. Some of the excess is plausibly model misspecification\n"
                  "     (real data outside what the training prior covers), not a purely\n"
                  "     astrophysical effect.")
        else:
            print("  -> scatter is not obviously concentrated in the low-acceptance\n"
                  "     epochs; the excess is spread through otherwise well-fit epochs too.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=str, required=True, help="Path to timeseries.csv")
    p.add_argument("--outdir", type=str, default="diag_out", help="Output directory for plots")
    p.add_argument("--pepoch", type=float, default=None,
                   help="Reference MJD for polynomial fits. Default: mean MJD of clean sample.")
    p.add_argument("--f0", type=float, default=62.295887837388923127,
                   help="Spin frequency in Hz, used only for the rough Delta-F1 estimate.")
    p.add_argument("--orbital-period-days", type=float, default=None,
                   help="Binary orbital period in days, marked on the periodograms.")
    p.add_argument("--top-n", type=int, default=8, help="Rows to show in outlier rankings.")
    args = p.parse_args()
 
    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.csv)
 
    ok = section_flags_and_outliers(df, args.top_n, args.outdir)
 
    pepoch = args.pepoch if args.pepoch is not None else float(np.mean(ok["mjd"]))
    print(f"\n  Using PEPOCH = {pepoch:.4f} for polynomial fits "
          f"({'user-supplied' if args.pepoch is not None else 'mean MJD of clean sample — NOT a real ephemeris PEPOCH'}).")
 
    section_calibration(ok, args.outdir)
 
    section_polynomial_trend(ok, "toa_us", "toa_us_std", "ToA residual (us)",
                              pepoch, args.f0, args.outdir)
    section_polynomial_trend(ok, "dm_residual_dcorr", "dm_residual_std",
                              "DM residual, Doppler-corrected (pc/cm^3)",
                              pepoch, None, args.outdir)
 
    section_periodogram(ok, "toa_us", "toa_us_std", "ToA residual", args.orbital_period_days, args.outdir)
    section_periodogram(ok, "dm_residual_dcorr", "dm_residual_std", "DM residual",
                         args.orbital_period_days, args.outdir)
 
    covariates = ["snr", "gamma_leading", "gamma_main", "amp_ref", "dm_residual_dcorr",
                  "acceptance_pct", "mjd", "toa_us_std", "solar_elongation"]
    section_correlations(ok, "toa_resid", "ToA residual", covariates, args.outdir)
 
    dm_covariates = ["snr", "gamma_leading", "gamma_main", "amp_ref", "toa_resid",
                      "acceptance_pct", "mjd", "dm_residual_std", "solar_elongation"]
    ok["dm_resid_tmp"] = ok["dm_residual_dcorr"] - np.average(
        ok["dm_residual_dcorr"], weights=1.0 / ok["dm_residual_std"].values ** 2)
    section_correlations(ok.rename(columns={"dm_resid_tmp": "dm_resid_tmp"}),
                          "dm_resid_tmp", "DM residual", dm_covariates, args.outdir)
 
    section_multivariate(ok, args.outdir)
    section_snr_scaling(ok, args.outdir)
    section_solar(ok, args.outdir)
    section_flagged_comparison(df, args.outdir)
 
    hr("Done")
    print(f"  Plots saved to: {os.path.abspath(args.outdir)}")
 
 
if __name__ == "__main__":
    main()