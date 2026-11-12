"""
fit_vonmises.py — Fit von Mises components to the frequency-scrunched
pulsar profile, identify all significant components, and measure precise
phase centres and relative amplitudes for the SBI simulator.

Accepts either:
  • A PSRCHIVE archive (.ar, .fits, ...)  — fscrunches internally. PSRCHIVE
    reads the format from the file itself, so .fits (PSRFITS) works as .ar does
  • An SBI .pt file           — averages the portrait over the 40 channels

Usage:
    python -m p2sbi.fit_vonmises path/to/archive.ar  [--outdir DIR]
    python -m p2sbi.fit_vonmises path/to/archive.fits [--outdir DIR]
    python -m p2sbi.fit_vonmises path/to/data.pt     [--outdir DIR]

  --scattering fits the whole portrait (config's n_chan x n_bins) instead of the
  band-averaged profile, with every model convolved with an exponential
  scattering tail tau(nu) = tau_ref (nu/nu_ref)^alpha. Same greedy + AIC search;
  see the SCATTERING MODE section.
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks

from .pulsar_config import get_config, resolve_config_path

# --------------------------------------------------
# Config
#
# Pre-parsed here because N_BINS and the reference frequency are module-level
# constants used by the von Mises model functions defined below.
#
# TRAP 1: this script is where the config's amplitude_ref / concentration_ref
# values come from, and they are defined AT reference_freq_mhz. Fitting against
# one config and pasting the results into another with a different
# reference_freq_mhz silently corrupts them. Reading the reference frequency
# from the config -- rather than the hardcoded I-LOFAR value that used to live
# here -- is what closes that loop.
# --------------------------------------------------

# No config is needed: without one, phase bins, channel grid, period and DM come
# from the archive (with --scattering the fitted values are written as a config to
# build FROM). These module constants are then filled in from the archive before
# anything uses them.



# ─────────────────────────────────────────────────────────────────────────────
# Von Mises model
# ─────────────────────────────────────────────────────────────────────────────

def vm_single(b, A, mu, kappa):
    """Single von Mises component on a circular phase grid [0, N_BINS)."""
    return A * np.exp(kappa * (np.cos(2 * np.pi * (b - mu) / N_BINS) - 1.0))


def vm_model(b, *params):
    """Sum of N von Mises components plus a constant baseline.
    params layout: A_0, mu_0, kappa_0, A_1, mu_1, kappa_1, ..., baseline
    """
    n = (len(params) - 1) // 3
    out = np.full(len(b), float(params[-1]))
    for i in range(n):
        out = out + vm_single(b, params[3*i], params[3*i+1], params[3*i+2])
    return out


def _wrap_bins(d, n):
    """Wrap a bin difference into (-n/2, n/2]."""
    return (np.asarray(d) + n / 2.0) % n - n / 2.0


def _fwhm_of(k):
    """von Mises FWHM in phase bins."""
    return 2 * np.arccos(max(1 - np.log(2) / max(k, 1e-9), -1.0)) / (2 * np.pi) * N_BINS


def compute_aic(profile, fitted, n_params):
    rss = np.sum((profile - fitted) ** 2)
    n   = len(profile)
    return n * np.log(max(rss / n, 1e-30)) + 2.0 * n_params


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fwhm_to_kappa(fwhm_bins):
    """Approximate kappa from FWHM (bins). Large-kappa approximation."""
    return max(2.0 * np.log(2.0) * (N_BINS / (np.pi * max(fwhm_bins, 2.0))) ** 2, 2.0)


def estimate_fwhm(profile, peak_bin):
    """FWHM in bins, walking outward CIRCULARLY from the peak.

    The non-circular version stopped at the array edges, so a component near
    phase 0 or 1 got a truncated FWHM and therefore a badly wrong kappa seed.
    """
    n    = len(profile)
    half = profile[int(peak_bin) % n] / 2.0
    L = R = 0
    while L < n // 2 and profile[(int(peak_bin) - L) % n] > half:
        L += 1
    while R < n // 2 and profile[(int(peak_bin) + R) % n] > half:
        R += 1
    return max(L + R, 2)


def fit_n_components(profile, n, seed_bins):
    """Fit exactly n components, seeding from seed_bins (sorted by amplitude)."""
    b = np.arange(N_BINS, dtype=float)

    p0, lo, hi = [], [], []
    for i in range(n):
        if i < len(seed_bins):
            mu = float(seed_bins[i])
            # Phase is circular, so a seed at/near the wrap point can round to
            # N_BINS. Wrap the INDEX rather than the seed value: the seed itself
            # stays where the previous fit put it, and only the amplitude/width
            # lookup is wrapped. Indexing without this raises IndexError for any
            # component whose centre sits in the last half bin — reachable
            # whenever the reference-channel fit is seeded from a component near
            # phase 1.0.
            _bi = int(round(mu)) % N_BINS
            A  = float(profile[_bi])
            k  = fwhm_to_kappa(estimate_fwhm(profile, _bi))
        else:
            mu = float((i + 1) * N_BINS // (n + 1))
            A  = 0.05
            k  = 30.0
        # mu bounds extend BEYOND [0, N_BINS] because the von Mises model is
        # periodic but curve_fit's bounds are not. With mu pinned to [0, N_BINS]
        # a component whose true centre sits near phase 0 or 1 gets stuck
        # exactly on the boundary and the fit silently returns phase 0.0 or 1.0.
        # parse_components() wraps the result back into [0, N_BINS).
        p0 += [A,   mu,            k     ]
        lo += [0.0, -0.5 * N_BINS, 1.0   ]
        hi += [2.0,  1.5 * N_BINS, 3000.0]

    p0 += [0.0]
    lo += [-0.5]
    hi += [0.5]

    # Clip initial guess to within bounds — prevents "outside bounds" failure
    # when fwhm_to_kappa returns values above the kappa ceiling for narrow peaks
    p0 = [float(np.clip(v, l + 1e-6, h - 1e-6))
          for v, l, h in zip(p0, lo, hi)]

    try:
        popt, pcov = curve_fit(vm_model, b, profile,
                               p0=p0, bounds=(lo, hi),
                               maxfev=200_000, method="trf")
        perr   = np.sqrt(np.diag(pcov))
        fitted = vm_model(b, *popt)
        return popt, perr, fitted
    except Exception as e:
        print(f"    Fit (N={n}) failed: {e}")
        return None, None, None


def parse_components(popt, perr, n, sort_by_amp=True):
    """Unpack a fit. `sort_by_amp=False` preserves the seeding order, which is
    what --seed_from_config needs so component i stays config component i."""
    comps = []
    for i in range(n):
        comps.append(dict(
            A   = popt[3*i],
            # Wrap into [0, N_BINS): mu is fitted on a deliberately wider
            # interval so periodic components are not pinned at the boundary.
            mu  = popt[3*i+1] % N_BINS,
            k   = popt[3*i+2],
            dA  = perr[3*i],
            dmu = perr[3*i+1],
            dk  = perr[3*i+2],
            seed_index = i,
        ))
    if sort_by_amp:
        comps.sort(key=lambda c: c["A"], reverse=True)
    main_A = max(c["A"] for c in comps)
    for c in comps:
        c["amp_rel"] = c["A"] / main_A
    return comps


def cfg_mid(field):
    """Mid-prior value for a component field that may be a '$param' reference.

    Used only to SEED a fit. Log-sampled parameters use the geometric mean.
    """
    if isinstance(field, str) and field.startswith("$"):
        nm = field[1:]
        p  = next(p for p in list(cfg.inferred) + list(cfg.nuisance) if p.name == nm)
        return float(np.sqrt(p.low * p.high)) if p.log else float(0.5 * (p.low + p.high))
    return float(field)


def fit_seeded_from_config(profile, ref_bin):
    """Fit exactly the config's components, seeded at the config's own geometry.

    A free fit does not recover a declared decomposition: on this profile the
    fourth component goes to sharpening the main peak rather than to the weak
    broad bridge the config calls 'Inter', because that minimises residuals
    better. Seeding at the config's RELATIVE phases and mid-prior widths asks a
    different, and for our purposes the right, question: holding the config's
    component structure, where does this profile put them?

    `ref_bin` is where the reference component sits in `profile`, which fixes
    the constant offset. Relative phases come from the config.

    Returns (popt, perr, fitted) or (None, None, None).
    """
    n   = len(cfg.components)
    b   = np.arange(N_BINS, dtype=float)
    ref = cfg.reference_phase

    p0, lo, hi = [], [], []
    for c in cfg.components:
        mu_seed = ref_bin + (float(c.phase_center_ref) - ref) * N_BINS
        a_seed  = cfg_mid(c.amplitude_ref)
        k_seed  = float(np.clip(cfg_mid(c.concentration_ref), 1.0, 2999.0))
        p0 += [a_seed, mu_seed, k_seed]
        lo += [0.0, mu_seed - 0.15 * N_BINS, 1.0]
        hi += [2.0, mu_seed + 0.15 * N_BINS, 3000.0]
    p0 += [0.0]; lo += [-0.5]; hi += [0.5]

    p0 = [float(np.clip(v, l + 1e-6, h - 1e-6)) for v, l, h in zip(p0, lo, hi)]
    try:
        popt, pcov = curve_fit(vm_model, b, profile, p0=p0, bounds=(lo, hi),
                               maxfev=200_000, method="trf")
        return popt, np.sqrt(np.diag(pcov)), vm_model(b, *popt)
    except Exception as e:
        print(f"    Config-seeded fit failed: {e}")
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def main(argv=None):
    global N_BINS, cfg
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=None)
    _pre.add_argument("--scattering", action="store_true")
    _known, _ = _pre.parse_known_args(argv)
    cfg = None
    if not (_known.scattering and _known.config is None):
        try:
            cfg = get_config(resolve_config_path(_known.config))
        except ValueError:
            pass                       # no config: everything comes from the archive
    CONFIG_FREE = cfg is None
    N_BINS      = cfg.n_bins if cfg else None
    REF_FREQ_MHZ = cfg.ref_freq_mhz if cfg else None
    FREQ_LOW_MHZ = cfg.freq_low_mhz if cfg else None
    FREQ_HI_MHZ  = cfg.freq_high_mhz if cfg else None
    N_CHAN       = cfg.n_chan if cfg else None

    parser = argparse.ArgumentParser(
        description=(f"Fit von Mises components. Config: {os.path.basename(cfg.path)} "
                     f"({cfg.name}, {cfg.n_chan} ch x {cfg.n_bins} bins, "
                     f"ref {cfg.ref_freq_mhz:.3f} MHz)") if cfg else
                    "Fit von Mises components. No config needed: without --config, phase "
                    "bins, channels, period and DM are read from the archive."
    )
    parser.add_argument("infile", help="PSRCHIVE archive (.ar, .fits, .calib, .rf, .zap) or SBI .pt file")
    parser.add_argument("--outdir", default=None, help="Output directory for the plot")
    parser.add_argument("--config", default=None,
                        help="Optional YAML config (or $SBI_PULSAR_CONFIG, except with "
                             "--scattering). Without it, phase bins, channels, period and the "
                             "reference frequency (--ref-freq) come from the archive. With it, "
                             "they come from the config, the fit is compared with the config's "
                             "components, and --emit-yaml, --seed_from_config and --recentre-on "
                             "are available. A .pt input needs it.")
    parser.add_argument("--max_components", type=int, default=10,
                        help="Maximum number of components to try (default 10)")
    parser.add_argument("--force_n_components", type=int, default=None,
                        help="Force exactly N components, bypassing AIC/whiteness selection.")
    parser.add_argument("--ref_window", type=int, default=5,
                        help="Number of channels to average around the reference "
                             "frequency for the 'at reference frequency' fit (default 5). "
                             "A single channel has sqrt(n_chan) times the noise of the "
                             "band-scrunched profile, which leaves the weakest component "
                             "unconstrained; a narrow window recovers S/N while keeping "
                             "the spectral-weighting bias negligible. Use 1 for a strictly "
                             "single-channel fit.")
    parser.add_argument("--seed_from_config", action="store_true",
                        help="Report the config-seeded fit rather than the free one in the "
                             "detailed sections, the plot and --emit-yaml. The free search "
                             "runs and is reported REGARDLESS, and the two are compared by "
                             "AIC and RMS, so this only changes which fit is described in "
                             "detail — it cannot make the config confirm itself. Useful "
                             "because component i is then config component i, so the "
                             "phase comparison needs no matching.")
    parser.add_argument("--min_significance", type=float, default=3.0,
                        help="Amplitude threshold, in units of the off-pulse RMS, for a "
                             "component to count as detected (default 3). Used by the "
                             "significance scan, which reports the largest N whose every "
                             "component clears it — AIC on its own happily adds "
                             "components that fit single noise bins.")
    parser.add_argument("--emit-yaml", action="store_true", dest="emit_yaml",
                        help="Print a ready-to-paste `components:` YAML block from the "
                             "fit, and report the constant phase offset against the "
                             "config's current phase centres.")
    parser.add_argument("--recentre-on", default=None, dest="recentre_on",
                        help="With --emit-yaml: subtract a constant from every fitted "
                             "phase centre so this config component's centre is "
                             "reproduced exactly (e.g. --recentre-on Main). Only the "
                             "DIFFERENCES between phase centres are physical — a "
                             "constant offset is degenerate with the `toa` parameter — "
                             "so this keeps the profile shape while leaving toa near 0.")
    parser.add_argument("--scattering", action="store_true",
                        help="Fit the whole portrait, not the band-averaged profile, with "
                             "an exponential scattering tail on every model: "
                             "tau(nu) = tau_ref (nu/nu_ref)^alpha, plus a residual DM. "
                             "Component amplitudes and widths are power laws in "
                             "frequency, as in the simulator. Use for any pulsar whose "
                             "tails grow to low frequency: without it the greedy search "
                             "spends components on the tail. Needs no config: without "
                             "--config everything comes from the archive and the fitted "
                             "values are printed as a config block.")
    parser.add_argument("--nchan", type=int, default=None,
                        help="With --scattering and no --config: frequency-scrunch the "
                             "archive to this many channels before fitting (default: "
                             "keep the archive's).")
    parser.add_argument("--ref-freq", type=float, default=None, dest="ref_freq",
                        help="Without --config: reference frequency in "
                             "MHz at which amplitudes, widths and tau are quoted (snapped "
                             "to the nearest channel centre). Default: the channel nearest "
                             "the band's geometric centre. The report also gives the "
                             "frequency where tau_ref and alpha are uncorrelated.")
    parser.add_argument("--prior-nsigma", type=float, default=5.0, dest="prior_nsigma",
                        help="With --scattering and no --config, a config is written to "
                             "--outdir. Every prior in it is the fitted value ± this many fit "
                             "sigmas (default 5), but no narrower than the floors below.")
    parser.add_argument("--prior-dm", type=float, default=0.01, dest="prior_dm",
                        help="Minimum half-width of the written DM prior, pc/cm^3 (default "
                             "0.01): room for the DM to differ between epochs.")
    parser.add_argument("--prior-tau-factor", type=float, default=3.0, dest="prior_tau_factor",
                        help="The written tau prior is log-uniform from tau/F to tau*F, F at "
                             "least this (default 3).")
    parser.add_argument("--prior-alpha", type=float, default=1.0, dest="prior_alpha",
                        help="Minimum half-width of the written scattering-index prior "
                             "(default 1.0).")
    parser.add_argument("--prior-toa", type=float, default=0.02, dest="prior_toa",
                        help="Minimum half-width of the written toa prior in turns, on top of "
                             "the shift the DM prior forces on it (default 0.02).")
    parser.add_argument("--prior-snr-factor", type=float, default=10.0, dest="prior_snr_factor",
                        help="The written S/N prior is log-uniform from S/N/F to S/N*F "
                             "(default 10).")
    parser.add_argument("--prior-floor", type=float, default=0.25, dest="prior_floor",
                        help="Minimum prior half-width for concentrations and amplitudes, "
                             "as a fraction of the fitted value (default 0.25). One epoch "
                             "cannot show how much a component varies between epochs.")
    parser.add_argument("--prior-index-floor", type=float, default=0.5, dest="prior_index_floor",
                        help="Minimum prior half-width for amplitude and width spectral "
                             "indices (default 0.5).")
    parser.add_argument("--spectral-indices", choices=("free", "shared"), default="free",
                        dest="spectral_indices",
                        help="With --scattering: 'free' gives every component its own "
                             "amplitude and width spectral index; 'shared' gives one width "
                             "index per pulse (main pulse / interpulse) and one amplitude "
                             "index for the interpulse relative to the main pulse. "
                             "Default free.")
    parser.add_argument("--n-subbands", type=int, default=8, dest="n_subbands",
                        help="With --scattering: subbands for the plots and the "
                             "tau-per-subband check (default 8).")
    args = parser.parse_args(argv)
    if cfg is None:
        _needs = [f for f, on in (("--emit-yaml", args.emit_yaml),
                                  ("--seed_from_config", args.seed_from_config),
                                  ("--recentre-on", args.recentre_on)) if on]
        if _needs:
            parser.error(f"{', '.join(_needs)}: needs --config (works against a config's components).")

    ext    = os.path.splitext(args.infile)[1].lower()
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.infile)) or "."
    stem   = os.path.splitext(os.path.basename(args.infile))[0]
    os.makedirs(outdir, exist_ok=True)


    # ═════════════════════════════════════════════════════════════════════════════
    # SCATTERING MODE  (--scattering)
    #
    # The band-averaged profile is the wrong thing to fit when the pulsar is
    # scattered. The exponential tail grows steeply to low frequency (~nu^-4), so
    # the band average is a blend of many different tails, and the greedy search
    # spends components on that blend -- broad bumps sitting on the tail, which the
    # simulator would then scatter a SECOND time.
    #
    # This mode fits the whole portrait (n_chan x n_bins: the config's, as
    # real_archive_loader.py delivers it, or with no config the archive's own after
    # --nchan) with the simulator's own forward model:
    #
    #   T_nu(phi) = sum_c A_c (nu/nu_ref)^gamma_c vM(phi; mu_c, kappa_c (nu/nu_ref)^s_c)
    #   I_nu(phi) = g_nu [PBF(tau_nu) (*) T_nu](phi - d_nu) + b_nu
    #   tau_nu = tau_ref (nu/nu_ref)^alpha          one tail shared by every component
    #   d_nu   = K_DM dDM (nu^-2 - nu_top^-2) / P   residual DM, zero at the top channel
    #                                               (the simulator's convention)
    #
    # * The PBF exp(-t/tau)/tau is applied in the Fourier domain,
    #   1 / (1 + 2 pi i f tau), so a tail running past phase 1 wraps as in the data.
    # * g_nu and b_nu, a gain and baseline per channel, are solved by linear least
    #   squares inside every model evaluation. They absorb the bandpass, so the
    #   nonlinear search sees only profile, scattering and DM parameters. Component
    #   0 (the first seed, the brightest peak) therefore has A = 1, and its gamma is
    #   read off the gains; every other gamma is fitted relative to it.
    # * The search is the 1-D one: seed at the brightest peaks, add the component
    #   the residual asks for, refit everything, keep it while AIC drops; then the
    #   significance scan decides how many components are real.
    #
    # This mode prints its own report, config block and plot, then exits. The 1-D
    # path is unchanged.
    # ═════════════════════════════════════════════════════════════════════════════

    SC_KDM       = 4.148808e3      # s MHz^2 / (pc cm^-3)
    SC_KAPPA_MAX = 3000.0          # same ceiling as fit_n_components
    # Filled in before _scattering_main() runs, from the config or the archive:
    #   period_s   folding period, converts DM delay to phase
    #   nu_top     DM delays are zero here: the top of the channel grid, as in the simulator
    SC = {}


    def _sc_shift(arr, bins, fk):
        """Shift each row of `arr` later by `bins` (fractional, circular)."""
        ph = np.exp(-2j * np.pi * fk[None, :] * np.asarray(bins, dtype=float)[:, None])
        return np.fft.ifft(np.fft.fft(arr, axis=-1) * ph, axis=-1).real


    class ScatteredPortraitModel:
        """The simulator's forward model on one portrait, with named parameters.

    groups: one letter per component, 'M' (main pulse) or 'I' (interpulse);
            used only when index_mode is 'shared'.
    """

        def __init__(self, D, nu, sig, n_comp, groups, index_mode):
            self.D, self.nu, self.sig = D, nu, sig
            self.q      = nu / REF_FREQ_MHZ
            self.n      = n_comp
            self.groups = groups
            self.mode   = index_mode
            self.b      = np.arange(D.shape[1], dtype=float)
            self.fk     = np.fft.fftfreq(D.shape[1])
            self.dm_bins = SC_KDM * (nu ** -2 - SC["nu_top"] ** -2) / SC["period_s"] * D.shape[1]
            names = []
            for c in range(n_comp):
                names += [f"mu{c}", f"lnk{c}"] + ([f"lnA{c}"] if c else [])
            if index_mode == "free":
                for c in range(n_comp):
                    names += [f"s{c}"] + ([f"dg{c}"] if c else [])
            else:
                names += [f"s_{g}" for g in sorted(set(groups))]
                if "I" in groups:
                    names.append("dg_I")
            names += ["lntau", "alpha", "ddm"]
            self.names = names
            self.ix    = {n: i for i, n in enumerate(names)}

        def components(self, x):
            """[(mu_bin, kappa_ref, width_index, A_ref, gamma - gamma_0)] per component."""
            out = []
            for c in range(self.n):
                mu  = x[self.ix[f"mu{c}"]]
                kap = np.exp(x[self.ix[f"lnk{c}"]])
                A   = 1.0 if c == 0 else np.exp(x[self.ix[f"lnA{c}"]])
                if self.mode == "free":
                    s  = x[self.ix[f"s{c}"]]
                    dg = 0.0 if c == 0 else x[self.ix[f"dg{c}"]]
                else:
                    s  = x[self.ix[f"s_{self.groups[c]}"]]
                    dg = x[self.ix["dg_I"]] if self.groups[c] == "I" else 0.0
                out.append((mu, kap, s, A, dg))
            return out

        def tau_bins(self, x):
            return np.exp(x[self.ix["lntau"]]) * self.q ** x[self.ix["alpha"]]

        def propagate(self, T, tau, ddm):
            """Scatter (exponential PBF) and DM-shift every channel of T."""
            H = (np.exp(-2j * np.pi * self.fk[None, :] * (ddm * self.dm_bins)[:, None])
                 / (1.0 + 2j * np.pi * self.fk[None, :] * tau[:, None]))
            return np.fft.ifft(np.fft.fft(T, axis=-1) * H, axis=-1).real

        def intrinsic(self, x):
            """Unscattered (n_chan, n_bins) portrait of each component."""
            nb, out = self.D.shape[1], []
            for mu, kap, s, A, dg in self.components(x):
                cosd = np.cos(2 * np.pi * (self.b - mu) / nb) - 1.0
                out.append((A * self.q ** dg)[:, None]
                           * np.exp((kap * self.q ** s)[:, None] * cosd[None, :]))
            return out

        def template(self, x):
            return self.propagate(np.sum(self.intrinsic(x), axis=0),
                                  self.tau_bins(x), x[self.ix["ddm"]])

        def project(self, T, D=None):
            """Per-channel gain and baseline minimising sum (D - g T - b)^2."""
            D = self.D if D is None else D
            nb = D.shape[1]
            stt, st = (T * T).sum(1), T.sum(1)
            sdt, sd = (T * D).sum(1), D.sum(1)
            det = stt * nb - st * st
            return (sdt * nb - st * sd) / det, (stt * sd - st * sdt) / det

        def resid(self, x):
            T = self.template(x)
            g, b0 = self.project(T)
            return ((self.D - g[:, None] * T - b0[:, None]) / self.sig[:, None]).ravel()

        def fit(self, start):
            x0 = np.array([start.get(n, 0.0) for n in self.names], dtype=float)
            lo, hi = map(np.array, zip(*[_sc_bounds(n, v) for n, v in zip(self.names, x0)]))
            x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
            try:
                r = least_squares(self.resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=8000)
            except Exception as e:
                print(f"    Scattering fit (N={self.n}) failed: {e}")
                return None
            n_lin = 2 * self.D.shape[0]                 # gains + baselines
            dof   = self.D.size - r.x.size - n_lin
            chi2  = 2.0 * r.cost
            cov   = np.linalg.pinv(r.jac.T @ r.jac) * max(chi2 / dof, 1.0)
            T     = self.template(r.x)
            g, b0 = self.project(T)
            return dict(model=self, x=r.x, err=np.sqrt(np.diag(cov)), cov=cov, chi2=chi2,
                        chi2r=chi2 / dof, aic=chi2 + 2.0 * (r.x.size + n_lin),
                        fitted=g[:, None] * T + b0[:, None], gain=g, base=b0,
                        start=dict(zip(self.names, r.x)))


    def _sc_bounds(name, v):
        """Search interval for one parameter, around its starting value."""
        if name.startswith("mu"):
            return v - 15.0, v + 15.0
        if name.startswith("lnk"):
            return 0.0, np.log(SC_KAPPA_MAX)
        if name.startswith("lnA"):
            return np.log(1e-3), np.log(10.0)
        if name.startswith("s"):
            return -4.0, 4.0
        if name.startswith("dg"):
            return -6.0, 6.0
        if name == "lntau":
            return np.log(0.01), np.log(N_BINS / 2.0)
        if name == "alpha":
            return -8.0, 0.0
        if name == "ddm":
            return -0.05, 0.05
        raise KeyError(name)


    def _sc_groups(mus):
        """'M' for components within a quarter turn of component 0, else 'I'."""
        return "".join("M" if abs(_wrap_bins(m - mus[0], N_BINS)) < 0.25 * N_BINS else "I"
                       for m in mus)


    def _sc_fit(D, nu, sig, mus, start):
        model = ScatteredPortraitModel(D, nu, sig, len(mus), _sc_groups(mus), args.spectral_indices)
        return model.fit(start)


    def _sc_significance(res):
        """Per-component checks. A component is a detection only if it passes all of:

      sig         peak of its contribution to the band-averaged profile, in
                  units of that profile's noise, >= --min_significance
      pinned      kappa below the ceiling (not a one-bin spike)
      idx_pinned  amplitude and width indices inside their search limits: an
                  index at the limit means a 20x amplitude or 7x kappa change
                  across the band, which is the fit imitating something else
      tail_like   NOT a scattering tail in disguise: broader than a brighter
                  component it trails by < 0.15 turns, and brightening towards
                  low frequency much faster than it (gamma lower by > 2). That is
                  what an exponential tail looks like to a sum of von Mises, and
                  such terms take flux away from tau and bias alpha. With
                  --spectral-indices shared a component cannot have its own
                  gamma, so the test there is width alone: more than twice as
                  broad as the component it trails. (On J1939 such terms moved
                  alpha from -4.18 to -4.92.)

    `blended` (inside a brighter component's half-width) is reported only: in a
    portrait, two overlapping terms that evolve differently with frequency
    describe how one feature changes shape, which only the frequency axis shows.
    """
        m, x = res["model"], res["x"]
        parts = m.intrinsic(x)
        tau, ddm = m.tau_bins(x), x[m.ix["ddm"]]
        sig_mean = np.sqrt((m.sig ** 2).sum()) / m.sig.size
        s_lim, g_lim = _sc_bounds("s0", 0.0)[1], _sc_bounds("dg1", 0.0)[1]
        out = []
        for (mu, kap, s, A, dg), part in zip(m.components(x), parts):
            prof_c = (res["gain"][:, None] * m.propagate(part, tau, ddm)).mean(axis=0)
            out.append(dict(mu=mu % N_BINS, k=kap, A=A, dg=dg, s=s, sig=prof_c.max() / sig_mean,
                            pinned=kap >= 0.99 * SC_KAPPA_MAX, fwhm=_fwhm_of(kap),
                            idx_pinned=abs(s) >= 0.99 * s_lim or abs(dg) >= 0.99 * g_lim))
        for i, r in enumerate(out):
            others = [o for j, o in enumerate(out) if j != i and o["A"] > r["A"]]
            r["blended"] = any(abs(float(_wrap_bins(r["mu"] - o["mu"], N_BINS))) < 0.5 * o["fwhm"]
                               for o in others)
            r["tail_like"] = any(0 < float(_wrap_bins(r["mu"] - o["mu"], N_BINS)) < 0.15 * N_BINS
                                 and r["fwhm"] > o["fwhm"]
                                 and (r["dg"] - o["dg"] < -2.0
                                      or (m.mode == "shared" and r["fwhm"] > 2.0 * o["fwhm"]))
                                 for o in others)
            r["ok"] = (r["sig"] >= args.min_significance and not r["pinned"]
                       and not r["idx_pinned"] and not r["tail_like"])
        return out


    def _sc_tau_per_subband(res, nsub):
        """tau fitted alone in each subband, profile and DM held at the best fit.
    A single power law should run through these."""
        m, x = res["model"], res["x"]
        Tin, ddm = np.sum(m.intrinsic(x), axis=0), x[m.ix["ddm"]]
        edges = np.linspace(0, m.D.shape[0], nsub + 1).astype(int)
        out = []
        for k in range(nsub):
            sl = slice(edges[k], edges[k + 1])
            if sl.stop <= sl.start:
                continue

            def r_k(p, sl=sl):
                tau = np.full(sl.stop - sl.start, np.exp(p[0]))
                H = (np.exp(-2j * np.pi * m.fk[None, :] * (ddm * m.dm_bins[sl])[:, None])
                     / (1.0 + 2j * np.pi * m.fk[None, :] * tau[:, None]))
                Tk = np.fft.ifft(np.fft.fft(Tin[sl], axis=-1) * H, axis=-1).real
                g, b0 = m.project(Tk, m.D[sl])
                return ((m.D[sl] - g[:, None] * Tk - b0[:, None]) / m.sig[sl, None]).ravel()

            nu_k = float(np.exp(np.log(m.nu[sl]).mean()))
            t0 = np.log(np.exp(x[m.ix["lntau"]]) * (nu_k / REF_FREQ_MHZ) ** x[m.ix["alpha"]])
            lo_, hi_ = np.log(0.005), np.log(float(N_BINS))
            rk = least_squares(r_k, [np.clip(t0, lo_ + 1e-6, hi_ - 1e-6)], bounds=([lo_], [hi_]))
            dof = max(rk.fun.size - 1 - 2 * (sl.stop - sl.start), 1)
            ek = float(np.sqrt(np.linalg.pinv(rk.jac.T @ rk.jac)[0, 0] * max(2 * rk.cost / dof, 1.0)))
            out.append((nu_k, float(np.exp(rk.x[0])), ek))
        return out


    def _sc_site(telescope):
        """Geodetic (WGS84) position of `telescope` from tempo2's observatories.dat
    ($TEMPO2, else this environment's share/tempo2), or None if it is not listed."""
        for root in (os.environ.get("TEMPO2"), os.path.join(sys.prefix, "share", "tempo2")):
            path = os.path.join(root or "", "observatory", "observatories.dat")
            if not root or not os.path.isfile(path):
                continue
            with open(path) as fh:
                for line in fh:
                    tok = line.split()
                    if len(tok) < 5 or line.lstrip().startswith("#"):
                        continue
                    if telescope.strip().lower() not in (" ".join(tok[3:-1]).lower(), tok[-1].lower()):
                        continue
                    try:
                        x, y, z = map(float, tok[:3])
                    except ValueError:
                        continue
                    a_, f_ = 6378137.0, 1.0 / 298.257223563
                    e2 = f_ * (2.0 - f_)
                    p_ = np.hypot(x, y)
                    lat, h = np.arctan2(z, p_ * (1.0 - e2)), 0.0
                    for _ in range(10):
                        n_ = a_ / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
                        h = p_ / np.cos(lat) - n_
                        lat = np.arctan2(z, p_ * (1.0 - e2 * n_ / (n_ + h)))
                    return dict(lon_deg=float(np.degrees(np.arctan2(y, x))),
                                lat_deg=float(np.degrees(lat)), height_m=float(h),
                                xyz=(x, y, z), file=path)
        return None


    def _sc_sexa(s, u1, u2, u3):
        """'19:39:38.56' -> '19h39m38.56s' (u1, u2, u3 = 'h', 'm', 's' or 'd', 'm', 's')."""
        p = s.split(":")
        return f"{p[0]}{u1}{p[1]}{u2}{p[2]}{u3}" if len(p) == 3 else s


    def _sc_priors(comps, fit):
        """Every prior the written config uses, from one rule (see the config header)."""
        ns = args.prior_nsigma
        order = sorted(comps, key=lambda c: -c["A"])
        P = SC["period_s"]
        tau_ms = fit["tau_bins"] * P * 1e3 / N_BINS
        h_dm = max(ns * fit["ddm_err"], args.prior_dm)
        f_tau = max(args.prior_tau_factor, float(np.exp(ns * fit["tau_err_frac"])))
        h_al = max(ns * fit["alpha_err"], args.prior_alpha)
        c_toa = SC_KDM * (SC["nu_top"] ** -2 - SC["nu_c"] ** -2) / P       # turns per pc/cm^3
        toa_err = order[0]["dmu"] / N_BINS
        h_toa = abs(c_toa) * h_dm + max(ns * toa_err, args.prior_toa)
        inferred = [
            ("DM", fit["ddm"] - h_dm, fit["ddm"] + h_dm, False,
             f"fitted dDM {fit['ddm']:+.5f} ± {fit['ddm_err']:.5f}"),
            ("tau_sc", tau_ms / f_tau, tau_ms * f_tau, True,
             f"fitted {tau_ms:.6f} ms ± {fit['tau_err_frac'] * 100:.1f}% at reference_freq_mhz"),
            ("alpha_sc", fit["alpha"] - h_al, fit["alpha"] + h_al, False,
             f"fitted {fit['alpha']:.3f} ± {fit['alpha_err']:.3f}"),
            ("toa", -h_toa, h_toa, False, f"0 = this archive; reference phase ± {toa_err:.2e} turns"),
            ("snr", fit["snr"] / args.prior_snr_factor, fit["snr"] * args.prior_snr_factor, True,
             f"measured {fit['snr']:.0f}"),
        ]

        def lim(v, err, floor, low_min=None):
            h = max(ns * err, floor)
            return (v - h if low_min is None else max(v - h, low_min)), v + h

        ff, fi = args.prior_floor, args.prior_index_floor
        nuis = []
        for c in order:
            e_ = c["dk_frac"] * c["k"]
            nuis.append((f"conc_{c['name']}", *lim(c["k"], e_, ff * c["k"], 1.0), 1,
                         f"fitted {c['k']:.1f} ± {e_:.1f}  (FWHM {c['fwhm']:.2f} bins)"))
        for c in order[1:]:
            e_ = c["dA_frac"] * c["amp_rel"]
            nuis.append((f"amp_{c['name']}", *lim(c["amp_rel"], e_, ff * c["amp_rel"], 0.0), 4,
                         f"fitted {c['amp_rel']:.4f} ± {e_:.4f}"))
        for c in order:
            nuis.append((f"gamma_{c['name']}", *lim(c["gamma"], c["dgamma"], fi), 3,
                         f"fitted {c['gamma']:+.3f} ± {c['dgamma']:.3f}"
                         + ("" if c is comps[0] else f" (vs {comps[0]['name']} ± {c['dgamma_rel']:.3f})")))
        for c in order:
            nuis.append((f"widx_{c['name']}", *lim(c["s"], c["ds"], fi), 3,
                         f"fitted {c['s']:+.3f} ± {c['ds']:.3f}"))
        return order, tau_ms, inferred, nuis, c_toa


    def _sc_config_text(comps, fit):
        """A complete SBI_PDT config, every value from this archive and this fit."""
        import datetime
        order, tau_ms, inferred, nuis, c_toa = _sc_priors(comps, fit)
        ns = args.prior_nsigma
        site = _sc_site(SC["telescope"])
        labels = {"DM": '"$\\\\Delta$DM", unit: "pc cm$^{-3}$"',
                  "tau_sc": '"$\\\\tau_{\\\\rm sc}$", unit: "ms", display: exp',
                  "alpha_sc": '"$\\\\alpha_{\\\\rm sc}$"',
                  "toa": '"ToA offset", unit: "$\\\\mu$s", display: phase_to_us',
                  "snr": '"S/N", display: exp'}
        L = []
        w = L.append
        w("# " + "=" * 76)
        w(f"# {SC['source']} @ {SC['telescope']} — written by fit_vonmises.py --scattering")
        w("#")
        w(f"# Fitted from: {os.path.abspath(args.infile)}")
        w(f"#              MJD {SC['mjd']:.6f}, {SC['tobs']:.1f} s")
        w(f"# Command:     python {' '.join(os.path.basename(a) if a == sys.argv[0] else a for a in sys.argv)}")
        w(f"# Written:     {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC")
        w("#")
        w("# Every number below comes from that archive and that fit. Components are")
        w("# INTRINSIC (before scattering); amplitude_ref, concentration_ref and tau_sc are")
        w("# defined AT reference_freq_mhz. Each fitted value and its 1-sigma error is in")
        w(f"# the comment beside it; the fit's full record is {stem}_fit.yaml.")
        w(f"# The fit: {fit['n_components']} components, reduced chi2 {fit['chi2_reduced']:.3f}, "
          f"S/N {fit['snr']:.0f}.")
        w("#")
        w("# PRIORS — one rule, set on the command line (defaults in brackets):")
        w(f"#   nuisance  fitted ± max({ns:g} sigma, floor); floor {args.prior_floor * 100:g}% of the value for")
        w(f"#             concentrations and amplitudes, ±{args.prior_index_floor:g} for spectral indices")
        w("#             [--prior-nsigma 5, --prior-floor 0.25, --prior-index-floor 0.5]")
        w(f"#   DM        fitted dDM ± max({ns:g} sigma, {args.prior_dm:g})   [--prior-dm 0.01]")
        w(f"#   tau_sc    log-uniform tau/F .. tau*F, F = max({args.prior_tau_factor:g}, exp({ns:g} sigma))"
          "   [--prior-tau-factor 3]")
        w(f"#   alpha_sc  fitted ± max({ns:g} sigma, {args.prior_alpha:g})   [--prior-alpha 1.0]")
        w(f"#   toa       0 ± (|c| x DM half-width + max({ns:g} sigma, {args.prior_toa:g})), "
          f"c = {c_toa:.3f} turns per")
        w(f"#             pc/cm^3: the archive is dedispersed to {SC['nu_c']:.3f} MHz but the simulator")
        w(f"#             references DM to {SC['nu_top']:.3f} MHz, so toa absorbs c x dDM   [--prior-toa 0.02]")
        w(f"#   snr       log-uniform S/N/F .. S/N*F, F = {args.prior_snr_factor:g}   [--prior-snr-factor 10]")
        w("# One epoch cannot show how much anything varies between epochs; the floors are")
        w("# that allowance. Any prior can be edited here (the fingerprint follows it).")
        w("# " + "=" * 76)
        w("")
        w("pulsar:")
        w(f"  name:       {SC['source']}")
        w(f"  period_ms:  {SC['period_s'] * 1e3:.13f}   # archive folding period: converts DM delay to phase")
        if SC["f0_par"]:
            w(f"  f0_par_hz:  {SC['f0_par']}   # ephemeris F0"
              + (f" (UNITS {SC['units']})" if SC["units"] else "") + "; post-inference only")
        w(f"  dm_header:  {sc_dm_header:.6f}   # archive DM: the inferred DM is relative to this")
        w(f"  dm_conv:    {SC['dm_par'] or f'{sc_dm_header:.6f}'}"
          + ("   # ephemeris DM" if SC["dm_par"] else "   # no ephemeris DM; archive DM"))
        w(f'  raj:        "{_sc_sexa(SC["raj"], "h", "m", "s")}"')
        w(f'  decj:       "{_sc_sexa(SC["decj"], "d", "m", "s")}"')
        w("")
        w("observatory:")
        w(f"  name:      {SC['telescope']}")
        if site:
            w(f"  # {site['file']}: ITRF ({site['xyz'][0]:.2f}, {site['xyz'][1]:.2f}, "
              f"{site['xyz'][2]:.2f}) m -> WGS84")
            w(f"  lon_deg:   {site['lon_deg']:.6f}")
            w(f"  lat_deg:   {site['lat_deg']:.6f}")
            w(f"  height_m:  {site['height_m']:.1f}")
        else:
            w(f"  # '{SC['telescope']}' is not in tempo2's observatories.dat: add lon_deg, lat_deg")
            w("  # and height_m by hand (used only by timeseries.py, after inference).")
        w("")
        w("instrument:")
        w(f"  n_freq_channels:    {sc_freqs.size}")
        w(f"  n_phase_bins:       {N_BINS}")
        w(f"  freq_low_mhz:       {SC['freq_low']:.6f}   # band edges: centre ± |BW|/2")
        w(f"  freq_high_mhz:      {SC['freq_high']:.6f}")
        w(f"  reference_freq_mhz: {REF_FREQ_MHZ:.6f}   # tau_ref and alpha are uncorrelated at "
          f"{fit['nu_pivot']:.1f} MHz")
        w("  channel_freqs_mhz:")
        for i in range(0, sc_freqs.size, 8):
            row = ", ".join(f"{v:.6f}" for v in sc_freqs[i:i + 8])
            w(f"    {'[' if i == 0 else ' '}{row}{']' if i + 8 >= sc_freqs.size else ','}")
        w("  centroid_mode:      circular   # right for a pulse anywhere in phase; needed by dead_channels")
        w(f"  sigma_bandpass:     {fit['bp_mad']:.3f}   # MAD of ln(channel gain) about a power law "
          f"(std {fit['bp_std']:.2f})")
        w("  freq_tol_mhz:       2.0")
        w("")
        w("simulation:")
        w(f"  epoch_mjd:   {SC['mjd']:.6f}")
        w("  add_noise:   true")
        w("  dm_smearing: false     # the simulator's smearing uses the signed dDM; leave off")
        w("  noise_model: avg_peak  # snr is this model's: peak of the band-averaged profile / its noise")
        w("  npol:        1")
        w(f"  t_obs_s:     {SC['tobs']:.1f}")
        w("  dm_param:    DM")
        w("  snr_param:   snr")
        w("")
        if fit["dead_mhz"]:
            w("dead_channels:")
            w("  # All-zero channels in this archive. p is a placeholder: set it to the")
            w("  # fraction of your epochs in which each channel is dead.")
            w("  site_channels:")
            for v in fit["dead_mhz"]:
                w(f"    - {{mhz: {v:.3f}, p: 0.9}}")
            w("")
        w("inferred:")
        for name, lo, hi, lg, note in inferred:
            w(f"  - {{name: {name + ',':<10} low: {lo:.6g}, high: {hi:.6g},"
              + (" log: true," if lg else "") + f" label: {labels[name]}}}   # {note}")
        w("")
        w("nuisance:")
        for name, lo, hi, dp, note in nuis:
            w(f"  - {{name: {name + ',':<11} low: {lo:>10.{dp}f}, high: {hi:>10.{dp}f}}}   # {note}")
        w("")
        w(f"reference_component: {order[0]['name']}")
        w("components:")
        for c in order:
            n = c["name"]
            w(f"  - name:                  {n}")
            w(f"    amplitude_ref:         " + ("1.0" if c is order[0] else f"$amp_{n}"))
            w(f"    phase_center_ref:      {c['mu'] / N_BINS:.8f}   # ± {c['dmu'] / N_BINS:.8f}")
            w(f"    concentration_ref:     $conc_{n}")
            w(f"    spectral_index_amp:    $gamma_{n}")
            w(f"    spectral_index_width:  $widx_{n}")
            w("    spectral_index_center: 0.0")
            w("    sigma_broadening_bins: 0.0")
        w("")
        w("scattering:")
        w("  tau_ms:         {type: free, param: tau_sc}")
        w("  spectral_index: {type: free, param: alpha_sc}")
        return "\n".join(L) + "\n"


    def _sc_write_outputs(comps, fit):
        """Write <stem>_config.yaml (loadable by SBI_PDT) and <stem>_fit.yaml (the
    full fit record) to --outdir, then load the config back to prove it valid."""
        import datetime
        p_us = SC["period_s"] * 1e6 / N_BINS
        cfg_path = os.path.join(outdir, f"{stem}_config.yaml")
        fit_path = os.path.join(outdir, f"{stem}_fit.yaml")
        with open(cfg_path, "w") as fh:
            fh.write(_sc_config_text(comps, fit))

        order = sorted(comps, key=lambda c: -c["A"])
        alpha, tb = fit["alpha"], fit["tau_bins"]
        record = {
            "archive": os.path.abspath(args.infile),
            "command": " ".join(sys.argv),
            "written_utc": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}",
            "config_written": os.path.abspath(cfg_path),
            "archive_header": {
                "source": SC["source"], "telescope": SC["telescope"], "mjd": SC["mjd"],
                "length_s": SC["tobs"], "period_ms": SC["period_s"] * 1e3, "dm": sc_dm_header,
                "ephemeris_dm": SC["dm_par"], "ephemeris_f0_hz": SC["f0_par"],
                "ephemeris_units": SC["units"], "raj": SC["raj"], "decj": SC["decj"],
                "centre_freq_mhz": SC["nu_c"], "band_mhz": [SC["freq_low"], SC["freq_high"]],
                "n_chan_fitted": int(sc_freqs.size), "n_bins": int(N_BINS)},
            "settings": {
                "reference_freq_mhz": REF_FREQ_MHZ, "spectral_indices": fit["spectral_indices"],
                "max_components": args.max_components, "min_significance": args.min_significance,
                "prior_nsigma": args.prior_nsigma, "prior_floor": args.prior_floor,
                "prior_index_floor": args.prior_index_floor, "prior_dm": args.prior_dm,
                "prior_tau_factor": args.prior_tau_factor, "prior_alpha": args.prior_alpha,
                "prior_toa": args.prior_toa, "prior_snr_factor": args.prior_snr_factor},
            "model_selection": {
                "aic_preferred_n": fit["aic_preferred_n"], "significance_n": fit["significance_n"],
                "reported_n": fit["n_components"], "steps": fit["scan"]},
            "fit": {"chi2_reduced": fit["chi2_reduced"], "aic": fit["aic"],
                    "snr_avg_peak": fit["snr"]},
            "components": [{
                "name": c["name"], "phase": c["mu"] / N_BINS, "phase_err": c["dmu"] / N_BINS,
                "amplitude": c["amp_rel"], "amplitude_err": c["dA_frac"] * c["amp_rel"],
                "kappa": c["k"], "kappa_err": c["dk_frac"] * c["k"], "fwhm_bins": c["fwhm"],
                "gamma": c["gamma"], "gamma_err": c["dgamma"],
                "gamma_err_relative_to_" + comps[0]["name"]: c["dgamma_rel"],
                "width_index": c["s"], "width_index_err": c["ds"],
                "significance": c["sig"], "blended": bool(c["blended"])} for c in order],
            "scattering": {
                "tau_ref_us": tb * p_us, "tau_ref_ms": tb * p_us * 1e-3, "tau_ref_bins": tb,
                "tau_err_frac": fit["tau_err_frac"], "alpha": alpha, "alpha_err": fit["alpha_err"],
                "corr_lntau_alpha": fit["corr_tau_alpha"], "uncorrelated_at_mhz": fit["nu_pivot"],
                "tau_per_subband": [{
                    "nu_mhz": nu_k, "tau_us": t_k * p_us, "err_frac": e_k,
                    "power_law_us": tb * p_us * (nu_k / REF_FREQ_MHZ) ** alpha,
                    "ratio": t_k / (tb * (nu_k / REF_FREQ_MHZ) ** alpha)}
                    for nu_k, t_k, e_k in fit["tau_subbands"]]},
            "dm": {"ddm": fit["ddm"], "ddm_err": fit["ddm_err"], "dm": sc_dm_header + fit["ddm"]},
            "channels": {
                "sigma_bandpass_mad": fit["bp_mad"], "sigma_bandpass_std": fit["bp_std"],
                "gamma_overall_slope": fit["gamma0"], "gamma_overall_slope_err": fit["gamma0_err"],
                "left_out_of_slope_mhz": fit["clipped_mhz"], "noise_rms_max_over_min": fit["noise_ratio"],
                "off_pulse_bins": fit["off_pulse_bins"], "dead_mhz": fit["dead_mhz"]},
        }

        def plain(v):
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            if isinstance(v, (np.floating, float)):
                return float(f"{float(v):.10g}")
            if isinstance(v, (np.integer, np.bool_)):
                return v.item()
            return v

        with open(fit_path, "w") as fh:
            yaml.safe_dump(plain(record), fh, sort_keys=False, default_flow_style=None, width=100)

        print(f"\n{'═'*60}")
        print(f"  WRITTEN")
        print(f"{'═'*60}")
        print(f"  config     {cfg_path}")
        print(f"  fit record {fit_path}")
        try:
            from .pulsar_config import load_config
            c_ = load_config(os.path.abspath(cfg_path))
            print(f"  the config loads in SBI_PDT: {len(c_.inferred)} inferred, "
                  f"{len(c_.nuisance)} nuisance, {len(c_.components)} components, "
                  f"x_dim {c_.x_dim}, fingerprint {c_.fingerprint}")
        except Exception as e:
            print(f"  [WARN] the written config does not load: {e}")


    def _scattering_main():
        P_US = SC["period_s"] * 1e6 / N_BINS          # one phase bin, in microseconds
        MIN_SIG_SC = args.min_significance

        # ── data: the loader's normalisation; dead (all-zero) rows dropped ───────
        D_all = sc_portrait / (sc_portrait.max() + 1e-12)
        live  = np.abs(D_all).sum(axis=1) > 0
        dead_mhz = sc_freqs[~live]
        D, nu = D_all[live], sc_freqs[live]
        nch = nu.size

        # Off-pulse: the window quietest in EVERY channel. A "below 5% of peak"
        # threshold would count the scattering tails as noise.
        W = max(8, int(round(0.15 * N_BINS)))
        Dn = D / np.maximum(np.abs(D).max(axis=1, keepdims=True), 1e-30)
        score = [np.max(Dn[:, (s + np.arange(W)) % N_BINS].mean(axis=1)) for s in range(N_BINS)]
        off0 = int(np.argmin(score))
        # Work with the quiet window straddling phase 0, so no pulse sits on the
        # wrap while seeding. Undone before anything is reported.
        sc_roll = int((-(off0 + W // 2)) % N_BINS)
        D   = np.roll(D, sc_roll, axis=1)
        OFF = (off0 + sc_roll + np.arange(W)) % N_BINS
        sig = D[:, OFF].std(axis=1)
        unroll = lambda mu: (mu - sc_roll) % N_BINS

        print(f"\n{'═'*60}")
        print(f"  SCATTERING MODE — portrait {nch} live channels x {N_BINS} bins "
              f"({sc_freqs.min():.2f}–{sc_freqs.max():.2f} MHz)")
        print(f"{'═'*60}")
        if dead_mhz.size:
            print(f"  dead (all-zero) channels left out: "
                  f"{', '.join(f'{v:.2f}' for v in dead_mhz)} MHz")
        print(f"  off-pulse window: bins {unroll(OFF[0])}–{unroll(OFF[-1])}   "
              f"noise rms range across channels: x{sig.max()/sig.min():.1f}")
        print(f"  nu_ref = {REF_FREQ_MHZ:.4f} MHz ({'config' if cfg else 'archive channel'}); "
              f"period {SC['period_s']*1e3:.10f} ms; DM delays referenced to "
              f"{SC['nu_top']:.4f} MHz (top of the channel grid, as in the simulator)")
        print(f"  spectral indices: {args.spectral_indices}")

        # ── seeds: peaks of the least-scattered quarter of the band ─────────────
        top = D[-max(1, nch // 4):].mean(axis=0)
        top = top / top.max()
        pk, _ = find_peaks(top, height=0.05, prominence=0.03, distance=10)
        pk = sorted(pk.tolist(), key=lambda p: -top[p]) or [int(np.argmax(top))]
        seeds = [float(p) for p in pk]
        print(f"  seed peaks (top quarter of the band): "
              f"{', '.join(f'bin {unroll(p):.0f} ({top[int(p)]:.2f})' for p in seeds)}")

        # Starting tau and alpha: the config's mid-prior values when there is one,
        # otherwise 2 bins at nu_ref and -4. The fit moves well away from either.
        sv = (cfg.scattering_values({p.name: cfg_mid("$" + p.name)
                                     for p in list(cfg.inferred) + list(cfg.nuisance)})
              if cfg else dict(tau_ms=0.0, spectral_index=-4.0))
        tau0 = sv["tau_ms"] * 1e-3 / SC["period_s"] * N_BINS if sv["tau_ms"] > 0 else 2.0
        base_start = dict(lntau=np.log(max(tau0, 0.02)), alpha=sv["spectral_index"], ddm=0.0)

        def new_comp(c, mu):
            b_ = int(round(mu)) % N_BINS
            return {f"mu{c}": mu,
                    f"lnk{c}": np.log(np.clip(fwhm_to_kappa(estimate_fwhm(top, b_)), 2.0, 0.9 * SC_KAPPA_MAX)),
                    **({f"lnA{c}": np.log(max(top[b_], 0.02))} if c else {})}

        # ── greedy search ───────────────────────────────────────────────────────
        # Each step tries up to SC_N_CAND starting positions for the new component
        # and keeps the one that fits best (lowest AIC): unused peaks found above
        # first, then the strongest bumps left in the previous step's residual. A
        # position that has produced a component is used up. Taking only the first
        # candidate let a stale seed -- one whose component had drifted 5.6 bins away
        # -- put the new component on the tail while a 12-sigma residual went untried.
        SC_N_CAND = 3
        n_max = max(args.max_components, args.force_n_components or 0)
        print(f"\n  AIC model selection (max_components={args.max_components}; up to "
              f"{SC_N_CAND} starting positions tried per step; stops after two steps "
              f"without improvement):")
        print(f"  {'N':>3}  {'seed bin':>8}  {'AIC':>12}  {'ΔAIC':>10}  {'χ²_red':>7}  "
              f"{'tau_ref(us)':>11}  {'alpha':>7}  {'min sig':>8}  {'κ pinned':>8}")
        store, history = {}, []
        best_n, best_aic, stale, prev = 0, np.inf, 0, None
        peaks_left, resid_cands = list(seeds), []
        for n_try in range(1, n_max + 1):
            mus = [prev["model"].components(prev["x"])[c][0] for c in range(n_try - 1)] if prev else []
            cands = []
            for p in peaks_left + resid_cands:
                if (all(abs(float(_wrap_bins(p - m_, N_BINS))) >= 5 for m_ in mus)
                        and all(abs(float(_wrap_bins(p - q_, N_BINS))) >= 2 for q_ in cands)):
                    cands.append(p)
                if len(cands) == SC_N_CAND:
                    break
            if not cands:
                print(f"  no further candidate position — stopping at N = {n_try - 1}")
                break
            base = dict(base_start) if prev is None else dict(prev["start"])
            res, cand = None, None
            for p in cands:
                st_ = dict(base)
                st_.update(new_comp(n_try - 1, p))
                r_ = _sc_fit(D, nu, sig, mus + [p], st_)
                if r_ is not None and (res is None or r_["aic"] < res["aic"]):
                    res, cand = r_, p
            if res is None:
                break
            if cand in peaks_left:
                peaks_left.remove(cand)
            rep = _sc_significance(res)
            store[n_try] = (res, rep)
            delta = res["aic"] - best_aic
            mark = ""
            if res["aic"] < best_aic - 2:
                best_n, best_aic, stale, mark = n_try, res["aic"], 0, "  ←"
            else:
                stale += 1
            m_ = res["model"]
            print(f"  {n_try:>3}  {unroll(cand):>8.0f}  {res['aic']:>12.1f}  {delta:>+10.1f}  "
                  f"{res['chi2r']:>7.3f}  {np.exp(res['x'][m_.ix['lntau']]) * P_US:>11.2f}  "
                  f"{res['x'][m_.ix['alpha']]:>7.3f}  {min(r['sig'] for r in rep):>8.1f}  "
                  f"{sum(r['pinned'] for r in rep):>8d}{mark}")
            history.append((n_try, res["aic"], res["chi2r"]))
            prev = res

            # Next seed: the largest positive residual, read in the top-channel frame
            # (DM shift undone) over the upper half of the band, where tails are short.
            # Each channel's residual is put in template units (divided by its gain)
            # and weighted by its (S/N)^2 = (gain/noise)^2. Weighting by 1/noise^2
            # alone let a near-empty band-edge channel -- tiny noise AND tiny signal
            # -- outvote every other channel and seed the next component in its noise.
            undo = -res["x"][m_.ix["ddm"]] * m_.dm_bins
            R  = _sc_shift(D - res["fitted"], undo, m_.fk)
            hi_half = slice(nch // 2, nch)
            g_ = np.clip(res["gain"][hi_half], 0.0, None)
            w_ = g_ / sig[hi_half] ** 2
            rp = (R[hi_half] * w_[:, None]).sum(0) / max(float((g_ * w_).sum()), 1e-30)
            rp_sig = rp[OFF].std()
            mp = (_sc_shift(m_.template(res["x"]), undo, m_.fk)[hi_half] * (g_ * w_)[:, None]).sum(0)
            on = mp > 0.05 * mp.max()
            c_, _ = find_peaks(rp, height=3.0 * rp_sig, distance=8)
            resid_cands = [float(p) for p in sorted((int(p) for p in c_ if on[p]),
                                                    key=lambda p: -rp[p])]

            if stale >= 2 and n_try >= (args.force_n_components or 0):
                print("  AIC has not improved for two steps — stopping.")
                break

        if not store:
            print("ERROR: every scattering fit failed.")
            sys.exit(1)

        # ── significance scan ───────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  SIGNIFICANCE SCAN  (threshold {MIN_SIG_SC:g}σ of the band-averaged noise)")
        print(f"{'─'*60}")
        print(f"  {'N':>3}  {'min sig':>8}  {'κ pinned':>8}  {'idx pinned':>10}  {'tail-like':>9}  "
              f"{'blended':>7}  {'verdict':>14}")
        sig_n, scan_rows = 0, []
        for n_, (res_, rep_) in sorted(store.items()):
            ms_ = min(r["sig"] for r in rep_)
            npin = sum(r["pinned"] for r in rep_)
            nidx = sum(r["idx_pinned"] for r in rep_)
            ntail = sum(r["tail_like"] for r in rep_)
            nbl = sum(r["blended"] for r in rep_)
            ok_ = all(r["ok"] for r in rep_)
            if ok_:
                sig_n = n_
            why = ("ok" if ok_ else f"below {MIN_SIG_SC:g}σ" if ms_ < MIN_SIG_SC else
                   "κ pinned" if npin else "index pinned" if nidx else "tail-like")
            print(f"  {n_:>3}  {ms_:>8.1f}  {npin:>8d}  {nidx:>10d}  {ntail:>9d}  {nbl:>7d}  {why:>14}")
            m_n = res_["model"]
            scan_rows.append(dict(
                n_components=int(n_), aic=round(float(res_["aic"]), 2),
                chi2_reduced=round(float(res_["chi2r"]), 4),
                tau_ref_us=round(float(np.exp(res_["x"][m_n.ix["lntau"]]) * P_US), 3),
                alpha=round(float(res_["x"][m_n.ix["alpha"]]), 4),
                min_significance=round(float(ms_), 1), kappa_pinned=int(npin),
                index_pinned=int(nidx), tail_like=int(ntail), blended=int(nbl),
                verdict=why.replace("σ", " sigma")))
        print(f"  Rejected: below threshold, κ at the ceiling, a spectral index at its search "
              f"limit, or tail-like (broader than a brighter component it trails, and brightening "
              f"to low frequency much faster -- or, with shared indices, over twice as broad: "
              f"a scattering tail imitated by a von Mises).")
        print(f"  'blended' is reported, not rejected: in a portrait two overlapping terms "
              f"can describe how one feature changes shape with frequency.")

        if args.force_n_components:
            if args.force_n_components not in store:
                print(f"ERROR: N = {args.force_n_components} was not reached.")
                sys.exit(1)
            chosen_n = args.force_n_components
            print(f"\n  --force_n_components: reporting N = {chosen_n}.")
        elif sig_n:
            chosen_n = sig_n
            if sig_n != best_n:
                print(f"\n  AIC preferred N = {best_n}; reporting N = {sig_n}, the largest N whose "
                      f"components all pass the checks above. --force_n_components overrides.")
        else:
            chosen_n = best_n
        chosen, chosen_rep = store[chosen_n]

        # ── config-seeded cross-check ───────────────────────────────────────────
        seeded = None
        if cfg is not None and cfg.components:
            cc_all = list(cfg.components)
            ri = [c.name for c in cc_all].index(cfg.reference_component)
            cc_ord = [cc_all[ri]] + [c for i, c in enumerate(cc_all) if i != ri]
            bright = max(range(chosen_n), key=lambda i: chosen_rep[i]["A"])
            anchor = chosen_rep[bright]["mu"]
            ref_ph = float(cc_ord[0].phase_center_ref)
            g_ref0 = cfg_mid(cc_ord[0].spectral_index_amp)
            a_ref0 = cfg_mid(cc_ord[0].amplitude_ref)
            mus_s = [anchor + float(_wrap_bins((float(c.phase_center_ref) - ref_ph) * N_BINS, N_BINS))
                     for c in cc_ord]
            st = dict(base_start)
            for c, (cc, mu) in enumerate(zip(cc_ord, mus_s)):
                st[f"mu{c}"] = mu
                st[f"lnk{c}"] = np.log(np.clip(cfg_mid(cc.concentration_ref), 2.0, 0.9 * SC_KAPPA_MAX))
                if c:
                    st[f"lnA{c}"] = np.log(max(cfg_mid(cc.amplitude_ref) / a_ref0, 1e-3))
                    st[f"dg{c}"] = cfg_mid(cc.spectral_index_amp) - g_ref0
                st[f"s{c}"] = cfg_mid(cc.spectral_index_width)
            groups_s = _sc_groups(mus_s)
            for gname in set(groups_s):
                st[f"s_{gname}"] = float(np.mean([cfg_mid(cc.spectral_index_width)
                                                  for cc, g in zip(cc_ord, groups_s) if g == gname]))
            st["dg_I"] = 0.0
            print(f"\n{'─'*60}")
            print(f"  CONFIG-SEEDED CROSS-CHECK  (N = {len(cc_ord)}, seeded at the config's geometry)")
            print(f"{'─'*60}")
            s_res = _sc_fit(D, nu, sig, mus_s, st)
            if s_res is not None:
                seeded = (s_res, _sc_significance(s_res), [c.name for c in cc_ord])
                same = store.get(len(cc_ord))
                print(f"  {'model':<34} {'AIC':>12} {'χ²_red':>8}")
                print(f"  {'config-seeded, N=%d' % len(cc_ord):<34} {s_res['aic']:>12.1f} {s_res['chi2r']:>8.3f}")
                if same:
                    print(f"  {'free search, same N=%d' % len(cc_ord):<34} {same[0]['aic']:>12.1f} "
                          f"{same[0]['chi2r']:>8.3f}")
                    d_ = s_res["aic"] - same[0]["aic"]
                    if abs(d_) < 2:
                        print(f"  [OK]   same optimum (ΔAIC = {d_:+.1f}): the config's decomposition is "
                              f"what the data prefer.")
                    elif d_ < 0:
                        print(f"  [NOTE] the config-seeded fit is better by ΔAIC = {d_:+.1f}: the free "
                              f"search is stuck in a worse optimum at this N.")
                    else:
                        print(f"  [WARN] the free search is better by ΔAIC = {-d_:.1f} at the same N: "
                              f"the config's geometry is not what this archive prefers.")
                print(f"  {'free search, reported N=%d' % chosen_n:<34} {chosen['aic']:>12.1f} "
                      f"{chosen['chi2r']:>8.3f}")
        if args.seed_from_config:
            if seeded is None:
                print("\nERROR: --seed_from_config needs a --config, and the config-seeded "
                      "fit to succeed.")
                sys.exit(1)
            chosen, chosen_rep, names = seeded
            print(f"\n  --seed_from_config: reporting the config-seeded fit; component i is "
                  f"config component i.")
        else:
            names = [f"#{i}" for i in range(chosen["model"].n)]

        # ── results ─────────────────────────────────────────────────────────────
        m, x, e = chosen["model"], chosen["x"], chosen["err"]
        E = lambda n_: float(e[m.ix[n_]])
        lq = np.log(m.q)
        # The reference component's overall spectral slope, from the per-channel
        # gains. Channels more than 3 MADs off the line (band edges with almost no
        # signal, RFI) are left out of the slope: kept in, two or three of them set
        # the slope and inflate its error ~3x.
        okg = chosen["gain"] > 0
        lg_, lq_ = np.log(chosen["gain"][okg]), lq[okg]
        keep = np.ones(lg_.size, bool)
        for _ in range(10):
            r_ = lg_ - np.polyval(np.polyfit(lq_[keep], lg_[keep], 1), lq_)
            mad_ = 1.4826 * np.median(np.abs(r_[keep] - np.median(r_[keep])))
            new = np.abs(r_ - np.median(r_[keep])) < 3.0 * mad_
            if (new == keep).all() or new.sum() < 4:
                break
            keep = new
        pg, pgc = np.polyfit(lq_[keep], lg_[keep], 1, cov=True)
        lres = lg_ - np.polyval(pg, lq_)
        g0, g0_err = float(pg[0]), float(np.sqrt(pgc[0, 0]))
        n_clip = int((~keep).sum())
        bp_mad = float(1.4826 * np.median(np.abs(lres - np.median(lres))))
        A_max = max(r["A"] for r in chosen_rep)
        comps_sc = []
        for c, ((mu, kap, s, A, dg), r) in enumerate(zip(m.components(x), chosen_rep)):
            s_name = f"s{c}" if m.mode == "free" else f"s_{m.groups[c]}"
            g_name = (f"dg{c}" if m.mode == "free" else "dg_I") if (c and (m.mode == "free" or m.groups[c] == "I")) else None
            comps_sc.append(dict(
                name=names[c], mu=unroll(mu), dmu=E(f"mu{c}"), k=kap, dk_frac=E(f"lnk{c}"),
                A=A, amp_rel=A / A_max, dA_frac=0.0 if c == 0 else E(f"lnA{c}"),
                gamma=g0 + dg, dgamma=float(np.hypot(g0_err, E(g_name) if g_name else 0.0)),
                dgamma_rel=E(g_name) if g_name else 0.0,
                s=s, ds=E(s_name),
                sig=r["sig"], pinned=r["pinned"], blended=r["blended"], fwhm=_fwhm_of(kap),
                idx_pinned=r["idx_pinned"], tail_like=r["tail_like"]))
        if cfg is None:
            # No config names to borrow: C0 is the brightest, C1 the next, ...
            for rank, i in enumerate(sorted(range(len(comps_sc)), key=lambda i: -comps_sc[i]["A"])):
                comps_sc[i]["name"] = f"C{rank}"
        tau_bins = float(np.exp(x[m.ix["lntau"]]))
        alpha, ddm = float(x[m.ix["alpha"]]), float(x[m.ix["ddm"]])
        c_ta = float(chosen["cov"][m.ix["lntau"], m.ix["alpha"]])
        nu_piv = REF_FREQ_MHZ * np.exp(-c_ta / chosen["cov"][m.ix["alpha"], m.ix["alpha"]])
        corr_ta = c_ta / np.sqrt(chosen["cov"][m.ix["lntau"], m.ix["lntau"]]
                                 * chosen["cov"][m.ix["alpha"], m.ix["alpha"]])
        prof = D.mean(axis=0)
        snr_avg = float(prof.max() / prof[OFF].std())

        print(f"\n{'═'*60}")
        print(f"  COMPONENTS  (intrinsic, unscattered; κ and amplitude at "
              f"nu_ref = {REF_FREQ_MHZ:.3f} MHz)")
        print(f"{'═'*60}")
        print(f"  {'':<6} {'phase':>10} {'σ_mu':>5} {'amp_rel':>8} {'κ':>8} {'FWHM':>6} "
              f"{'gamma':>7} {'w.idx':>7} {'sig':>7}  flags")
        order = range(len(comps_sc)) if args.seed_from_config else \
            sorted(range(len(comps_sc)), key=lambda i: -comps_sc[i]["A"])
        for i in order:
            c = comps_sc[i]
            fl = " ".join(f for f, on_ in (("κ-PINNED", c["pinned"]), ("INDEX-PINNED", c["idx_pinned"]),
                                            ("TAIL-LIKE", c["tail_like"]), ("blended", c["blended"]),
                                            (f"NOISE(<{MIN_SIG_SC:g}σ)", c["sig"] < MIN_SIG_SC)) if on_)
            print(f"  {c['name']:<6} {c['mu']/N_BINS:>10.6f} {c['dmu']:>5.2f} {c['amp_rel']:>8.4f} "
                  f"{c['k']:>8.1f} {c['fwhm']:>6.2f} {c['gamma']:>+7.2f} {c['s']:>+7.2f} "
                  f"{c['sig']:>7.1f}  {fl}")
        print(f"  gamma = amplitude spectral index; w.idx = width index "
              f"(κ(nu) = κ_ref (nu/nu_ref)^w.idx). Component 0's gamma comes from the gains.")

        print(f"\n{'─'*60}")
        print(f"  SCATTERING AND DM")
        print(f"{'─'*60}")
        print(f"  tau(nu_ref)  = {tau_bins*P_US:.2f} us ± {E('lntau')*100:.1f}%  "
              f"= {tau_bins*P_US*1e-3:.5f} ms = {tau_bins:.2f} bins")
        print(f"  tau at {nu.min():.0f} / {nu.max():.0f} MHz = "
              f"{tau_bins*(nu.min()/REF_FREQ_MHZ)**alpha:.1f} / "
              f"{tau_bins*(nu.max()/REF_FREQ_MHZ)**alpha:.1f} bins")
        print(f"  alpha        = {alpha:.3f} ± {E('alpha'):.3f}  (formal)")
        print(f"  dDM          = {ddm:+.5f} ± {E('ddm'):.5f}  -> DM {sc_dm_header + ddm:.5f} "
              f"(archive {sc_dm_header:.5f})")
        grid = cfg.freqs_mhz if cfg else sc_freqs
        print(f"  corr(ln tau, alpha) = {corr_ta:+.2f}; zero at {nu_piv:.1f} MHz -- a "
              f"reference_freq_mhz there decouples tau_ref and alpha "
              f"(nearest channel {grid[np.argmin(np.abs(grid - nu_piv))]:.4f} MHz)")
        print(f"  reduced χ² = {chosen['chi2r']:.3f}")

        sb = _sc_tau_per_subband(chosen, args.n_subbands)
        print(f"\n  tau per subband (profile held at the best fit) vs the power law:")
        for nu_k, t_k, e_k in sb:
            pl = tau_bins * (nu_k / REF_FREQ_MHZ) ** alpha
            print(f"    {nu_k:7.1f} MHz   {t_k*P_US:8.2f} us ± {e_k*100:4.1f}%   "
                  f"power law {pl*P_US:8.2f} us   ratio {t_k/pl:.3f}")

        print(f"\n{'─'*60}")
        print(f"  FOR THE CONFIG'S instrument/simulation BLOCKS")
        print(f"{'─'*60}")
        print(f"  sigma_bandpass  ~ {bp_mad:.2f}   (MAD of ln gain about its power law; "
              f"std {lres.std():.2f})")
        if n_clip:
            print(f"  overall amplitude slope (component 0's gamma {g0:+.2f} ± {g0_err:.2f}) from "
                  f"{int(keep.sum())} channels; {n_clip} more than 3 MAD off the line left out: "
                  f"{', '.join(f'{v:.1f}' for v in m.nu[okg][~keep])} MHz")
        print(f"  snr (avg_peak)  = {snr_avg:.0f}   (band-averaged peak / its off-pulse rms)")
        print(f"  noise rms varies x{sig.max()/sig.min():.1f} across channels "
              f"(the fit weights each channel by its own noise)")

        if cfg is not None:
            # ── pairing with the config, offsets and priors ─────────────────────────
            cfg_comps = list(cfg.components)
            by_name = {c.name: c for c in cfg_comps}
            if args.seed_from_config:
                pairs = [(by_name[c["name"]], c) for c in comps_sc]
            else:
                ref_c = max(comps_sc, key=lambda c: c["A"])
                off_ = float(_wrap_bins(ref_c["mu"] - float(by_name[cfg.reference_component].phase_center_ref)
                                        * N_BINS, N_BINS))
                pairs, used_ = [], set()
                for cc in cfg_comps:
                    tgt = float(cc.phase_center_ref) * N_BINS + off_
                    cand_ = [(abs(float(_wrap_bins(c["mu"] - tgt, N_BINS))), i)
                             for i, c in enumerate(comps_sc) if i not in used_]
                    if cand_:
                        i_ = min(cand_)[1]
                        used_.add(i_)
                        pairs.append((cc, comps_sc[i_]))
                if len(comps_sc) != len(cfg_comps):
                    print(f"\n  [WARN] fit has {len(comps_sc)} components, config {len(cfg_comps)}: "
                          f"check the pairing below.")

            spec = {p.name: p for p in list(cfg.inferred) + list(cfg.nuisance)}
            rows = []

            def chk(label, val, ref):
                if isinstance(ref, str) and ref.startswith("$") and ref[1:] in spec:
                    p_ = spec[ref[1:]]
                    pct = (val - p_.low) / (p_.high - p_.low) * 100
                    stt = "OUTSIDE PRIOR" if not p_.low <= val <= p_.high else (
                        "at edge" if pct < 5 or pct > 95 else "ok")
                    rows.append((label, p_.name, val, p_.low, p_.high, pct, stt))

            ref_pair = next((pr for pr in pairs if pr[0].name == cfg.reference_component), None)
            toa_off = (float(_wrap_bins(ref_pair[1]["mu"] - float(ref_pair[0].phase_center_ref) * N_BINS,
                                        N_BINS)) / N_BINS) if ref_pair else 0.0
            for cc, fc in pairs:
                chk(f"{cc.name} concentration", fc["k"], cc.concentration_ref)
                chk(f"{cc.name} amplitude", fc["amp_rel"], cc.amplitude_ref)
                chk(f"{cc.name} gamma", fc["gamma"], cc.spectral_index_amp)
                chk(f"{cc.name} width index", fc["s"], cc.spectral_index_width)
            for key, val in (("tau_ms", tau_bins * P_US * 1e-3), ("spectral_index", alpha)):
                ent = cfg.scattering.get(key)
                if isinstance(ent, dict) and ent.get("type") == "free":
                    chk(f"scattering {key}", val, "$" + ent["param"])
            chk("dDM", ddm, "$" + cfg.dm_param)
            chk("toa (phase offset)", toa_off, "$toa")

            print(f"\n{'─'*60}")
            print(f"  PAIRED WITH {os.path.basename(cfg.path)}  "
                  f"({'by construction' if args.seed_from_config else 'anchored on ' + cfg.reference_component})")
            print(f"{'─'*60}")
            print(f"  {'component':<10} {'config phase':>13} {'fitted phase':>13} {'Δ (bins)':>9}")
            for cc, fc in pairs:
                dd = float(_wrap_bins(fc["mu"] - float(cc.phase_center_ref) * N_BINS, N_BINS))
                print(f"  {cc.name:<10} {float(cc.phase_center_ref):>13.8f} {fc['mu']/N_BINS:>13.8f} {dd:>+9.2f}")
            print(f"\n  fitted values vs the config's priors:")
            if not rows:
                print("    (no sampled parameters to compare)")
            for lab, pn, v, plo, phi, pct, stt in rows:
                print(f"    {lab:<26} {v:>10.4f}  {pn:<14} U[{plo:g}, {phi:g}]  {pct:5.1f}%  {stt}")

            # ── --emit-yaml ─────────────────────────────────────────────────────────
            if args.emit_yaml:
                shift = 0.0
                if args.recentre_on:
                    hit = next((fc for cc, fc in pairs if cc.name == args.recentre_on), None)
                    if hit is None:
                        print(f"ERROR: --recentre-on '{args.recentre_on}' was not matched to a "
                              f"fitted component.")
                        sys.exit(1)
                    shift = (hit["mu"] / N_BINS - float(by_name[args.recentre_on].phase_center_ref) + 0.5) % 1.0 - 0.5
                print(f"\n{'═'*60}")
                print(f"  components: BLOCK FOR {os.path.basename(cfg.path)}  (--scattering)")
                print(f"{'═'*60}")
                print(f"# Fitted from: {os.path.basename(args.infile)} with --scattering")
                print(f"# Intrinsic (unscattered) components. amplitude_ref / concentration_ref are")
                print(f"# defined AT reference_freq_mhz = {REF_FREQ_MHZ:.11f}.")
                if args.recentre_on:
                    print(f"# Re-centred on '{args.recentre_on}': {shift:+.8f} phase subtracted from every centre.")
                print("components:")
                for cc, fc in pairs:
                    print(f"  - name:                  {cc.name}")
                    print(f"    amplitude_ref:         {cc.amplitude_ref}        # fitted: {fc['amp_rel']:.4f}")
                    print(f"    phase_center_ref:      {(fc['mu']/N_BINS - shift) % 1.0:.8f}")
                    print(f"    concentration_ref:     {cc.concentration_ref}        # fitted: {fc['k']:.1f}")
                    print(f"    spectral_index_amp:    {cc.spectral_index_amp}        # fitted: {fc['gamma']:+.3f}")
                    print(f"    spectral_index_width:  {cc.spectral_index_width}        # fitted: {fc['s']:+.3f}")
                    print(f"    spectral_index_center: {cc.spectral_index_center}")
                    print(f"    sigma_broadening_bins: {cc.sigma_broadening_bins}")
                    print()
                unmatched = [c for c in comps_sc if all(c is not fc for _, fc in pairs)]
                for c in unmatched:
                    print(f"  # unmatched fitted component: phase {(c['mu']/N_BINS - shift) % 1.0:.8f}  "
                          f"amp {c['amp_rel']:.4f}  kappa {c['k']:.1f}  gamma {c['gamma']:+.3f}  "
                          f"w.idx {c['s']:+.3f}")
                print(f"# Fitted scattering, for the scattering: block and its priors:")
                print(f"#   tau_ms at {REF_FREQ_MHZ:.5f} MHz = {tau_bins*P_US*1e-3:.6f}   "
                      f"spectral_index = {alpha:.3f}")
                print(f"# sigma_bandpass ~ {bp_mad:.2f}; avg_peak S/N {snr_avg:.0f}; "
                      f"dDM {ddm:+.5f} vs archive DM {sc_dm_header:.5f}")
                print()
        else:
            _sc_write_outputs(comps_sc, dict(
                n_components=int(chosen["model"].n), aic_preferred_n=int(best_n),
                significance_n=int(sig_n), chi2_reduced=float(chosen["chi2r"]),
                aic=float(chosen["aic"]), spectral_indices=args.spectral_indices,
                tau_bins=tau_bins, tau_err_frac=E("lntau"), alpha=alpha, alpha_err=E("alpha"),
                ddm=ddm, ddm_err=E("ddm"), corr_tau_alpha=float(corr_ta), nu_pivot=float(nu_piv),
                gamma0=g0, gamma0_err=g0_err, n_clip=n_clip,
                clipped_mhz=[round(float(v), 3) for v in m.nu[okg][~keep]],
                bp_mad=bp_mad, bp_std=float(lres.std()), snr=snr_avg,
                noise_ratio=float(sig.max() / sig.min()),
                off_pulse_bins=[int(unroll(OFF[0])), int(unroll(OFF[-1]))],
                dead_mhz=[round(float(v), 3) for v in dead_mhz],
                tau_subbands=sb, scan=scan_rows))


        # ── plot ────────────────────────────────────────────────────────────────
        import matplotlib.gridspec as gridspec
        unr = lambda a: np.roll(a, -sc_roll, axis=-1)
        Rn = (D - chosen["fitted"]) / sig[:, None]
        fig = plt.figure(figsize=(17, 14))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1.0, 1.25, 1.0], hspace=0.35, wspace=0.28)
        ext = [0, N_BINS, nu.min(), nu.max()]
        for j, (img, ttl, kw) in enumerate(((D, "data", {}), (chosen["fitted"], "model", {}),
                                            (Rn, "(data − model) / noise",
                                             dict(cmap="RdBu_r", vmin=-5, vmax=5)))):
            a = fig.add_subplot(gs[0, j])
            im = a.imshow(unr(img), aspect="auto", origin="lower", extent=ext,
                          interpolation="nearest", **kw)
            a.set_title(ttl, fontsize=10); a.set_xlabel("Phase bin")
            if j == 0:
                a.set_ylabel("Frequency (MHz)")
            plt.colorbar(im, ax=a, fraction=0.046, pad=0.02)

        a = fig.add_subplot(gs[1, :2])
        edges = np.linspace(0, nch, args.n_subbands + 1).astype(int)
        cmap = plt.cm.viridis(np.linspace(0, 1, args.n_subbands))
        bb = np.arange(N_BINS)
        for k in range(args.n_subbands):
            sl = slice(edges[k], edges[k + 1])
            if sl.stop <= sl.start:
                continue
            dk, mk = unr(D[sl].mean(0)), unr(chosen["fitted"][sl].mean(0))
            s_ = 1.0 / mk.max()
            a.plot(bb, dk * s_ + 1.1 * k, color="k", lw=0.7)
            a.plot(bb, mk * s_ + 1.1 * k, color=cmap[k], lw=1.3)
            a.text(N_BINS - 2, 1.1 * k + 0.5, f"{np.exp(np.log(nu[sl]).mean()):.0f} MHz",
                   ha="right", fontsize=8)
        a.set_title("Subband profiles: data (black) and scattered model", fontsize=10)
        a.set_xlabel("Phase bin"); a.set_yticks([])

        a = fig.add_subplot(gs[1, 2])
        nus = np.array([s_[0] for s_ in sb]); tus = np.array([s_[1] for s_ in sb]) * P_US
        a.errorbar(nus, tus, yerr=tus * np.array([s_[2] for s_ in sb]), fmt="o", label="tau per subband")
        nn = np.linspace(nu.min(), nu.max(), 200)
        a.plot(nn, tau_bins * P_US * (nn / REF_FREQ_MHZ) ** alpha, "r-",
               label=f"joint fit: alpha = {alpha:.2f}")
        a.axhline(P_US, color="0.5", ls=":", label="1 phase bin")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("Frequency (MHz)"); a.set_ylabel("tau (us)")
        a.set_title(f"tau = {tau_bins*P_US:.1f} us at {REF_FREQ_MHZ:.1f} MHz", fontsize=10)
        a.legend(fontsize=7)

        a = fig.add_subplot(gs[2, :2])
        jr = int(np.argmin(np.abs(nu - REF_FREQ_MHZ)))
        sh_r = ddm * m.dm_bins[jr]
        tot = np.zeros(N_BINS)
        for c, part in enumerate(m.intrinsic(x)):
            y = unr(_sc_shift(part[jr:jr + 1], [sh_r], m.fk)[0]) * chosen["gain"][jr]
            tot += y
            a.fill_between(bb, y, alpha=0.25, label=f"{comps_sc[c]['name']} intrinsic")
        a.plot(bb, tot, "b--", lw=1.0, label="intrinsic sum")
        a.plot(bb, unr(D[jr]) - chosen["base"][jr], "k", lw=0.7, label="data")
        a.plot(bb, unr(chosen["fitted"][jr]) - chosen["base"][jr], "r", lw=1.3,
               label=f"scattered model (tau {tau_bins*(nu[jr]/REF_FREQ_MHZ)**alpha:.1f} bins)")
        a.set_title(f"Channel nearest nu_ref ({nu[jr]:.2f} MHz): components before and after "
                    f"scattering", fontsize=10)
        a.set_xlabel("Phase bin"); a.legend(fontsize=7, ncol=2)

        a = fig.add_subplot(gs[2, 2])
        aics = np.array([h[1] for h in history])
        a.semilogy([h[0] for h in history], aics - aics.min() + 1.0, "o-", color="steelblue")
        a.axvline(chosen["model"].n, color="crimson", ls="--", label=f"reported N = {chosen['model'].n}")
        a.set_xlabel("N components"); a.set_ylabel("AIC − min(AIC) + 1")
        a.set_title("Greedy search", fontsize=10); a.legend(fontsize=7)

        fig.suptitle(f"{os.path.basename(args.infile)} — scattered von Mises fit, "
                     f"{chosen['model'].n} components, reduced χ² {chosen['chi2r']:.2f}", fontsize=11)
        out_png = os.path.join(outdir, f"{stem}_vonmises_scattering_fit.png")
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved → {out_png}")


    # ─────────────────────────────────────────────────────────────────────────────
    # --scattering without a config: everything comes from the archive itself.
    # The archive is used as it is -- only time/polarisation-averaged, baseline
    # removed and dedispersed at its own DM, plus the --nchan scrunch if asked.
    # ─────────────────────────────────────────────────────────────────────────────

    if CONFIG_FREE:
        if ext not in (".ar", ".fits", ".calib", ".rf", ".zap"):
            print(f"ERROR: without --config, fit_vonmises needs a PSRCHIVE archive (.ar, .fits, "
                  f"...): a {ext} file does not carry the period. Give the archive, or --config.")
            sys.exit(1)
        try:
            import psrchive
        except ImportError:
            print("ERROR: psrchive Python bindings not available (use the penv_psr env).")
            sys.exit(1)
        arch = psrchive.Archive_load(args.infile)
        _mjd0 = float(arch.start_time().in_days())    # tscrunch moves start_time
        arch.tscrunch()
        arch.pscrunch()
        arch.remove_baseline()
        arch.dedisperse()
        _n0 = arch.get_nchan()
        _nt = args.nchan or _n0
        if _nt > _n0 or _n0 % _nt:
            print(f"ERROR: --nchan {_nt} does not divide the archive's {_n0} channels.")
            sys.exit(1)
        if _nt < _n0:
            arch.fscrunch(_n0 // _nt)
        _si          = arch.get_Integration(0)
        sc_freqs     = np.array([_si.get_centre_frequency(i) for i in range(arch.get_nchan())])
        sc_portrait  = arch.get_data()[0, 0].astype(float)
        if sc_freqs.size > 1 and sc_freqs[0] > sc_freqs[-1]:
            sc_freqs, sc_portrait = sc_freqs[::-1].copy(), sc_portrait[::-1].copy()
        sc_dm_header = float(arch.get_dispersion_measure())
        _cf, _bw     = arch.get_centre_frequency(), abs(arch.get_bandwidth())

        N_BINS       = arch.get_nbin()
        N_CHAN       = sc_freqs.size
        FREQ_LOW_MHZ = _cf - _bw / 2.0
        FREQ_HI_MHZ  = _cf + _bw / 2.0
        _want        = args.ref_freq or np.sqrt(FREQ_LOW_MHZ * FREQ_HI_MHZ)
        REF_FREQ_MHZ = float(sc_freqs[np.argmin(np.abs(sc_freqs - _want))])
        _eph = arch.get_ephemeris() if arch.has_ephemeris() else None

        def _par(key):
            try:
                return str(_eph.get_value(key)).strip() if _eph is not None else ""
            except Exception:
                return ""

        _hd = arch.get_coordinates().getHMSDMS().split()
        SC.update(period_s=float(_si.get_folding_period()), nu_top=float(sc_freqs.max()),
                  source=arch.get_source(), freq_low=FREQ_LOW_MHZ, freq_high=FREQ_HI_MHZ,
                  nu_c=float(_cf), telescope=arch.get_telescope(),
                  mjd=_mjd0, tobs=float(arch.integration_length()),
                  raj=_par("RAJ") or _hd[0], decj=_par("DECJ") or _hd[1],
                  f0_par=_par("F0"), dm_par=_par("DM"), units=_par("UNITS"))

        print(f"\nLoaded {args.infile} — no config: everything below is from the archive")
        print(f"  {arch.get_source()}  P = {SC['period_s']*1e3:.10f} ms  DM = {sc_dm_header:.6f}")
        print(f"  {_n0} channels" + (f" scrunched to {_nt}" if _nt < _n0 else "")
              + f" × {N_BINS} bins, {FREQ_LOW_MHZ:.4f}–{FREQ_HI_MHZ:.4f} MHz")
        print(f"  reference frequency {REF_FREQ_MHZ:.5f} MHz "
              + (f"(channel nearest --ref-freq {args.ref_freq:g})" if args.ref_freq
                 else "(channel nearest the band's geometric centre; --ref-freq to change)"))
        if args.scattering:
            _scattering_main()
            sys.exit(0)


    # ─────────────────────────────────────────────────────────────────────────────
    # Load data and frequency-scrunch
    # ─────────────────────────────────────────────────────────────────────────────

    def _ref_window_slice(portrait, freqs, idx):
        """Average `--ref_window` channels centred on `idx`.

    Returns (lo, hi, mean_profile). A single reference channel carries
    sqrt(n_chan) times the band-scrunched noise, which leaves the weakest
    component unconstrained — on the German archive it drove Inter's kappa to 3.1,
    below its own prior floor, purely from noise. Averaging a narrow window
    recovers S/N while keeping the spectral-weighting bias negligible: over a
    few channels nu varies by well under a percent, so amplitude ~ nu^gamma and
    width ~ nu^index are essentially constant across the window.
    """
        w  = max(1, int(args.ref_window))
        h  = w // 2
        lo = max(0, idx - h)
        hi = min(portrait.shape[0], lo + w)
        lo = max(0, hi - w)
        return lo, hi, portrait[lo:hi, :].mean(axis=0)


    archive_freqs = None   # set on the .ar path; used by --emit-yaml

    if ext == ".pt":
        d = torch.load(args.infile, map_location="cpu", weights_only=False)
        portrait = d["portrait"].numpy()                # (n_chan, n_bins) — normalised to max=1
        fscrunch = portrait.mean(axis=0)
        if portrait.shape[0] != N_CHAN or portrait.shape[1] != N_BINS:
            print(f"ERROR: portrait is {portrait.shape} but config "
                  f"{os.path.basename(cfg.path)} describes {N_CHAN} ch x {N_BINS} bins.")
            sys.exit(1)
        _meta        = d.get("metadata") or {}
        sc_portrait  = portrait.astype(float)
        sc_freqs     = np.asarray(_meta.get("channel_freqs_mhz", cfg.freqs_mhz), dtype=float)
        sc_dm_header = float(_meta.get("dm_header", cfg.dm_header))
        # Reference-frequency channel: nearest channel centre to the config's
        # reference frequency, using the config's true channel grid.
        _freqs   = cfg.freqs_mhz
        ref_idx  = int(np.argmin(np.abs(_freqs - REF_FREQ_MHZ)))
        ref_lo, ref_hi, ref_chan = _ref_window_slice(portrait, _freqs, ref_idx)
        src_label = f"{args.infile}  [portrait averaged over {N_CHAN} channels]"
        print(f"\nLoaded .pt file — portrait shape {portrait.shape}")
        print(f"Reference channel index: {ref_idx}  "
              f"({_freqs[ref_idx]:.3f} MHz; config reference {REF_FREQ_MHZ:.3f} MHz, "
              f"offset {_freqs[ref_idx]-REF_FREQ_MHZ:+.3f} MHz)")
        print(f"Reference window: channels {ref_lo}–{ref_hi-1} "
              f"({_freqs[ref_lo]:.3f}–{_freqs[ref_hi-1]:.3f} MHz, {ref_hi-ref_lo} averaged)")

    elif ext in (".ar", ".fits", ".calib", ".rf", ".zap"):
        try:
            import psrchive
        except ImportError:
            print("ERROR: psrchive Python bindings not available. Use a .pt file.")
            sys.exit(1)
        arch = psrchive.Archive_load(args.infile)
        arch.tscrunch()
        arch.pscrunch()
        arch.remove_baseline()
        arch.dedisperse()
        raw          = arch.get_data()                  # (1, 1, n_chan, n_bin)
        portrait_raw = raw[0, 0, :, :]                  # (n_chan, n_bin)
        n_bin        = portrait_raw.shape[1]
        if n_bin != N_BINS:
            print(f"ERROR: archive has {n_bin} phase bins; expected {N_BINS}.")
            sys.exit(1)
        n_chan_ar    = portrait_raw.shape[0]
        center_freq  = arch.get_centre_frequency()
        bw           = abs(arch.get_bandwidth())
        freq_lo      = center_freq - bw / 2

        # TRUE per-channel centres from the archive, not a linspace across the band.
        # Real LOFAR channels are unevenly spaced (German stations: 0.974-1.370 MHz
        # between neighbours), so the old `freq_lo + i*bw/n_chan` estimate could pick
        # a channel up to ~0.2 MHz away from the intended reference frequency.
        _subint       = arch.get_Integration(0)
        archive_freqs = np.array([_subint.get_centre_frequency(i)
                                  for i in range(n_chan_ar)], dtype=float)

        # Ascending order, matching the simulator and real_archive_loader.py
        if archive_freqs.size > 1 and archive_freqs[0] > archive_freqs[-1]:
            archive_freqs = archive_freqs[::-1].copy()
            portrait_raw  = portrait_raw[::-1, :].copy()
            print("  [FLIP] descending channel order detected → flipped to ascending")

        ref_idx       = int(np.argmin(np.abs(archive_freqs - REF_FREQ_MHZ)))
        ref_lo, ref_hi, ref_chan = _ref_window_slice(portrait_raw, archive_freqs, ref_idx)
        fscrunch      = portrait_raw.mean(axis=0)
        src_label     = f"{args.infile}  [fscrunched from {n_chan_ar} channels]"

        _d = np.diff(archive_freqs)
        print(f"\nLoaded archive — {n_chan_ar} channels × {n_bin} bins")
        print(f"  band            : {archive_freqs[0]:.4f} – {archive_freqs[-1]:.4f} MHz "
              f"(centres); centre_freq={center_freq:.5f}, BW={bw:.4f}")
        if _d.size:
            print(f"  channel spacing : min={_d.min():.4f} max={_d.max():.4f} "
                  f"mean={_d.mean():.4f} std={_d.std():.4f} MHz")
            if _d.std() > 0.01 * _d.mean():
                print(f"  [NOTE] channels are NOT evenly spaced. Put the true centres in the "
                      f"config as instrument.channel_freqs_mhz — a linspace model would "
                      f"misplace the simulated DM sweep. Use --emit-yaml to print them.")
        print(f"  reference channel: index {ref_idx}  ({archive_freqs[ref_idx]:.3f} MHz; "
              f"{'config reference' if cfg else 'reference'} {REF_FREQ_MHZ:.3f} MHz, "
              f"offset {archive_freqs[ref_idx]-REF_FREQ_MHZ:+.3f} MHz)")
        print(f"  reference window : channels {ref_lo}–{ref_hi-1} "
              f"({archive_freqs[ref_lo]:.3f}–{archive_freqs[ref_hi-1]:.3f} MHz, "
              f"{ref_hi-ref_lo} ch averaged; sqrt(N) noise gain "
              f"{np.sqrt(ref_hi-ref_lo):.1f}x)")
        # real_archive_loader.py fscrunches by any integer factor, so only a channel
        # count that cannot reach N_CHAN means the config is wrong for the instrument.
        if n_chan_ar < N_CHAN or n_chan_ar % N_CHAN:
            print(f"  [WARN] archive has {n_chan_ar} channels but config "
                  f"{os.path.basename(cfg.path)} declares n_freq_channels={N_CHAN}. "
                  f"The fit itself is frequency-scrunched so this does not affect it, "
                  f"but the config is wrong for this instrument."
                  + (f"  ({n_chan_ar} is prime — it cannot be fscrunched to {N_CHAN}.)"
                     if all(n_chan_ar % k for k in range(2, int(n_chan_ar**0.5) + 1))
                     and n_chan_ar > 1 else ""))

        if args.scattering:
            # The portrait at the config's channel count, scrunched by PSRCHIVE as
            # real_archive_loader.py does: weighted, so a zapped channel does not
            # dilute its live partner, and with the true scrunched centres.
            if n_chan_ar < N_CHAN or n_chan_ar % N_CHAN:
                print(f"ERROR: --scattering needs the archive's {n_chan_ar} channels to "
                      f"scrunch to the config's {N_CHAN}.")
                sys.exit(1)
            _sa = arch.clone()
            if n_chan_ar > N_CHAN:
                _sa.fscrunch(n_chan_ar // N_CHAN)
            _si          = _sa.get_Integration(0)
            sc_freqs     = np.array([_si.get_centre_frequency(i)
                                     for i in range(_sa.get_nchan())], dtype=float)
            sc_portrait  = _sa.get_data()[0, 0].astype(float)
            if sc_freqs.size > 1 and sc_freqs[0] > sc_freqs[-1]:
                sc_freqs, sc_portrait = sc_freqs[::-1].copy(), sc_portrait[::-1].copy()
            sc_dm_header = float(arch.get_dispersion_measure())

    else:
        print(f"Unsupported file type: {ext}  (expected .ar, .fits, .calib, .rf, .zap or .pt)")
        sys.exit(1)


    # ─────────────────────────────────────────────────────────────────────────────
    # Normalise both profiles so the main peak = 1
    # ─────────────────────────────────────────────────────────────────────────────

    fscrunch = fscrunch.astype(float)
    fscrunch /= fscrunch.max()

    ref_chan = ref_chan.astype(float)
    rc_max   = ref_chan.max()
    if rc_max > 1e-12:
        ref_chan /= rc_max


    # ─────────────────────────────────────────────────────────────────────────────
    # Centre the profile before fitting
    #
    # Everything below — find_peaks, estimate_fwhm, the mu bounds — is easier and
    # safer when the pulse sits in the middle of the array rather than straddling
    # phase 0. A profile that wraps around the edge otherwise gets split into two
    # partial peaks by find_peaks and produces components pinned at the boundary.
    #
    # Rolling is harmless because only RELATIVE phases carry physical meaning: a
    # constant shift of every component is exactly degenerate with the `toa`
    # parameter. The roll is undone before anything is reported, so all printed
    # phases are in the original archive's phase frame.
    # ─────────────────────────────────────────────────────────────────────────────

    _fscrunch_orig = fscrunch.copy()
    _ref_chan_orig = ref_chan.copy()

    roll = int(N_BINS // 2 - int(np.argmax(fscrunch)))
    if roll % N_BINS:
        fscrunch = np.roll(fscrunch, roll)
        ref_chan = np.roll(ref_chan, roll)
        print(f"\n  Profile centred for fitting: rolled by {roll:+d} bins "
              f"({roll/N_BINS:+.6f} phase). Undone before reporting.")

    if args.scattering:      # with a config; without one it ran and exited above
        SC.update(period_s=cfg.period_s, nu_top=float(cfg.freqs_mhz.max()))
        _scattering_main()
        sys.exit(0)


    # ─────────────────────────────────────────────────────────────────────────────
    # Detect candidate peaks
    # ─────────────────────────────────────────────────────────────────────────────

    print(f"\n{'─'*60}")
    print(f"  Peak detection on fscrunched profile (threshold > 5% of peak)")
    print(f"{'─'*60}")

    peak_bins, props = find_peaks(fscrunch,
                                   height=0.05,
                                   prominence=0.03,
                                   distance=10)
    peak_amps = fscrunch[peak_bins]

    # Sort peaks by amplitude (brightest first)
    order      = np.argsort(peak_amps)[::-1]
    peak_bins  = peak_bins[order]
    peak_amps  = peak_amps[order]

    print(f"  Found {len(peak_bins)} candidate peaks:")
    for pb, pa in zip(peak_bins, peak_amps):
        print(f"    bin {pb:4d}  ({pb/N_BINS:.6f} phase)  amplitude = {pa:.4f}")


    # ─────────────────────────────────────────────────────────────────────────────
    # Off-pulse noise floor (bins where profile < 5% of peak)
    # ─────────────────────────────────────────────────────────────────────────────

    off_pulse_mask = fscrunch < 0.05
    on_pulse       = ~off_pulse_mask
    off_pulse_rms  = fscrunch[off_pulse_mask].std()
    print(f"\n  Off-pulse noise floor (RMS) = {off_pulse_rms:.5f}")
    print(f"  Whiteness threshold (3σ)   = {3*off_pulse_rms:.5f}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Fit 1 … max_components; iterative residual-guided seeding; AIC + whiteness
    # ─────────────────────────────────────────────────────────────────────────────

    best_aic     = np.inf
    best_popt    = None
    best_perr    = None
    best_fitted  = None
    best_n       = 0

    aic_history  = []   # (n, aic, rms, max_resid, is_white)
    fit_store    = {}   # n -> (popt, perr, fitted), for the significance scan below

    KAPPA_CEIL = 3000.0   # must match the upper bound in fit_n_components

    # Seed list grows iteratively: start from detected peaks, then residual peaks
    seed_bins = list(peak_bins)

    # The free search ALWAYS runs, whether or not a config is available to seed
    # from. The config-seeded fit below is an additional CROSS-CHECK, never a
    # substitute: a measurement that changes depending on whether you passed the
    # config is not a measurement of the profile.
    print(f"\n  AIC model selection (max_components={args.max_components}):")
    print(f"  {'N':>3}  {'AIC':>10}  {'ΔAIC':>8}  {'RMS resid':>10}  "
          f"{'Max |resid|':>12}  {'White?':>7}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*7}")

    for n_try in range(1, args.max_components + 1):
        popt, perr, fitted = fit_n_components(fscrunch, n_try, seed_bins[:n_try])
        if popt is None:
            break

        a         = compute_aic(fscrunch, fitted, 3 * n_try + 1)
        resid     = fscrunch - fitted
        rms       = resid.std()
        max_resid = np.abs(resid[on_pulse]).max()
        is_white  = bool(max_resid < 3 * off_pulse_rms)
        delta     = a - best_aic

        mark = ""
        if a < best_aic - 2:
            mark = "  ←"
            best_aic    = a
            best_popt   = popt
            best_perr   = perr
            best_fitted = fitted
            best_n      = n_try

        white_str = "YES ✓" if is_white else "no"
        print(f"  {n_try:>3}  {a:>10.2f}  {delta:>+8.2f}  {rms:>10.5f}  {max_resid:>12.5f}  {white_str:>7}{mark}")
        aic_history.append((n_try, a, rms, max_resid, is_white))
        fit_store[n_try] = (popt, perr, fitted)

        # ── Seed the next component where the fit is worst ──────────────────────
        #
        # This block previously did the opposite of what it should, in two ways:
        #
        #   * `height=off_pulse_rms` accepted 1-sigma residual bumps, i.e. noise.
        #   * seeds were ranked by DISTANCE from existing seeds, largest first.
        #     Every existing seed is in the on-pulse region, so "farthest" means the
        #     middle of the off-pulse baseline — the search was actively driven into
        #     the noise, which is why components landed at bins 58, 117, 174, 255,
        #     301 and 326 on real data.
        #
        # A blended component is a SHOULDER, not a local maximum, so it is invisible
        # to peak detection on the profile itself but shows up clearly as a
        # localised residual excess once the brighter components are fitted. Seeding
        # at the largest on-pulse residual is therefore what finds them.
        _cand, _ = find_peaks(np.abs(resid), height=3.0 * off_pulse_rms, distance=8)
        _cand = [int(p) for p in _cand if on_pulse[p]]
        _used = [int(round(s)) % N_BINS for s in seed_bins]
        for _ns in sorted(_cand, key=lambda p: -abs(resid[p])):   # largest residual first
            if all(abs(int(_wrap_bins(_ns - u, N_BINS))) >= 5 for u in _used):
                seed_bins.append(float(_ns))
                break

        if is_white:
            print(f"\n  ✓ Residuals are white at N={n_try} "
                  f"(max |resid| = {max_resid:.5f} < 3σ = {3*off_pulse_rms:.5f})")
            # Whiteness criterion overrides AIC: always use this model
            if n_try != best_n:
                print(f"  (Overrides AIC-best N={best_n})")
            best_aic    = a
            best_popt   = popt
            best_perr   = perr
            best_fitted = fitted
            best_n      = n_try
            break

    if args.force_n_components is not None:
        n_force = args.force_n_components
        print(f"\n  --force_n_components={n_force}: fitting exactly N={n_force} (AIC ignored) ...")
        popt_f, perr_f, fitted_f = fit_n_components(fscrunch, n_force, seed_bins[:n_force])
        if popt_f is None:
            print(f"  WARNING: forced N={n_force} fit failed; falling back to AIC-best N={best_n}.")
        else:
            best_popt   = popt_f
            best_perr   = perr_f
            best_fitted = fitted_f
            best_n      = n_force
            print(f"  Forced fit succeeded: using N={n_force}.")

    if best_popt is None:
        print("  ERROR: all fits failed.")
        sys.exit(1)

    components = parse_components(best_popt, best_perr, best_n,
                                  sort_by_amp=not args.seed_from_config)
    residuals  = fscrunch - best_fitted
    rms        = residuals.std()

    print(f"\n  {'Config-seeded' if args.seed_from_config else 'AIC/whiteness'} fit: N = {best_n}  (AIC = {best_aic:.2f},  RMS residual = {rms:.5f})")


    # ─────────────────────────────────────────────────────────────────────────────
    # Significance scan — how many components are actually DETECTED?
    #
    # AIC alone is easily gamed on a pulse profile: each extra von Mises term can
    # mop up a single noise bin, so AIC keeps improving while the added components
    # are meaningless. Two signatures give them away:
    #
    #   * amplitude below a few x the off-pulse RMS   -> not a detection
    #   * kappa pinned at the fitter's upper bound    -> narrower than any real
    #                                                    component, i.e. a spike
    #                                                    fitted to one bin
    #
    # This picks the largest N for which EVERY component passes both tests, which
    # is the honest component count.
    # ─────────────────────────────────────────────────────────────────────────────

    def significance_report(popt, n):
        """Per-component significance, kappa-ceiling and blending flags.

    Three ways a fitted component fails to be an independent detection:
      sig     amplitude below a few x the off-pulse RMS -> not a detection
      pinned  kappa at the fitter's ceiling -> a spike fitted to one bin
      blended sitting inside a BRIGHTER component's half-width -> the fit is
              describing ONE feature's shape with two terms, which adds no new
              component. On the German archive this is how both the spurious
              5th (a kappa=456 core on the PreMain shoulder) and the earlier
              spurious 4th (a kappa=441 core on the main peak) appear: both
              clear 8-15 sigma on amplitude, so significance alone misses them.
    """
        out = []
        for i in range(n):
            A  = float(popt[3*i])
            mu = float(popt[3*i+1]) % N_BINS
            k  = float(popt[3*i+2])
            out.append(dict(
                A      = A,
                mu     = mu,
                k      = k,
                sig    = A / off_pulse_rms if off_pulse_rms > 0 else np.inf,
                pinned = k >= 0.99 * KAPPA_CEIL,
                fwhm   = _fwhm_of(k),
            ))
        for i, r in enumerate(out):
            r["blended"], r["blend_with"] = False, None
            for j, o in enumerate(out):
                if i == j or o["A"] <= r["A"]:
                    continue
                if abs(float(_wrap_bins(r["mu"] - o["mu"], N_BINS))) < 0.5 * o["fwhm"]:
                    r["blended"], r["blend_with"] = True, j
                    break
        return out


    MIN_SIG = args.min_significance

    print(f"\n{'─'*60}")
    print(f"  SIGNIFICANCE SCAN  (off-pulse RMS = {off_pulse_rms:.5f}, "
          f"threshold = {MIN_SIG:g}σ)")
    print(f"{'─'*60}")
    print(f"  {'N':>3}  {'min amp sig':>12}  {'κ ceiling':>10}  {'blended':>8}  {'verdict':>12}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*12}")

    sig_n = 0
    for _n in sorted(fit_store):
        _rep     = significance_report(fit_store[_n][0], _n)
        _min_sig = min(r["sig"] for r in _rep)
        _n_pin   = sum(r["pinned"] for r in _rep)
        _n_bl    = sum(r["blended"] for r in _rep)
        _ok      = (_min_sig >= MIN_SIG) and (_n_pin == 0) and (_n_bl == 0)
        if _ok:
            sig_n = _n
        _why = ("all distinct" if _ok else
                ("below %gσ" % MIN_SIG if _min_sig < MIN_SIG else
                 ("κ pinned" if _n_pin else "not distinct")))
        print(f"  {_n:>3}  {_min_sig:>12.1f}  {_n_pin:>10d}  {_n_bl:>8d}  {_why:>12}")

    print(f"\n  Largest N whose components are all ≥{MIN_SIG:g}σ, unpinned in κ, and "
          f"spatially distinct: N = {sig_n if sig_n else '(none)'}")
    print(f"  'Blended' = inside a brighter component's half-width, i.e. a second "
          f"term on one feature rather than a new feature.")

    # Significance overrides AIC for the REPORTED model.
    #
    # AIC and the whiteness test both keep rewarding components that fit single
    # noise bins — on real data AIC preferred N=10 where only 4 components were
    # detections. Reporting a model that contains terms we can demonstrate are
    # noise is worse than reporting a slightly under-fitted one, because those
    # terms end up pasted into a config as though they were profile structure.
    if sig_n and sig_n != best_n and not args.force_n_components:
        print(f"  AIC/whiteness preferred N = {best_n}; the extra "
              f"{best_n - sig_n} component(s) do not clear {MIN_SIG:g}σ.")
        print(f"  -> Reporting N = {sig_n} (significance-selected). "
              f"--force_n_components overrides.")
        best_popt, best_perr, best_fitted = fit_store[sig_n]
        best_n   = sig_n
        best_aic = next(h[1] for h in aic_history if h[0] == sig_n)
        # `components` was parsed from the previous best_popt above, so it has to be
        # rebuilt — everything downstream (the component table, the config
        # comparison, the reference-channel seeding, the plot) reads it.
        components = parse_components(best_popt, best_perr, best_n,
                                      sort_by_amp=not args.seed_from_config)
        residuals  = fscrunch - best_fitted
        rms        = residuals.std()


    # ─────────────────────────────────────────────────────────────────────────────
    # Is the config's component count still a defensible description?
    #
    # "Does this archive still show four components?" AIC alone will usually prefer
    # more, because extra von Mises terms keep mopping up low-level structure. What
    # matters is how much worse the config's N is, and whether its residuals are
    # already at the noise floor.
    # ─────────────────────────────────────────────────────────────────────────────

    # ── Config-seeded cross-check ───────────────────────────────────────────────
    #
    # Always run, and always report alongside the free result. The config only
    # supplies a STARTING POINT; AIC and RMS then say objectively which optimum the
    # data prefers. If the free search and the seeded fit land on the same optimum,
    # the config's decomposition is what the data wants and nothing here is
    # circular. If they disagree, the numbers below show which model is better —
    # the seeded fit is never accepted on the strength of matching the config.
    _seeded = {}
    if cfg is not None and cfg.components:
        _sp, _se, _sf = fit_seeded_from_config(fscrunch, int(np.argmax(fscrunch)))
        if _sp is not None:
            _sn   = len(cfg.components)
            _saic = compute_aic(fscrunch, _sf, 3 * _sn + 1)
            _srms = (fscrunch - _sf).std()
            _srep = significance_report(_sp, _sn)
            _seeded = dict(popt=_sp, perr=_se, fitted=_sf, n=_sn, aic=_saic, rms=_srms,
                           rep=_srep)
            _free_at_n = next((h for h in aic_history if h[0] == _sn), None)

            print(f"\n{'─'*60}")
            print(f"  CONFIG-SEEDED CROSS-CHECK  (N = {_sn}, seeded at the config's "
                  f"own geometry)")
            print(f"{'─'*60}")
            print(f"  {'model':<34} {'AIC':>11} {'RMS':>10}")
            print(f"  {'─'*34} {'─'*11} {'─'*10}")
            print(f"  {'config-seeded, N=%d' % _sn:<34} {_saic:>11.2f} {_srms:>10.5f}")
            if _free_at_n:
                print(f"  {'free search, same N=%d' % _sn:<34} "
                      f"{_free_at_n[1]:>11.2f} {_free_at_n[2]:>10.5f}")
            print(f"  {'free search, selected N=%d' % best_n:<34} "
                  f"{best_aic:>11.2f} {rms:>10.5f}")

            if _free_at_n is not None:
                _d = _saic - _free_at_n[1]
                if abs(_d) < 2:
                    print(f"\n  [OK]   at N={_sn} the free search and the config-seeded "
                          f"fit reach the SAME optimum (ΔAIC = {_d:+.2f}). The config's "
                          f"decomposition is what the data prefers — this conclusion "
                          f"does not depend on having the config.")
                elif _d < 0:
                    print(f"\n  [NOTE] the config-seeded fit is better by ΔAIC = {_d:+.2f}, "
                          f"so the free search is stuck in a worse optimum at this N. "
                          f"The config's geometry is supported by the data, but the free "
                          f"search could not find it unaided — treat its N and phases "
                          f"with suspicion and check the seeding.")
                else:
                    print(f"\n  [WARN] the free search finds a BETTER fit at the same N "
                          f"(ΔAIC = {_d:+.2f} against the seeded fit). The config's "
                          f"component geometry is not what this profile prefers. Do not "
                          f"adopt the seeded numbers; investigate the free solution.")

    # --seed_from_config selects the seeded fit for the detailed report, plot and
    # --emit-yaml. The free result above is printed either way, so the reader can
    # always see what the data says without the config's help.
    if args.seed_from_config:
        if not _seeded:
            print("\nERROR: --seed_from_config given but the config-seeded fit failed.")
            sys.exit(1)
        print(f"\n  --seed_from_config: reporting the config-seeded N={_seeded['n']} fit "
              f"below (the free result is above for comparison).")
        best_popt, best_perr, best_fitted = (_seeded["popt"], _seeded["perr"],
                                             _seeded["fitted"])
        best_n, best_aic = _seeded["n"], _seeded["aic"]
        components = parse_components(best_popt, best_perr, best_n, sort_by_amp=False)
        residuals  = fscrunch - best_fitted
        rms        = residuals.std()

    if cfg is not None:
        _n_cfg = len(cfg.components)
        _hit   = next((h for h in aic_history if h[0] == _n_cfg), None)

        print(f"\n{'─'*60}")
        print(f"  CONFIG COMPONENT COUNT CHECK  (config declares N = {_n_cfg})")
        print(f"{'─'*60}")
        if _hit is None:
            print(f"  N = {_n_cfg} was not reached (fits stopped at N = {len(aic_history)}). "
                  f"Raise --max_components.")
        else:
            _n_h, _aic_h, _rms_h, _maxr_h, _white_h = _hit
            print(f"  N = {_n_cfg}:  AIC = {_aic_h:.2f}   RMS resid = {_rms_h:.5f}   "
                  f"max|resid| = {_maxr_h:.5f}   white = {'YES' if _white_h else 'no'}")
            print(f"  AIC-preferred N = {best_n} (ΔAIC = {_aic_h - best_aic:+.2f} for the "
                  f"config's N)")
            print(f"  off-pulse 3σ = {3*off_pulse_rms:.5f}")
            if _white_h:
                print(f"  [OK]   N = {_n_cfg} already leaves white residuals — the extra "
                      f"components AIC prefers are fitting structure at or below the noise. "
                      f"Four components remain an adequate description.")
            elif _maxr_h < 6 * off_pulse_rms:
                print(f"  [WARN] N = {_n_cfg} leaves residuals at "
                      f"{_maxr_h/off_pulse_rms:.1f}σ (not white, but within 6σ). Defensible "
                      f"for the simulator, though real structure is being absorbed into the "
                      f"four components. Compare the fitted kappas against the config's "
                      f"nuisance priors.")
            else:
                print(f"  [FAIL] N = {_n_cfg} leaves {_maxr_h/off_pulse_rms:.1f}σ residuals. "
                      f"This profile is not well described by {_n_cfg} components — adding "
                      f"components to the config is a real change to the forward model and "
                      f"invalidates existing posteriors (the theta vector gains parameters).")
            print(f"\n  Re-run with --force_n_components {_n_cfg} to fit exactly the config's "
                  f"count and compare phase centres directly.")


    # ─────────────────────────────────────────────────────────────────────────────
    # Fit the reference-frequency channel with the same N (fixed seeds from above)
    # ─────────────────────────────────────────────────────────────────────────────

    if args.seed_from_config:
        # Same seeding as the fscrunch fit, anchored on the reference component
        # as IT was fitted, so comps_rc[i] is also config component i.
        _rb = int(round(components[[c.name for c in cfg.components].index(
            cfg.reference_component)]["mu"]))
        popt_rc, perr_rc, fitted_rc = fit_seeded_from_config(ref_chan, _rb)
    else:
        seed_from_fscrunch = [c["mu"] for c in components]
        popt_rc, perr_rc, fitted_rc = fit_n_components(ref_chan, best_n,
                                                       seed_from_fscrunch)
    comps_rc = None
    if popt_rc is not None:
        comps_rc = parse_components(popt_rc, perr_rc, best_n,
                                    sort_by_amp=not args.seed_from_config)


    # ─────────────────────────────────────────────────────────────────────────────
    # Undo the centring roll — everything from here on is in the ORIGINAL phase
    # frame of the input archive, so reported phase centres are directly
    # comparable with the config's.
    # ─────────────────────────────────────────────────────────────────────────────

    if roll % N_BINS:
        for _c in components:
            _c["mu"] = (_c["mu"] - roll) % N_BINS
        if comps_rc is not None:
            for _c in comps_rc:
                _c["mu"] = (_c["mu"] - roll) % N_BINS
        best_fitted = np.roll(best_fitted, -roll)
        if fitted_rc is not None:
            fitted_rc = np.roll(fitted_rc, -roll)
        fscrunch  = _fscrunch_orig
        ref_chan  = _ref_chan_orig
        residuals = fscrunch - best_fitted
        rms       = residuals.std()


    # ─────────────────────────────────────────────────────────────────────────────
    # Console report
    # ─────────────────────────────────────────────────────────────────────────────

    b = np.arange(N_BINS, dtype=float)

    print(f"\n{'═'*60}")
    print(f"  COMPONENT PARAMETERS  (fscrunched profile, normalised to main=1)")
    print(f"{'═'*60}")
    header = (f"  {'#':>2}  {'Bin':>7}  {'Phase [0,1]':>12}  {'amp_rel':>8}  {'kappa':>7}  "
              f"{'FWHM':>6}  {'σ_mu':>5}  {'amp σ':>7}  flags")
    print(header)
    print(f"  {'─'*2}  {'─'*7}  {'─'*12}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*5}")

    for i, c in enumerate(components):
        sig   = c["A"] / off_pulse_rms if off_pulse_rms > 0 else np.inf
        fwhm  = _fwhm_of(c["k"])
        flags = []
        if sig < MIN_SIG:
            flags.append(f"NOISE(<{MIN_SIG:g}σ)")
        if c["k"] >= 0.99 * KAPPA_CEIL:
            flags.append("κ-PINNED")
        # A component sitting well inside another's width is not an independent
        # feature — it is the fit describing one peak's shape with two terms.
        for j, o in enumerate(components):
            if j == i:
                continue
            sep = abs(float(_wrap_bins(c["mu"] - o["mu"], N_BINS)))
            if o["A"] > c["A"] and sep < 0.5 * _fwhm_of(o["k"]):
                flags.append(f"BLENDED w/ #{j} ({sep:.1f} bins)")
                break
        print(f"  {i:>2}  {c['mu']:>7.2f}  {c['mu']/N_BINS:>12.8f}"
              f"  {c['amp_rel']:>8.4f}  {c['k']:>7.1f}  {fwhm:>6.1f}  {c['dmu']:>5.2f}"
              f"  {sig:>7.1f}  {' '.join(flags)}")

    _n_real = sum(1 for c in components
                  if c["A"] / max(off_pulse_rms, 1e-30) >= MIN_SIG
                  and c["k"] < 0.99 * KAPPA_CEIL)
    print(f"\n  {_n_real} of {len(components)} components clear {MIN_SIG:g}σ "
          f"with an unpinned κ.")

    if comps_rc is not None:
        print(f"\n{'─'*60}")
        print(f"  At reference frequency (~{REF_FREQ_MHZ:.0f} MHz, channel {ref_idx})")
        print(f"{'─'*60}")
        for i, c in enumerate(comps_rc):
            label = " ← main" if i == 0 else f" ← comp {i}"
            print(f"  {i:>2}  {c['mu']:>7.2f}  {c['mu']/N_BINS:>12.8f}"
                  f"  {c['amp_rel']:>8.4f}  {c['k']:>7.1f}{label}")

    print(f"\n  Baseline: {best_popt[-1]:+.5f}")
    print(f"\n{'═'*60}")
    print(f"  SIMULATOR PARAMETERS")
    print(f"{'═'*60}")
    main = components[0]
    print(f"  Main:  phase_center_ref = {main['mu']/N_BINS:.8f}   (bin {main['mu']:.1f})")
    for i, c in enumerate(components[1:], 1):
        print(f"  Comp{i}: phase_center_ref = {c['mu']/N_BINS:.8f}   (bin {c['mu']:.1f})")
        print(f"         amplitude_ref    = {c['amp_rel']:.4f}   (fscrunched ratio)")
        if comps_rc is not None and i < len(comps_rc):
            print(f"         amplitude_ref    = {comps_rc[i]['amp_rel']:.4f}   (at ~{REF_FREQ_MHZ:.0f} MHz channel)")
    print()


    # ─────────────────────────────────────────────────────────────────────────────
    # Phase-offset report and --emit-yaml
    #
    # Only the DIFFERENCES between phase centres are physical. Every component in
    # the config takes phase_offset_param: toa, so `toa` shifts them all together
    # and a constant added to every centre is exactly degenerate with it.
    #
    # The constant is not entirely free, though: `toa` has a bounded prior. If the
    # fitted profile sits further from the config's centres than that prior allows,
    # the true toa is unreachable, the posterior piles up on the boundary, and DM is
    # biased because ToA and DM covary. That is what this section checks.
    # ─────────────────────────────────────────────────────────────────────────────

    if cfg is None:
        print("\n  No --config: the comparison with a config's components is skipped.")
    else:
        _cfg_comps = cfg.components
        _toa_spec  = next((p for p in list(cfg.inferred) + list(cfg.nuisance)
                           if p.name == "toa"), None)

        print(f"{'═'*60}")
        print(f"  PHASE OFFSET vs CONFIG  ({os.path.basename(cfg.path)})")
        print(f"{'═'*60}")

        # PHASES come from the frequency-scrunched fit; WIDTHS and AMPLITUDES from the
        # reference window. They are different questions and want different estimators.
        #
        # Every component has spectral_index_center = 0.0, so a component's phase does
        # NOT depend on frequency: there is nothing for a reference-frequency fit to
        # measure that the band average does not measure better. Residual dedispersion
        # does shift the band-averaged phase, but it shifts every component by the same
        # DM, so the DIFFERENCES between phase centres — the only physical part, since a
        # constant is degenerate with toa — survive. The fscrunched profile averages all
        # n_chan channels, so it has sqrt(n_chan/ref_window) times the S/N.
        #
        # An earlier version compared phases against the reference-window fit on the
        # grounds that the config's values are "defined at reference_freq_mhz". That is
        # true of amplitude_ref and concentration_ref, which really are frequency
        # dependent — but not of phase_center_ref, and the noise cost was severe: on
        # DE602 the 5-channel fit fell into the main-peak-splitting degeneracy and
        # reported a 71-sigma phase disagreement that its own fscrunched fit did not
        # show.
        _fit_comps = components
        print(f"  Phases from the FREQUENCY-SCRUNCHED fit (all {cfg.n_chan} channels): "
              f"spectral_index_center is 0, so phase centres carry no frequency "
              f"dependence, and this has the best S/N.")
        if comps_rc is not None:
            _rf = (archive_freqs if archive_freqs is not None else cfg.freqs_mhz)
            print(f"  Widths/amplitudes from the REFERENCE-WINDOW fit (channels "
                  f"{ref_lo}–{ref_hi-1}, {_rf[ref_lo]:.3f}–{_rf[ref_hi-1]:.3f} MHz), "
                  f"which is where the config defines them.")

        def _wrap(x):
            """Wrap a phase difference into (-0.5, 0.5]."""
            return (np.asarray(x) + 0.5) % 1.0 - 0.5


        if not _fit_comps or not _cfg_comps:
            print("  Nothing to compare (no fitted or no configured components).")
            _pairs, _matched, mean_off, resid = [], [], 0.0, np.array([])
            anchor_off, _n_cmp = 0.0, 0
        else:
            # Pairing. `components` is sorted by amplitude; the config is in declaration
            # order. Matching by position is therefore wrong in general.
            #
            # Anchor instead on the config's reference_component, which is by definition
            # the brightest — so it corresponds to fitted component 0. That fixes the
            # constant offset, after which every other config component is matched to
            # its nearest fitted component in phase.
            _ref_i = [c.name for c in _cfg_comps].index(cfg.reference_component)

            if args.seed_from_config:
                # Component i IS config component i by construction, so no phase
                # matching is needed — and none should be attempted, since matching is
                # free to re-order and would hide a component that genuinely moved.
                anchor_off = float(_wrap(_fit_comps[_ref_i]["mu"] / N_BINS
                                         - float(_cfg_comps[_ref_i].phase_center_ref)))
                _pairs = [(cc, _fit_comps[i],
                           float(_wrap(_fit_comps[i]["mu"] / N_BINS
                                       - float(cc.phase_center_ref))))
                          for i, cc in enumerate(_cfg_comps)]
                print(f"  Pairing is by construction (--seed_from_config): fitted "
                      f"component i is config component i.")
            else:
                anchor_off = float(_wrap(_fit_comps[0]["mu"] / N_BINS
                                         - float(_cfg_comps[_ref_i].phase_center_ref)))

                _used, _pairs = set(), []
                for ci, cc in enumerate(_cfg_comps):
                    target = float(cc.phase_center_ref) + anchor_off
                    best, best_d = None, None
                    for fi, fc in enumerate(_fit_comps):
                        if fi in _used:
                            continue
                        d = abs(float(_wrap(fc["mu"] / N_BINS - target)))
                        if best_d is None or d < best_d:
                            best, best_d = fi, d
                    if best is None:
                        _pairs.append((cc, None, np.nan))
                    else:
                        _used.add(best)
                        off = float(_wrap(_fit_comps[best]["mu"] / N_BINS
                                          - float(cc.phase_center_ref)))
                        _pairs.append((cc, _fit_comps[best], off))

                print(f"  Pairing anchored on reference_component "
                      f"'{cfg.reference_component}' (assumed brightest); others matched "
                      f"by nearest phase. --seed_from_config removes this guesswork.")
            if len(_fit_comps) != len(_cfg_comps):
                print(f"  [WARN] fit found {len(_fit_comps)} components, config declares "
                      f"{len(_cfg_comps)}. Check the pairing below before trusting it.")
            print()
            print(f"  {'component':<10} {'config phase':>13} {'fitted phase':>13} "
                  f"{'offset':>10} {'Δ vs anchor':>12}")
            print(f"  {'─'*10} {'─'*13} {'─'*13} {'─'*10} {'─'*12}")
            for cc, fc, off in _pairs:
                if fc is None:
                    print(f"  {cc.name:<10} {float(cc.phase_center_ref):>13.8f} "
                          f"{'—':>13} {'—':>10} {'unmatched':>12}")
                    continue
                print(f"  {cc.name:<10} {float(cc.phase_center_ref):>13.8f} "
                      f"{fc['mu']/N_BINS:>13.8f} {off:>+10.6f} {off-anchor_off:>+12.6f}")

            _matched = [(cc, fc, off) for cc, fc, off in _pairs if fc is not None]
            offsets  = np.array([off for _, _, off in _matched])
            mean_off = anchor_off          # the anchor defines the constant, not the mean
            resid    = offsets - mean_off
            _n_cmp   = len(_matched)

            print(f"\n  constant offset (from anchor) : {mean_off:+.6f} phase "
                  f"= {mean_off * cfg.period_ms * 1000:+.2f} μs")
            print(f"  relative-phase residual (after removing the constant):")
            print(f"    max|resid| = {np.abs(resid).max():.6f} phase "
                  f"= {np.abs(resid).max() * cfg.period_ms * 1000:.2f} μs "
                  f"= {np.abs(resid).max() * N_BINS:.2f} bins")

            if _toa_spec is not None:
                lim = min(abs(_toa_spec.low), abs(_toa_spec.high))
                print(f"\n  toa prior: U[{_toa_spec.low}, {_toa_spec.high}] phase "
                      f"(±{lim * cfg.period_ms * 1000:.1f} μs)")
                if abs(mean_off) < 0.5 * lim:
                    print(f"  [OK]   |mean offset| = {abs(mean_off):.4f} < half the prior "
                          f"half-width ({0.5*lim:.4f}). The constant offset is absorbed by "
                          f"toa with margin — the config's phase centres are usable as they are.")
                elif abs(mean_off) < lim:
                    print(f"  [WARN] |mean offset| = {abs(mean_off):.4f} is inside the prior "
                          f"(±{lim}) but uses {100*abs(mean_off)/lim:.0f}% of it. The true toa "
                          f"will sit near the boundary, which biases DM through the ToA–DM "
                          f"covariance. Re-centre with --recentre-on {_cfg_comps[0].name}.")
                else:
                    print(f"  [FAIL] |mean offset| = {abs(mean_off):.4f} EXCEEDS the toa prior "
                          f"(±{lim}). The true toa is unreachable; the posterior will pile up "
                          f"on the boundary and DM will be biased. Re-centre the phase centres "
                          f"with --recentre-on {_cfg_comps[0].name} — do NOT simply widen the "
                          f"toa prior, which costs ToA precision and risks phase wrap.")

            # Is the SHAPE different, i.e. beyond the fit's own uncertainty?
            _dmu_phase = np.array([fc["dmu"] / N_BINS for _, fc, _ in _matched])
            if np.all(_dmu_phase > 0):
                _worst = np.abs(resid) / _dmu_phase
                print(f"\n  relative residual in units of the fit's own σ_mu: "
                      f"max {_worst.max():.1f}σ")
                if _worst.max() < 3:
                    print(f"  [OK]   relative phases agree with the config within 3σ — the "
                          f"profile shape carries over, only the constant offset differs.")
                else:
                    print(f"  [NOTE] relative phases differ by >3σ. That is a genuinely "
                          f"different profile geometry, not a folding convention, so the "
                          f"config's phase centres need updating (not just re-centring).")

            # ── Do the fitted values sit inside the config's PRIORS? ─────────────────
            #
            # A single-epoch fit cannot set a prior — it measures one realisation of a
            # parameter that is sampled over a range. But it can show that a prior is
            # in the wrong place: a fitted value outside its bounds, or hugging an edge,
            # means the sampler can never reproduce this observation and the posterior
            # will pile up against that boundary.
            print(f"\n{'─'*60}")
            print(f"  FITTED VALUES vs CONFIG PRIORS  (reference channel)")
            print(f"{'─'*60}")
            print(f"  {'component':<10} {'field':<14} {'fitted':>9} {'prior':>16} "
                  f"{'% up':>6}  status")
            print(f"  {'─'*10} {'─'*14} {'─'*9} {'─'*16} {'─'*6}  {'─'*6}")

            _prior_flags = []
            for cc, fc, _o in _matched:
                # Widths and amplitudes ARE frequency dependent, so take them from the
                # reference-window fit — matched to this component by phase, since that
                # fit is run separately and its component order need not agree.
                _rc = fc
                if comps_rc is not None:
                    _rc = min(comps_rc,
                              key=lambda q: abs(float(_wrap(q["mu"]/N_BINS - fc["mu"]/N_BINS))))
                    if abs(float(_wrap(_rc["mu"]/N_BINS - fc["mu"]/N_BINS))) * N_BINS > 15:
                        _rc = fc      # no credible counterpart; fall back, flagged below
                        print(f"  [NOTE] {cc.name}: no reference-window counterpart within "
                              f"15 bins; width/amplitude below are band-averaged.")
                for _field, _val in (("concentration_ref", _rc["k"]),
                                     ("amplitude_ref",     _rc["amp_rel"])):
                    _spec = getattr(cc, _field)
                    if not (isinstance(_spec, str) and _spec.startswith("$")):
                        continue   # a literal, not a sampled parameter — nothing to check
                    _p = next(p for p in list(cfg.inferred) + list(cfg.nuisance)
                              if p.name == _spec[1:])
                    _pct = (_val - _p.low) / (_p.high - _p.low) * 100.0
                    if _val < _p.low or _val > _p.high:
                        _st = "OUTSIDE PRIOR"
                        _prior_flags.append((cc.name, _p.name, _val, _p.low, _p.high, "outside"))
                    elif _pct < 5 or _pct > 95:
                        _st = "at edge"
                        _prior_flags.append((cc.name, _p.name, _val, _p.low, _p.high, "edge"))
                    else:
                        _st = "ok"
                    print(f"  {cc.name:<10} {_field:<14} {_val:>9.3f} "
                          f"{'U[%g, %g]' % (_p.low, _p.high):>16} {_pct:>6.1f}  {_st}")

            if _prior_flags:
                print()
                for _cn, _pn, _v, _lo, _hi, _kind in _prior_flags:
                    if _kind == "outside":
                        print(f"  [FAIL] {_cn}: fitted {_v:.3f} is OUTSIDE {_pn} "
                              f"U[{_lo:g}, {_hi:g}]. The simulator can never draw this "
                              f"value, so the posterior will jam against the bound and "
                              f"bias everything it covaries with. Widen {_pn}.")
                    else:
                        print(f"  [WARN] {_cn}: fitted {_v:.3f} hugs an edge of {_pn} "
                              f"U[{_lo:g}, {_hi:g}]. Little room on one side — check "
                              f"more epochs before trusting this prior.")
            else:
                print(f"\n  [OK]   every fitted value sits comfortably inside its prior.")

            # ── The actual question: are the COMPONENT SEPARATIONS preserved? ────────
            # Separations are measured from the reference component, so they are
            # independent of the constant offset / folding convention entirely.
            _ref_pair = next(((cc, fc, o) for cc, fc, o in _matched
                              if cc.name == cfg.reference_component), None)
            if _ref_pair is not None and len(_matched) > 1:
                _rc_cfg = float(_ref_pair[0].phase_center_ref)
                _rc_fit = _ref_pair[1]["mu"] / N_BINS
                print(f"\n{'─'*60}")
                print(f"  COMPONENT SEPARATIONS from '{cfg.reference_component}'")
                print(f"  (independent of the constant offset — this is the profile geometry)")
                print(f"{'─'*60}")
                print(f"  {'component':<10} {'config':>11} {'fitted':>11} "
                      f"{'Δ (phase)':>11} {'Δ (bins)':>9} {'Δ (μs)':>9} {'σ_sep':>7} {'Δ/σ':>6}")
                print(f"  {'─'*10} {'─'*11} {'─'*11} {'─'*11} {'─'*9} {'─'*9} "
                      f"{'─'*7} {'─'*6}")
                _worst_sep  = 0.0
                _worst_nsig = 0.0
                _worst_frac = 0.0     # |Δ| as a fraction of the component's OWN width
                _ref_dmu    = _ref_pair[1]["dmu"]
                for cc, fc, _o in _matched:
                    if cc.name == cfg.reference_component:
                        continue
                    s_cfg = float(_wrap(float(cc.phase_center_ref) - _rc_cfg))
                    s_fit = float(_wrap(fc["mu"] / N_BINS - _rc_fit))
                    d     = s_fit - s_cfg
                    # A separation is a difference of two fitted centres, so its
                    # uncertainty is theirs in quadrature. Judging Δ in bins alone
                    # punishes a weak, broad component for being hard to localise.
                    sig   = float(np.hypot(fc["dmu"], _ref_dmu))
                    nsig  = abs(d) * N_BINS / sig if sig > 0 else np.inf
                    _worst_sep  = max(_worst_sep, abs(d))
                    _worst_nsig = max(_worst_nsig, nsig)
                    _worst_frac = max(_worst_frac,
                                      abs(d) * N_BINS / max(_fwhm_of(fc["k"]), 1e-9))
                    print(f"  {cc.name:<10} {s_cfg:>+11.6f} {s_fit:>+11.6f} "
                          f"{d:>+11.6f} {d*N_BINS:>+9.2f} {d*cfg.period_ms*1000:>+9.1f} "
                          f"{sig:>7.2f} {nsig:>6.1f}")
                print(f"\n  worst separation change: {_worst_sep:.6f} phase "
                      f"= {_worst_sep*N_BINS:.2f} bins = {_worst_sep*cfg.period_ms*1000:.1f} μs"
                      f"  ({_worst_nsig:.1f}σ)")
                print(f"  σ_sep = the two fitted centres' σ_mu in quadrature, in bins.")

                if len(_fit_comps) != len(_cfg_comps):
                    # With more fitted components than the config declares, the extra
                    # ones are usually noise spikes or a second term describing one
                    # peak's shape. Nearest-phase matching then pairs a config component
                    # against the wrong thing and the "separation change" is an artefact
                    # of the pairing, not a measurement. Do not render a verdict.
                    print(f"  [SKIP] verdict withheld: the fit has {len(_fit_comps)} "
                          f"components against the config's {len(_cfg_comps)}, so this "
                          f"pairing is unreliable.")
                    print(f"         Re-run with --force_n_components {len(_cfg_comps)} "
                          f"for a like-for-like comparison.")
                elif _worst_nsig < 3.0:
                    print(f"  [OK]   every separation agrees with the config within "
                          f"{_worst_nsig:.1f}σ of the fit's own precision. The profile "
                          f"geometry carries over: re-centre the whole set on "
                          f"'{cfg.reference_component}' and the config's component "
                          f"structure is reusable.")
                elif _worst_frac < 0.3:
                    print(f"  [WARN] separations differ by up to {_worst_nsig:.1f}σ "
                          f"({_worst_sep*N_BINS:.1f} bins) — formally significant, but only "
                          f"{100*_worst_frac:.0f}% of the shifted component's own width, so "
                          f"this is recognisably the same decomposition, measured more "
                          f"precisely. Paste the fitted centres in rather than reusing the "
                          f"old values; the component structure itself carries over.")
                else:
                    print(f"  [FAIL] separations differ by up to {_worst_nsig:.1f}σ "
                          f"({_worst_sep*N_BINS:.1f} bins = {100*_worst_frac:.0f}% of the "
                          f"component's own width). Either the fit found a different solution "
                          f"(check the pairing and component count above) or the profile "
                          f"genuinely differs — do not reuse the config's phase centres.")
                print(f"  worst shift as a fraction of its own component width: "
                      f"{_worst_frac:.2f}")
        print()

        if args.emit_yaml:
            if not _matched:
                print("Nothing to emit — no config component could be paired with a fitted one.")
                sys.exit(1)

            shift = 0.0
            if args.recentre_on:
                # Use the SAME phase-matched pairing as the table above, not positional
                # order: `components` is sorted by amplitude while the config is in
                # declaration order, so indexing them together is wrong in general.
                _hit = next(((cc, fc, off) for cc, fc, off in _matched
                             if cc.name == args.recentre_on), None)
                if _hit is None:
                    _names   = [c.name for c in _cfg_comps]
                    _matched_names = [cc.name for cc, _, _ in _matched]
                    if args.recentre_on not in _names:
                        print(f"ERROR: --recentre-on '{args.recentre_on}' is not a component "
                              f"in {os.path.basename(cfg.path)}. Choose from: "
                              f"{', '.join(_names)}")
                    else:
                        print(f"ERROR: config component '{args.recentre_on}' was not matched "
                              f"to any fitted component. Matched: {', '.join(_matched_names)}")
                    sys.exit(1)
                shift = _hit[2]

            print(f"{'═'*60}")
            print(f"  components: BLOCK FOR {os.path.basename(cfg.path)}")
            print(f"{'═'*60}")
            print(f"# Fitted from: {os.path.basename(args.infile)}")
            print(f"# amplitude_ref / concentration_ref below are defined AT "
                  f"reference_freq_mhz = {REF_FREQ_MHZ:.11f}")
            print(f"# (channel {ref_idx}). Do not paste these into a config with a different"
                  f"\n# reference frequency without re-expressing them via the spectral indices.")
            if args.recentre_on:
                print(f"# Re-centred on '{args.recentre_on}': {shift:+.8f} phase subtracted from")
                print(f"# every centre, so relative phases are preserved and toa stays near 0.")
            print("components:")
            for cc, fc, _off in _matched:
                phase = (fc["mu"] / N_BINS - shift) % 1.0
                # Keep the config's existing $refs for the free fields — the fit measures
                # this epoch's realisation of them, not their priors.
                print(f"  - name:                  {cc.name}")
                print(f"    amplitude_ref:         {cc.amplitude_ref}"
                      f"        # fitted this epoch: {fc['amp_rel']:.4f}")
                print(f"    phase_center_ref:      {phase:.8f}")
                print(f"    concentration_ref:     {cc.concentration_ref}"
                      f"        # fitted this epoch: {fc['k']:.1f}")
                print(f"    spectral_index_amp:    {cc.spectral_index_amp}")
                print(f"    spectral_index_width:  {cc.spectral_index_width}")
                print(f"    spectral_index_center: {cc.spectral_index_center}")
                print(f"    sigma_broadening_bins: {cc.sigma_broadening_bins}")
                print()
            print("# NOTE: amplitude_ref / concentration_ref keep the config's existing")
            print("#       literals and $refs. A single-epoch fit measures ONE realisation of")
            print("#       parameters that are sampled over a prior, so it cannot set them —")
            print("#       use the 'fitted this epoch' comments to sanity-check that the priors")
            print("#       actually bracket the observed values.")
            print()

            if archive_freqs is not None:
                _d = np.diff(archive_freqs)
                if _d.size and _d.std() > 0.01 * _d.mean():
                    _bw = _d.mean()
                    print(f"{'═'*60}")
                    print(f"  instrument: BLOCK — channels are unevenly spaced")
                    print(f"{'═'*60}")
                    print(f"  n_freq_channels:    {archive_freqs.size}")
                    print(f"  n_phase_bins:       {N_BINS}")
                    print(f"  freq_low_mhz:       {archive_freqs[0] - _bw/2:.6f}"
                          f"   # first centre - half mean channel width")
                    print(f"  freq_high_mhz:      {archive_freqs[-1] + _bw/2:.6f}"
                          f"   # last centre + half mean channel width")
                    print(f"  reference_freq_mhz: {REF_FREQ_MHZ:.11f}   # unchanged from this config")
                    print(f"  channel_freqs_mhz:")
                    for v in archive_freqs:
                        print(f"    - {v:.9f}")
                    print()


    # ─────────────────────────────────────────────────────────────────────────────
    # Weighted σ_toa from fit uncertainties (Fisher information)
    # ─────────────────────────────────────────────────────────────────────────────

    PULSAR_PERIOD_S = cfg.period_s if cfg else SC["period_s"]

    fisher_sum = sum(c["amp_rel"]**2 * c["k"] / c["dmu"]**2
                     for c in components if c["dmu"] > 0)

    print(f"\n{'═'*60}")
    print(f"  EXPECTED TOA UNCERTAINTY FROM FIT (σ_toa)")
    print(f"{'═'*60}")

    if fisher_sum > 0:
        sigma_toa_bins  = 1.0 / np.sqrt(fisher_sum)
        sigma_toa_phase = sigma_toa_bins / N_BINS
        sigma_toa_us    = sigma_toa_phase * PULSAR_PERIOD_S * 1e6

        print(f"  σ_toa = {sigma_toa_bins:.4f} bins  =  {sigma_toa_phase:.7f} phase  =  {sigma_toa_us:.3f} μs")
        print()
        print(f"  Per-component contributions (Fisher weight = A²·κ/σ_μ²):")
        print(f"  {'#':>2}  {'amp_rel':>8}  {'kappa':>7}  {'σ_mu [bins]':>12}  {'Fisher wt':>10}  {'frac':>6}")
        print(f"  {'─'*2}  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*10}  {'─'*6}")
        for i, c in enumerate(components):
            if c["dmu"] > 0:
                fw   = c["amp_rel"]**2 * c["k"] / c["dmu"]**2
                frac = fw / fisher_sum * 100.0
                print(f"  {i:>2}  {c['amp_rel']:>8.4f}  {c['k']:>7.1f}  {c['dmu']:>12.4f}"
                      f"  {fw:>10.2f}  {frac:>5.1f}%")
            else:
                print(f"  {i:>2}  {c['amp_rel']:>8.4f}  {c['k']:>7.1f}  {'N/A':>12}  {'—':>10}  {'—':>6}")
        print()
        print(f"  NOTE: Total toa uncertainty on real data = SBI posterior width ⊕ σ_toa(fit)")
        print(f"        Add in quadrature: σ_total = sqrt(σ_SBI² + σ_fit²)")
    else:
        print("  WARNING: could not compute σ_toa (all component position errors are zero)")

    print(f"{'═'*60}")
    print()


    # ─────────────────────────────────────────────────────────────────────────────
    # Plot
    # ─────────────────────────────────────────────────────────────────────────────

    COLORS = ["steelblue", "darkorange", "green", "crimson", "purple",
              "brown", "teal", "olive", "navy", "darkred"]

    n_zoom = min(best_n, 5)
    fig    = plt.figure(figsize=(16, 12))

    import matplotlib.gridspec as gridspec
    # Row 0: full profile, Row 1: residuals, Row 2: AIC curve + zooms
    n_cols = max(n_zoom, 2)
    gs = gridspec.GridSpec(3, n_cols,
                           height_ratios=[3, 1, 2.5],
                           hspace=0.55, wspace=0.35)

    # ── Full profile ──────────────────────────────────────────────────────────────
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(b, fscrunch, color="k", lw=1.2, label="Data (fscrunched, normalised)")
    ax_main.plot(b, best_fitted, "r--", lw=1.8, label=f"Total fit  ({best_n} components)")

    for i, c in enumerate(components):
        col = COLORS[i % len(COLORS)]
        y   = vm_single(b, c["A"], c["mu"], c["k"])
        ax_main.fill_between(b, y, alpha=0.18, color=col)
        ax_main.plot(b, y, color=col, lw=1.3,
                     label=f"Comp {i}: μ={c['mu']/N_BINS:.5f}  A={c['amp_rel']:.3f}  κ={c['k']:.0f}")
        ax_main.axvline(c["mu"], color=col, lw=0.8, ls=":", alpha=0.8)
        ax_main.text(c["mu"], 1.02, f"bin {c['mu']:.1f}", ha="center",
                     fontsize=7, color=col, rotation=45)

    ax_main.set_xlim(0, N_BINS)
    ax_main.set_ylim(-0.05, 1.18)
    ax_main.set_xlabel("Phase bin")
    ax_main.set_ylabel("Normalised amplitude")
    ax_main.set_title("Frequency-scrunched profile — von Mises fit", fontsize=10)
    ax_main.legend(fontsize=8, loc="upper left", ncol=2)

    # ── Residuals ─────────────────────────────────────────────────────────────────
    ax_res = fig.add_subplot(gs[1, :])
    ax_res.plot(b, residuals, color="dimgray", lw=0.9)
    ax_res.axhline(0, color="k", lw=0.8)
    ax_res.fill_between(b, rms, -rms, alpha=0.15, color="gray", label=f"±1 RMS ({rms:.4f})")
    ax_res.set_xlim(0, N_BINS)
    ax_res.set_xlabel("Phase bin")
    ax_res.set_ylabel("Residual")
    ax_res.set_title(f"Residuals   RMS = {rms:.5f}", fontsize=9)
    ax_res.legend(fontsize=7, loc="upper right")

    # ── AIC curve ────────────────────────────────────────────────────────────────
    ax_aic = fig.add_subplot(gs[2, 0])
    ns_plot   = [h[0] for h in aic_history]
    aic_plot  = [h[1] for h in aic_history]
    rms_plot  = [h[2] for h in aic_history]
    white_ns  = [h[0] for h in aic_history if h[4]]

    ax_aic.plot(ns_plot, aic_plot, "o-", color="steelblue", lw=1.5, ms=5, label="AIC")
    ax_aic.axvline(best_n, color="crimson", lw=1.2, ls="--", label=f"Best N={best_n}")
    if white_ns:
        ax_aic.axvline(white_ns[0], color="green", lw=1.2, ls=":", label=f"White N={white_ns[0]}")
    ax_aic.set_xlabel("N components")
    ax_aic.set_ylabel("AIC")
    ax_aic.set_title("AIC vs N", fontsize=9)
    ax_aic.legend(fontsize=7)
    ax_aic.set_xticks(ns_plot)

    ax_rms = ax_aic.twinx()
    ax_rms.plot(ns_plot, rms_plot, "s--", color="darkorange", lw=1.0, ms=4, alpha=0.7, label="RMS")
    ax_rms.axhline(3 * off_pulse_rms, color="green", lw=0.8, ls=":", alpha=0.6, label="3σ noise")
    ax_rms.set_ylabel("Residual RMS", color="darkorange", fontsize=8)
    ax_rms.tick_params(axis="y", labelcolor="darkorange")
    ax_rms.legend(fontsize=6, loc="upper right")

    # ── Per-component zooms ───────────────────────────────────────────────────────
    for zi, c in enumerate(components[:n_zoom - 1]):
        ax = fig.add_subplot(gs[2, zi + 1])
        hw  = 80
        lo  = int(max(0, c["mu"] - hw))
        hi  = int(min(N_BINS, c["mu"] + hw))
        bz  = b[lo:hi]
        col = COLORS[zi % len(COLORS)]

        ax.plot(bz, fscrunch[lo:hi],    color="k",   lw=1.2, label="Data")
        ax.plot(bz, best_fitted[lo:hi], color="r",   lw=1.5, ls="--", label="Total fit")
        y = vm_single(bz, c["A"], c["mu"], c["k"])
        ax.fill_between(bz, y, alpha=0.25, color=col)
        ax.plot(bz, y, color=col, lw=1.3,
                label=f"μ={c['mu']:.1f}  κ={c['k']:.0f}")
        ax.axvline(c["mu"], color=col, lw=0.8, ls=":", alpha=0.9)

        title_tag = "Main" if zi == 0 else f"Comp {zi}"
        ax.set_title(f"{title_tag} — amp_ref={c['amp_rel']:.3f}  "
                     f"phase={c['mu']/N_BINS:.5f}", fontsize=9)
        ax.set_xlabel("Phase bin")
        if zi == 0:
            ax.set_ylabel("Normalised amplitude")
        ax.legend(fontsize=7)

        # Also overlay ref-channel fit if available
        if comps_rc is not None:
            # find matching component (closest mu)
            best_match = min(comps_rc, key=lambda x: abs(x["mu"] - c["mu"]))
            y_rc = vm_single(bz, best_match["A"] / comps_rc[0]["A"] * c["A"],
                              best_match["mu"], best_match["k"])
            ax.plot(bz, y_rc * c["amp_rel"] / best_match["amp_rel"],
                    color=col, lw=1.0, ls=":",
                    label=f"@{REF_FREQ_MHZ:.0f}MHz  amp={best_match['amp_rel']:.3f}")
            ax.legend(fontsize=6)

    fig.suptitle(f"{os.path.basename(args.infile)}\n{src_label}", fontsize=9)
    outpath = os.path.join(outdir, f"{stem}_vonmises_fit.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


if __name__ == "__main__":
    main()
