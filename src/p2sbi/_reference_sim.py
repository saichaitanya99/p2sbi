"""
_reference_sim.py — FROZEN. DO NOT EDIT.
----------------------------------------
Verbatim copy of `J2145_components()` and `sim_generator()` as they existed in
pulsarDMvary.py before the config refactor (commit fd7f1cf, lines 742-876).

Its only purpose is to let verify_refactor.py assert that the config-driven
simulator reproduces the hardcoded one BIT-FOR-BIT under the baseline config
`configs/ilofar_J2145.yaml`, as CLAUDE.md requires.

Any difference means a prior bound, phase centre or spectral index was
transcribed into the YAML wrong — and that would not be visible in the trained
posteriors until much later.

Editing this file defeats the entire point of it. If the config-driven
simulator is deliberately changed (new noise model, extra component, changed
prior), do not "fix" this file to match — instead retire the exact-equality
check for that config and record why in REFACTOR_PLAN.md.
"""

import numpy as np

from .pulsarDMvary import ProfileComponent, PulsarProfileSimulator


def J2145_components(gamma_leading, gamma_main, toa, amp_ref,
                     conc_ref_main, conc_ref_lead,
                     conc_ref_premain, conc_ref_inter,
                     amp_premain, amp_inter):
    main = ProfileComponent(
        name="Main",
        amplitude_ref=1.0,
        phase_center_ref=0.58036245 + toa,
        concentration_ref=conc_ref_main,
        spectral_index_amp=gamma_main,
        spectral_index_width=0.5,
        spectral_index_center=0.0,
        sigma_broadening_bins=0.0,
    )
    premain = ProfileComponent(
        name="PreMain",
        amplitude_ref=amp_premain,
        phase_center_ref=0.55222918 + toa,
        concentration_ref=conc_ref_premain,
        spectral_index_amp=gamma_main,
        spectral_index_width=0.5,
        spectral_index_center=0.0,
        sigma_broadening_bins=0.0,
    )
    leading = ProfileComponent(
        name="Leading",
        amplitude_ref=amp_ref,
        phase_center_ref=0.39381792 + toa,
        concentration_ref=conc_ref_lead,
        spectral_index_amp=gamma_leading,
        spectral_index_width=2.5,
        spectral_index_center=0.0,
        sigma_broadening_bins=0.0,
    )
    inter = ProfileComponent(
        name="Inter",
        amplitude_ref=amp_inter,
        phase_center_ref=0.43450154 + toa,
        concentration_ref=conc_ref_inter,
        spectral_index_amp=gamma_leading,
        spectral_index_width=2.5,
        spectral_index_center=0.0,
        sigma_broadening_bins=0.0,
    )
    return [main, premain, leading, inter]


def sim_generator(n_sims):
    #print(f"Generating {n_sims} simulated pulsar portraits...")
    for _ in range(n_sims):
        P0 = 1/62.295887837388923127 * 1000  # ms, from J2145-0750
        DM = np.random.uniform(-0.025, 0.025)
        gamma_leading = np.random.uniform(-4.0, 1.0)
        gamma_main    = np.random.uniform(-6.0, 1.0)   # extended to -6 — real data pushes against -4 boundary
        toa = np.random.uniform(-0.08, 0.08)
        snr = float(np.exp(np.random.uniform(np.log(7.0), np.log(120.0))))
        amp_ref = np.random.uniform(0.1, 0.5)
        conc_ref_main    = np.random.uniform(40, 100)  # fit: κ≈67 — sharp component
        conc_ref_premain = np.random.uniform(10,  40)  # fit: κ≈23 — broad bridge
        conc_ref_lead    = np.random.uniform(40, 100)  # fit: κ≈66 — sharp component
        conc_ref_inter   = np.random.uniform( 8,  30)  # fit: κ≈18 — broad bridge
        amp_premain      = np.random.uniform(0.25, 0.65)  # nuisance — marginalised over
        amp_inter        = np.random.uniform(0.05, 0.25)  # nuisance — marginalised over

        sim = PulsarProfileSimulator(
            n_phase_bins=512,
            n_freq_channels=40,
            freq_low_mhz=112.01177701950073,  # replicating I-LOFAR settings
            freq_high_mhz=190.13677701950073,  # replicating I-LOFAR settings
            reference_freq_mhz=151.07427701950073,
            period_ms=P0
        )

        components_complex = J2145_components(
            gamma_leading, gamma_main, toa, amp_ref,
            conc_ref_main, conc_ref_lead,
            conc_ref_premain, conc_ref_inter,
            amp_premain, amp_inter,
        )

        result_complex = sim.generate_portrait(
            components_complex,
            epoch_mjd=59500,
            dm=DM,
            tau_scatter_ms=0.0,   # tau_scatter_ms=tau_scatter — removed, sub-bin for J2145-0750
            add_noise=True,
            snr=snr
        )

        portrait=result_complex["portrait"]
        sigma_bandpass = 0.1  # nuisance parameter: fixed low-level bandpass irregularity, marginalised over
        portrait = portrait * np.random.lognormal(0, sigma_bandpass, size=(portrait.shape[0], 1))
        portrait = portrait / (portrait.max() + 1e-12)   # max-normalise to match real_archive_loader.py
        flux_per_channel = portrait.mean(axis=1)
        phase_axis = np.arange(portrait.shape[1])

        centroids    = (portrait * phase_axis).sum(axis=1) / (portrait.sum(axis=1) + 1e-12)

        # At low SNR, Gaussian noise can push centroids outside [0, N_bins).
        # Replace with MAIN_PHASE_BIN — same convention as real_archive_loader.py,
        # so the network sees the same sentinel value in training and inference.
        _n_bins   = portrait.shape[1]
        _main_bin = int(0.58036245 * _n_bins)   # 297 at 512 bins
        _bad_cent = (centroids < 0.0) | (centroids >= _n_bins)
        centroids[_bad_cent] = float(_main_bin)

        mean_profile = portrait.mean(axis=0)                          # (512,) frequency-averaged profile
        mean_profile = mean_profile / (mean_profile.max() + 1e-12)   # unit-peak normalised — decouples from SNR

        yield {
            'DM': DM,
            'gamma_leading': gamma_leading,
            'gamma_main': gamma_main,
            'toa': toa,
            'snr': snr,
            'amp_ref': amp_ref,
            'portrait': portrait,
            'flux_per_channel': flux_per_channel,
            'centroids': centroids,
            'mean_profile': mean_profile,
        }
