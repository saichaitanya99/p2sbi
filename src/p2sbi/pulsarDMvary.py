from math import gamma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy.special import i0
from scipy.ndimage import gaussian_filter1d
import warnings
import torch
warnings.filterwarnings('ignore')
import argparse
import os
import torch.nn as nn
from .pulsar_config import get_config
from .features import build_features
from .scintillation import draw_scint_gain
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
import platform
from scipy import cluster
from sbi.utils import BoxUniform

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"




@dataclass
class ProfileComponent:
    """Single von Mises component with frequency/time evolution"""
    # Base parameters at reference frequency
    amplitude_ref: float      # amplitude at reference freq
    phase_center_ref: float    # phase location [0, 1]
    concentration_ref: float   # von Mises κ (width parameter, larger = narrower)
    
    # Frequency evolution (power-law indices)
    spectral_index_amp: float = 0.0   # amplitude ∝ ν^α
    spectral_index_width: float = 0.0  # width evolution with frequency
    spectral_index_center: float = 0.0 # position drift with frequency (in phase units)
    
    # Time evolution parameters (fractional change per year)
    amplitude_drift_rate: float = 0.0  # slow amplitude change
    phase_drift_rate: float = 0.0      # slow position drift (phase/year)
    width_drift_rate: float = 0.0      # slow width change
    
    # Optional: mode switching
    mode_probability: float = 1.0  # probability component is "on"
    mode_timescale: float = 0.0    # switching timescale in years

    # Instrumental/profile broadening (Gaussian sigma in phase bins)
    sigma_broadening_bins: float = 0.0

    # Component label
    name: str = "Component"

class PulsarProfileSimulator:
    """Simulator for realistic pulsar profiles with evolution"""
    
    def __init__(
        self,
        n_phase_bins: int = 1024,
        n_freq_channels: int = 32,
        freq_low_mhz: float = 800,
        freq_high_mhz: float = 1600,
        reference_freq_mhz: float = 1200,
        period_ms: float = 1000.0,  # Pulsar period in milliseconds (default 1 second)
        freqs_mhz: Optional[np.ndarray] = None
    ):
        self.n_phase = n_phase_bins
        self.n_freq = n_freq_channels
        self.phases = np.linspace(0, 1, n_phase_bins, endpoint=False)
        self.period_ms = period_ms  # Store the period

        if freqs_mhz is not None:
            self.freqs_mhz = np.asarray(freqs_mhz, dtype=float)
            if self.freqs_mhz.size != n_freq_channels:
                raise ValueError(
                    f"freqs_mhz has {self.freqs_mhz.size} entries but "
                    f"n_freq_channels={n_freq_channels}"
                )
        else:
            self.freqs_mhz = np.linspace(
                freq_low_mhz,
                freq_high_mhz,
                n_freq_channels
            )
        self.ref_freq = reference_freq_mhz
        
        # Pre-compute frequency ratios for efficiency
        self.freq_ratio = self.freqs_mhz / self.ref_freq
        
        # Pre-compute per-channel bandwidths (Δν) from log-spaced centers for DM smearing and noise
        # Compute edges in log-frequency space for robust bandwidth estimates
        logc = np.log10(self.freqs_mhz)
        edges_log = np.empty(self.n_freq + 1)
        edges_log[1:-1] = 0.5 * (logc[:-1] + logc[1:])
        # Extrapolate first and last edges assuming local spacing
        edges_log[0] = logc[0] - 0.5 * (logc[1] - logc[0]) if self.n_freq > 1 else logc[0] - 0.5
        edges_log[-1] = logc[-1] + 0.5 * (logc[-1] - logc[-2]) if self.n_freq > 1 else logc[-1] + 0.5
        self.channel_edges_mhz = 10 ** edges_log
        self.channel_bw_mhz = self.channel_edges_mhz[1:] - self.channel_edges_mhz[:-1]

    @classmethod
    def from_config(cls, cfg) -> "PulsarProfileSimulator":
        return cls(
            n_phase_bins       = cfg.n_bins,
            n_freq_channels    = cfg.n_chan,
            freq_low_mhz       = cfg.freq_low_mhz,
            freq_high_mhz      = cfg.freq_high_mhz,
            reference_freq_mhz = cfg.ref_freq_mhz,
            period_ms          = cfg.period_ms,
            # True channel centres when the config declares them, else the
            # linspace across the band. cfg.freqs_mhz decides.
            freqs_mhz          = cfg.freqs_mhz,
        )

    def generate_portrait(
        self,
        components: List[ProfileComponent],
        epoch_mjd: float,
        mjd_start: float = 58000.0,
        dm: float = 10.0,  # Dispersion measure in pc/cm^3
        tau_scatter_ms: float = 0.0,  # Scattering time at reference frequency in ms
        scattering_index: float = -4.4,  # Kolmogorov turbulence scattering index
        dm_smearing: bool = False,  # Intra-channel DM smearing toggle
        add_noise: bool = True,
        snr: float = 100.0,
        seed: Optional[int] = None,
        enable_time_evolution: bool = True,
        noise_model: str = 'avg_peak',  # 'avg_peak' | 'per_channel_peak' | 'radiometer'
        npol: int = 2,
        t_obs_s: float = 30.0,
        channel_gain: Optional[np.ndarray] = None,  # per-channel scintillation gain (Stage S)
        noise_scale: Optional[np.ndarray] = None,   # per-channel system-temperature factor (Stage S6b)
    ) -> Dict:

        
        if seed is not None:
            np.random.seed(seed)
            
        # Time since start (in years for slow evolution)
        time_years = (epoch_mjd - mjd_start) / 365.25
        if not enable_time_evolution:
            time_years = 0.0
        
        # Initialize clean portrait
        portrait_clean = np.zeros((self.n_freq, self.n_phase))
        
        # Add each component with evolution
        component_portraits = []
        for comp in components:
            comp_portrait = self._generate_component(comp, time_years, epoch_mjd)
            portrait_clean += comp_portrait
            component_portraits.append(comp_portrait)
        
        # Apply propagation effects
        portrait_propagated = self._apply_ism_effects(
            portrait_clean, dm, tau_scatter_ms, scattering_index, dm_smearing
        )

        # Scintillation: signal only, before the noise
        if channel_gain is not None:
            portrait_propagated = portrait_propagated * np.asarray(channel_gain, dtype=float)[:, None]

        # Add noise
        if add_noise:
            portrait_final = self._add_radiometer_noise(
                portrait_propagated, snr, noise_model=noise_model, npol=npol, t_obs_s=t_obs_s,
                noise_scale=noise_scale,
            )
        else:
            portrait_final = portrait_propagated

        return {
            'portrait': portrait_final,
            'portrait_clean': portrait_clean,
            'portrait_propagated': portrait_propagated,
            'component_portraits': component_portraits,
            'epoch_mjd': epoch_mjd,
            'time_years': time_years,
            'channel_gain': channel_gain,
            'noise_scale': noise_scale,
        }
    
    def _generate_component(
        self, 
        comp: ProfileComponent, 
        time_years: float,
        epoch_mjd: float
    ) -> np.ndarray:
        """Generate single component with all evolution"""
        portrait = np.zeros((self.n_freq, self.n_phase))
        
        # Mode switching disabled for now: always generate component
        # Frequency evolution
        amplitudes = comp.amplitude_ref * self.freq_ratio ** comp.spectral_index_amp
        
        # Width evolution (concentration parameter)
        concentrations = comp.concentration_ref * self.freq_ratio ** comp.spectral_index_width
        
        # Position evolution
        centers = comp.phase_center_ref + comp.spectral_index_center * np.log10(self.freq_ratio)
        
        # Apply slow time evolution
        amplitudes *= (1 + comp.amplitude_drift_rate * time_years)
        concentrations *= (1 + comp.width_drift_rate * time_years)
        centers += comp.phase_drift_rate * time_years
        
        # Ensure phase wrapping
        centers = centers % 1.0
        
        # Generate von Mises profiles for each frequency
        for i in range(self.n_freq):
            if amplitudes[i] > 0 and concentrations[i] > 0:
                portrait[i] = self._von_mises(centers[i], concentrations[i], amplitudes[i])
                if comp.sigma_broadening_bins > 0:
                    portrait[i] = gaussian_filter1d(portrait[i], sigma=comp.sigma_broadening_bins)

        return portrait
    
    def _von_mises(self, mu: float, kappa: float, amplitude: float) -> np.ndarray:
        """Generate von Mises distribution"""
        # Convert phase center to radians
        mu_rad = 2 * np.pi * mu
        phases_rad = 2 * np.pi * self.phases
        
        # von Mises formula with normalization
        # Using scipy's i0 for numerical stability with large kappa
        profile = amplitude * np.exp(kappa * (np.cos(phases_rad - mu_rad) - 1))
        
        # Normalize to preserve amplitude
        if profile.max() > 0:
            profile = profile / profile.max() * amplitude
            
        return profile
    
    def _apply_ism_effects(
        self,
        portrait: np.ndarray,
        dm: float,
        tau_scatter_ms: float,
        scattering_index: float,
        dm_smearing: bool = False
    ) -> np.ndarray:
        """Apply ISM propagation effects (DM and scattering)"""
        portrait_out = portrait.copy()

        # Apply dispersion (DM)
        if dm > -1:
            portrait_out = self._apply_dispersion(portrait_out, dm, dm_smearing=dm_smearing)

        # Apply scattering
        if tau_scatter_ms > 0:
            portrait_out = self._apply_scattering(portrait_out, tau_scatter_ms, scattering_index)
            
        return portrait_out
    
    def _apply_dispersion(self, portrait: np.ndarray, dm: float, dm_smearing: bool = False) -> np.ndarray:
        """Apply dispersion delay across frequency channels"""
        portrait_out = np.zeros_like(portrait)
        
        # DM constant (MHz^2 * ms / (pc cm^-3));
        # 4.1488064239e3 with seconds → 4.1488064239e6 in milliseconds
        k_dm = 4.1488064239e6
        
        # Calculate delays relative to highest frequency
        ref_freq = self.freqs_mhz.max()
        delays_ms = k_dm * dm * (1/self.freqs_mhz**2 - 1/ref_freq**2)
        
        # Convert to phase bins using the actual pulsar period
        delays_phase = delays_ms / self.period_ms
        delays_bins = delays_phase * self.n_phase
        
        # Helper: fractional circular shift via FFT (keeps periodicity)
        def frac_shift(row: np.ndarray, shift_bins: float) -> np.ndarray:
            n = row.size
            k = np.fft.fftfreq(n)
            phase = np.exp(-2j * np.pi * k * shift_bins)
            return np.fft.ifft(np.fft.fft(row) * phase).real
        
        # Fractionally shift each channel
        shifted = np.zeros_like(portrait)
        for i in range(self.n_freq):
            shifted[i] = frac_shift(portrait[i], float(delays_bins[i]))
        
        if dm_smearing:
            # Intra-channel DM smearing: symmetric boxcar blur with
            # Δt_smear_ms ≈ 8.3e-3 * DM * Δν_MHz / ν_GHz^3
            nu_ghz = self.freqs_mhz / 1000.0
            dt_smear_ms = 8.3e-3 * dm * self.channel_bw_mhz / (nu_ghz ** 3)
            smear_bins = (dt_smear_ms / self.period_ms) * self.n_phase
            
            # Helper: circular convolution using FFT
            def circ_convolve(row: np.ndarray, kernel: np.ndarray) -> np.ndarray:
                n = row.size
                kvec = np.zeros(n)
                L = kernel.size
                idx = (np.arange(L) - L//2) % n
                kvec[idx] = kernel
                return np.fft.ifft(np.fft.fft(row) * np.fft.fft(kvec)).real
            
            for i in range(self.n_freq):
                L = int(np.ceil(float(smear_bins[i])))
                if L > 1:
                    kernel = np.ones(L) / L
                    portrait_out[i] = circ_convolve(shifted[i], kernel)
                else:
                    portrait_out[i] = shifted[i]
        else:
            portrait_out = shifted
        
        return portrait_out
    
    def _apply_scattering(self, portrait: np.ndarray, tau_ref_ms: float, index: float) -> np.ndarray:
        tau_bins = (tau_ref_ms * (self.freqs_mhz / self.ref_freq) ** index
                    / self.period_ms * self.n_phase)
        f = np.fft.fftfreq(self.n_phase)                      # cycles per bin
        H = 1.0 / (1.0 + 2j * np.pi * f[None, :] * tau_bins[:, None])
        return np.fft.ifft(np.fft.fft(portrait, axis=1) * H, axis=1).real
    
    def _add_radiometer_noise(self, portrait: np.ndarray, snr: float,
                               noise_model: str = 'avg_peak',
                               npol: int = 2, t_obs_s: float = 30.0,
                               noise_scale: Optional[np.ndarray] = None) -> np.ndarray:
        n = self.n_freq
        prof_avg = portrait.mean(axis=0)
        A = float(np.max(prof_avg)) if np.any(prof_avg > 0) else 1.0
        A = max(A, 1e-12)
        ns = 1.0 if noise_scale is None else np.asarray(noise_scale, dtype=float)[:, None]

        if noise_model == 'per_channel_peak':
            peaks = np.max(portrait, axis=1)
            sigma = (peaks / max(snr, 1e-6))[:, None]
            noise = np.random.normal(0.0, 1.0, portrait.shape) * sigma
            return portrait + noise * ns
        elif noise_model == 'radiometer':
            # sigma_i ∝ 1 / sqrt(npol * Δν_i * t_obs) (constant SEFD implicit)
            base = 1.0 / np.sqrt(np.maximum(self.channel_bw_mhz, 1e-9) * 1e6 * max(t_obs_s, 1e-6) * max(npol, 1))
            denom = np.sqrt(np.sum(base**2))
            c = (A / max(snr, 1e-6)) * (n / max(denom, 1e-12))
            sigma = (c * base)[:, None]
            noise = np.random.normal(0.0, 1.0, portrait.shape) * sigma
            return portrait + noise * ns
        else:
            sigma_pix = (A / max(snr, 1e-6)) * np.sqrt(n)
            noise = np.random.normal(0.0, sigma_pix, portrait.shape)
            return portrait + noise * ns

def plot_portrait(portrait_dict, title="Pulsar Portrait", components=None):
    """Comprehensive visualization of portrait data"""
    try:
        from .natastro_style import set_style
        _usetex = set_style()
    except ImportError:
        _usetex = False

    _pbin  = "Phase bin"
    _intens = "Intensity"
    _freq_mhz = "Frequency (MHz)"
    _freq_ch  = "Frequency channel"
    _norm_off = "Normalised intensity (offset)" if not _usetex else r"Normalised intensity (offset)"


    fig = plt.figure(figsize=(9.0, 5.5))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    portrait = portrait_dict['portrait']
    n_freq, n_phase = portrait.shape

    # Main 2D portrait
    ax1 = fig.add_subplot(gs[:2, :2])
    im = ax1.imshow(portrait, aspect='auto', origin='lower',
                    cmap='viridis', interpolation='bilinear')
    ax1.set_xlabel(_pbin)
    if 'freqs_mhz' in portrait_dict:
        freqs = portrait_dict['freqs_mhz']
        tick_idx = [0, n_freq // 4, n_freq // 2, 3 * n_freq // 4, n_freq - 1]
        ax1.set_yticks(tick_idx)
        ax1.set_yticklabels([f"{freqs[i]:.0f}" for i in tick_idx])
        ax1.set_ylabel(_freq_mhz)
    else:
        ax1.set_ylabel(_freq_ch)
    plt.colorbar(im, ax=ax1, label=_intens, fraction=0.046, pad=0.02)

    # Frequency-averaged profile
    ax2 = fig.add_subplot(gs[2, :2])
    profile_avg = np.mean(portrait, axis=0)
    ax2.plot(profile_avg, 'k-', label='Observed')
    if 'portrait_clean' in portrait_dict:
        profile_clean_avg = np.mean(portrait_dict['portrait_clean'], axis=0)
        ax2.plot(profile_clean_avg, 'b--', alpha=0.7, label='Intrinsic')
    ax2.set_xlabel(_pbin)
    ax2.set_ylabel(_intens)
    ax2.legend()



    # Profiles at different frequencies
    ax3 = fig.add_subplot(gs[:2, 2])
    freq_indices = [0, n_freq // 4, n_freq // 2, 3 * n_freq // 4, n_freq - 1]
    offsets = [idx / 5 for idx in freq_indices]

    for idx, off in zip(freq_indices, offsets):
        pmax = portrait[idx].max()
        profile_norm = portrait[idx] / pmax if pmax > 0 else portrait[idx]
        ax3.plot(profile_norm + off, color='k', lw=0.9)

    ax3.set_xlabel(_pbin)
    ax3.set_ylabel(_norm_off)

    # Right-hand axis labelling each trace by channel number
    ax3r = ax3.twinx()
    ax3r.set_ylim(ax3.get_ylim())          # must share the limits
    ax3r.set_yticks(offsets)
    ax3r.set_yticklabels([str(i) for i in freq_indices])
    ax3r.set_ylabel('Frequency channel')
    ax3r.tick_params(axis='y', length=3)

    # Component breakdown if available
    if components and 'component_portraits' in portrait_dict:
        ax4 = fig.add_subplot(gs[2, 2])
        for comp, comp_portrait in zip(components, portrait_dict['component_portraits']):
            comp_avg = np.mean(comp_portrait, axis=0)
            ax4.plot(comp_avg, label=comp.name, alpha=0.7)
        ax4.set_xlabel(_pbin)
        ax4.set_ylabel(_intens)
        ax4.legend()

    return fig

def plot_time_evolution(simulator, components, mjds, dm=10.0, scattering_time=0.5):
    """Show how profile evolves over time"""
    try:
        from .natastro_style import set_style, NAT_W_DOUBLE
        set_style()
        fig_w = NAT_W_DOUBLE
    except ImportError:
        fig_w = 9.0

    n_epochs = len(mjds)
    fig, axes = plt.subplots(2, n_epochs, figsize=(fig_w, 3.8))

    if n_epochs == 1:
        axes = axes.reshape(2, 1)

    portraits = []
    for i, mjd in enumerate(mjds):
        result = simulator.generate_portrait(
            components, mjd,
            dm=dm,
            scattering_time=scattering_time,
            add_noise=False,
        )
        portraits.append(result)

        axes[0, i].imshow(result['portrait'], aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_xlabel('Phase bin')
        if i == 0:
            axes[0, i].set_ylabel('Frequency channel')

        profile = np.mean(result['portrait'], axis=0)
        axes[1, i].plot(profile, 'k-')
        axes[1, i].set_xlabel('Phase bin')
        if i == 0:
            axes[1, i].set_ylabel('Intensity')

    plt.tight_layout(pad=0.5)
    return fig, portraits

#define a complex profile with multiple components
def complex_comp_old(mp_comps=2, ip_comps=1):
    comps = []
    
    # --- 1. Generate correlated spectral parameters for main pulse ---
    base_spectral_index_amp = np.random.uniform(-1.5, -0.5)
    base_spectral_index_width = np.random.uniform(-0.1, 0.1)

    
    main_phase_ref = np.random.uniform(0.2, 0.3)
    for i in range(mp_comps):
        if i == 0:
            # First component (main pulse) has distinct parameters
            comp = ProfileComponent(
                name=f"Main Pulse",
                amplitude_ref=np.random.uniform(0.5, 1.0),
                phase_center_ref=main_phase_ref,
                concentration_ref=np.random.uniform(50, 120),
                spectral_index_amp=base_spectral_index_amp + np.random.normal(0, 0.02),
                spectral_index_width=base_spectral_index_width + np.random.normal(0, 0.02),
                amplitude_drift_rate=np.random.uniform(-0.03, 0.03),
                phase_drift_rate=np.random.uniform(-0.001, 0.001),
                width_drift_rate=np.random.uniform(-0.01, 0.01)
            )
        else:
            comp_type = np.random.choice(['leading', 'trailing'])
            if comp_type == 'leading':
                phase_center = main_phase_ref - np.random.uniform(0.01, 0.20)
            else:
                phase_center = main_phase_ref + np.random.uniform(0.01, 0.20)
            comp = ProfileComponent(
                name=f"Component_{i+1}_{comp_type}",
                amplitude_ref=np.random.uniform(0.1, 0.5),
                phase_center_ref=phase_center,
                concentration_ref=np.random.uniform(30, 100),
                spectral_index_amp=base_spectral_index_amp + np.random.normal(-0.05, 0.05),
                spectral_index_width=base_spectral_index_width + np.random.normal(-0.02, 0.02),
                amplitude_drift_rate=np.random.uniform(-0.03, 0.03),
                phase_drift_rate=np.random.uniform(-0.001, 0.001),
                width_drift_rate=np.random.uniform(-0.01, 0.01)
            )
        comps.append(comp)
    
    # --- 2. Add interpulse if requested ---
    if ip_comps > 0:
        ip_base_spectral_index_amp = base_spectral_index_amp + np.random.normal(-0.5, 0.1)
        ip_base_spectral_index_width = base_spectral_index_width + np.random.normal(0, 0.05)
        ip_amp_ref = np.random.uniform(0.1, 0.5)
        for ip_comp_index in range(ip_comps):
            if ip_comp_index == 0:
                main_phase_center = np.mean([c.phase_center_ref for c in comps])
                ip_phase_center = (main_phase_center + 0.5) % 1.0  # 180 deg away
                ip_comp = ProfileComponent(
                    name="Interpulse",
                    amplitude_ref=ip_amp_ref,
                    phase_center_ref=ip_phase_center + np.random.normal(0, 0.02),  # small jitter
                    concentration_ref=np.random.uniform(30, 80),
                    spectral_index_amp=ip_base_spectral_index_amp + np.random.normal(0, 0.05),  # steeper
                    spectral_index_width=ip_base_spectral_index_width + np.random.normal(0, 0.02),
                    amplitude_drift_rate=np.random.uniform(-0.03, 0.03),
                    phase_drift_rate=np.random.uniform(-0.001, 0.001),
                    width_drift_rate=np.random.uniform(-0.01, 0.01)
                )
            else:
                ip_comp_type = np.random.choice(['leading_ip', 'trailing_ip'])
                if ip_comp_type == 'leading_ip':
                    ip_phase_center = (main_phase_center + 0.5 - np.random.uniform(0.01, 0.05)) % 1.0
                else:
                    ip_phase_center = (main_phase_center + 0.5 + np.random.uniform(0.01, 0.05)) % 1.0
                ip_comp = ProfileComponent(
                    name=f"Interpulse_{ip_comp_type}",
                    amplitude_ref=np.random.uniform(ip_amp_ref*0.05, ip_amp_ref),
                    phase_center_ref=ip_phase_center + np.random.normal(0, 0.02),  # small jitter
                    concentration_ref=np.random.uniform(30, 80),
                    spectral_index_amp=ip_base_spectral_index_amp + np.random.normal(0, 0.05),  # steeper
                    spectral_index_width=ip_base_spectral_index_width + np.random.normal(0, 0.02),
                    amplitude_drift_rate=np.random.uniform(-0.03, 0.03),
                    phase_drift_rate=np.random.uniform(-0.001, 0.001),
                    width_drift_rate=np.random.uniform(-0.01, 0.01)
                )
            comps.append(ip_comp)
    
    return comps

def complex_comp(mp_comps=30):
    comps = []
    
    # --- 0. Choose the number of physical peaks (clusters) ---
    # Realistic MSP-like distribution
    n_clusters = 1#np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
    
    #print(f"Generating profile with {n_clusters} physical peaks (clusters)")
    # Generate cluster centers around pulse phase
    cluster_centers = 0.5#np.random.uniform(0, 1, size=n_clusters)
    
    
    # width of each physical peak
    peak_width = 0.03

    # --- 1. Generate correlated spectral parameters for main pulse ---
    base_spectral_index_amp = -1.2#np.random.uniform(-1.5, -0.5)
    base_spectral_index_width = -0.05#np.random.uniform(-0.1, 0.1)

    # Put the main pulse inside cluster 0
    
    main_phase_ref = cluster_centers# + np.random.normal(0, peak_width)
    main_phase_ref %= 1.0
    main_amp_ref = 1.0#np.random.uniform(0.5, 1.0)
    # --- 1a. Main Pulse Component ---
    comp_main = ProfileComponent(
        name=f"Main Pulse",
        amplitude_ref=main_amp_ref,
        phase_center_ref=main_phase_ref,
        concentration_ref=100,#np.random.uniform(50, 120),
        spectral_index_amp=1,# + np.random.normal(0, 0.02),
        spectral_index_width=base_spectral_index_width,# + np.random.normal(0, 0.02),
        amplitude_drift_rate=0,#np.random.uniform(-0.03, 0.03),
        phase_drift_rate=0,#np.random.uniform(-0.001, 0.001),
        width_drift_rate=0,#np.random.uniform(-0.01, 0.01)
    )
    comps.append(comp_main)

    # --- 1b. Additional components belonging to the main pulse cluster ---
    for i in range(1, mp_comps):
        comp_type = 'leading'#np.random.choice(['leading', 'trailing'])
        
        # Leading/trailing offset inside the same peak cluster
        offset = np.random.uniform(0.1, 0.3)
        phase = (
            main_phase_ref - offset if comp_type == 'leading'
            else main_phase_ref + offset
        ) % 1.0

        comp = ProfileComponent(
            name=f"Component_{i}_{comp_type}",
            amplitude_ref=0.2,
            phase_center_ref=phase,
            concentration_ref=50,#np.random.uniform(30, 100),
            spectral_index_amp=1.5,# + np.random.normal(-0.05, 0.05),
            spectral_index_width=base_spectral_index_width,# + np.random.normal(-0.02, 0.02),
            amplitude_drift_rate=0,#np.random.uniform(-0.03, 0.03),
            phase_drift_rate=0,#np.random.uniform(-0.001, 0.001),
            width_drift_rate=0,#np.random.uniform(-0.01, 0.01)
        )
        comps.append(comp)

    return comps

def compute_phase_aligned_profile(portrait):
    """
    portrait: (n_channels, n_time_bins) numpy array
    """
    # Frequency average
    profile = portrait.mean(axis=0)  # (512,)
    
    # Centre of mass
    indices = np.arange(len(profile))
    centroid = (profile * indices).sum() / profile.sum().clip(min=1e-8)
    
    # Roll to centre
    shift = len(profile) // 2 - int(centroid)
    aligned = np.roll(profile, shift)
    
    # Normalise so amplitude doesnt interfere with shape
    aligned = aligned / aligned.max().clip(min=1e-8)
    
    return aligned  # (512,)


def extract_leading_component_profile(portrait, phase_center, half_width, amplify=10.0, centroids=None):
    if centroids is not None:
        centroid_mean = centroids.mean()
        portrait = np.stack([
            np.roll(portrait[i], -int(round(centroids[i] - centroid_mean)))
            for i in range(portrait.shape[0])
        ])

    left  = phase_center - half_width
    right = phase_center + half_width

    # Crop to leading component region
    leading_crop = portrait[:, left:right]        # (n_channels, width)

    # Frequency average to 1D
    leading_profile = leading_crop.mean(axis=0)   # (width,)

    # Amplify
    leading_profile = leading_profile * amplify

    # Phase align within this window
    indices  = np.arange(len(leading_profile))
    mass     = leading_profile.sum().clip(min=1e-8)
    centroid = (leading_profile * indices).sum() / mass
    shift    = len(leading_profile) // 2 - int(centroid)
    aligned  = np.roll(leading_profile, shift)

    # Normalise so only shape information remains
    aligned = aligned / aligned.max().clip(min=1e-8)

    return aligned  # (2 * half_width,)


def build_components(cfg, sampled: Dict[str, float]) -> List[ProfileComponent]:
    return [ProfileComponent(**kw) for kw in cfg.build_component_kwargs(sampled)]


def _place_generic_dead(mask, k, generic, freqs):
    modes  = [m for m in generic["modes"]]
    w      = np.array([generic["modes"][m] for m in modes], dtype=float)
    w      = w / w.sum()
    target = int(mask.sum()) + k
    while mask.sum() < target:
        live = np.flatnonzero(~mask)
        if live.size == 0:
            break
        if modes[np.random.choice(len(modes), p=w)] == "random":
            mask[np.random.choice(live)] = True
            continue
        width  = np.random.uniform(*generic["block_width_mhz"])
        centre = np.random.uniform(freqs.min(), freqs.max())
        near   = live[np.argsort(np.abs(freqs[live] - centre), kind="stable")]
        run    = near[np.abs(freqs[near] - centre) <= 0.5 * width]
        if run.size == 0:
            run = near[:1]                       # narrower than a channel: one channel
        mask[run[: target - int(mask.sum())]] = True


def _draw_dead_mask(cfg):
    dc = cfg.dead_channels
    if dc is None:
        return None
    n     = cfg.n_chan
    freqs = np.asarray(cfg.freqs_mhz, dtype=float)
    mask  = np.zeros(n, dtype=bool)

    wo = dc["wipeout"]
    if wo["p"] > 0.0 and np.random.random() < wo["p"]:
        k = int(round(np.random.uniform(*wo["frac"]) * n))
        mask[np.random.choice(n, min(max(k, 1), n - 1), replace=False)] = True
        return mask

    if dc["site_idx"].size:
        sigma = dc["bad_day_sigma"]
        z     = np.random.normal(0.0, sigma) if sigma > 0.0 else 0.0
        p_now = 0.5 * (1.0 + np.tanh(0.5 * (dc["site_logit"] + z)))   # expit
        mask[dc["site_idx"][np.random.random(dc["site_idx"].size) < p_now]] = True

    g = dc["generic"]
    if g["p_any"] > 0.0 and np.random.random() < g["p_any"]:
        k = min(1 + int(np.random.poisson(g["n_mean"] - 1.0)),
                dc["max_dead"] - int(mask.sum()))
        if k > 0:
            _place_generic_dead(mask, k, g, freqs)
    return mask


def _draw_channel_effects(cfg, sampled, sim):
    out = {"channel_gain": None, "noise_scale": None}
    values = cfg.scintillation_values(sampled)
    if values is not None:
        out["channel_gain"] = draw_scint_gain(values, sim.freqs_mhz, sim.channel_edges_mhz,
                                              cfg.ref_freq_mhz)
    if cfg.tsys_scatter > 0:
        s = np.random.lognormal(0.0, cfg.tsys_scatter, size=sim.n_freq)
        out["noise_scale"] = s / np.sqrt(np.mean(s ** 2))
    return out


def _finalise_portrait(portrait, cfg, dead_mask=None):
    portrait = portrait * np.random.lognormal(
        0, cfg.sigma_bandpass, size=(portrait.shape[0], 1)
    )

    if dead_mask is None:
        dead_mask = _draw_dead_mask(cfg)
    if dead_mask is not None:
        dead_mask = np.asarray(dead_mask, dtype=bool)
        if dead_mask.shape != (portrait.shape[0],):
            raise ValueError(f"dead_mask has shape {dead_mask.shape}, "
                             f"expected ({portrait.shape[0]},)")
        if dead_mask.any():
            portrait[dead_mask, :] = 0.0

    # Normalisation (trap 4) and the per-channel centroid (trap 3) both come
    # from features.py, which real_archive_loader.py also calls. They used to be
    # implemented separately in each file and kept in step by hand.
    portrait, flux_per_channel, centroids = build_features(portrait, cfg)

    # DIAGNOSTIC ONLY — consumed by simulation_maker.py plots. Deliberately NOT
    # part of the feature vector: frequency-averaging bakes the spectral indices
    # into the profile shape, which contaminates ToA. Do not re-add this to x.
    mean_profile = portrait.mean(axis=0)
    mean_profile = mean_profile / (mean_profile.max() + 1e-12)

    return portrait, flux_per_channel, centroids, mean_profile


def simulate_from_theta(theta, config_path=None, seed=None, dead_mask=None):
    cfg = get_config(config_path)

    if seed is not None:
        np.random.seed(seed)

    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size != len(cfg.inferred):
        raise ValueError(
            f"theta has {theta.size} entries but config {os.path.basename(cfg.path)} "
            f"declares {len(cfg.inferred)} inferred parameters "
            f"({', '.join(cfg.param_names)})"
        )

    # Invert the theta transform, then redraw the nuisance parameters.
    sampled = {p.name: (float(np.exp(v)) if p.log else float(v))
               for p, v in zip(cfg.inferred, theta)}
    for p in cfg.nuisance:
        sampled[p.name] = p.draw()

    scattering = cfg.scattering_values(sampled)
    sim        = PulsarProfileSimulator.from_config(cfg)
    chan_fx    = _draw_channel_effects(cfg, sampled, sim)   # Nones without the Stage S options

    result = sim.generate_portrait(
        build_components(cfg, sampled),
        epoch_mjd        = cfg.epoch_mjd,
        dm               = sampled[cfg.dm_param],
        tau_scatter_ms   = scattering["tau_ms"],
        scattering_index = scattering["spectral_index"],
        dm_smearing      = cfg.dm_smearing,
        add_noise        = cfg.add_noise,
        snr              = sampled[cfg.snr_param],
        noise_model      = cfg.noise_model,
        npol             = cfg.npol,
        t_obs_s          = cfg.t_obs_s,
        **chan_fx,
    )

    portrait, flux, centroids, _ = _finalise_portrait(result["portrait"], cfg,
                                                      dead_mask=dead_mask)
    return portrait, flux, centroids


def sim_generator(n_sims, config_path=None):
    cfg = get_config(config_path)

    # Hoisted out of the loop — none of this consumes RNG, so the draw order is
    # unaffected, but it saves re-deriving them on every one of N simulations.
    sim          = PulsarProfileSimulator.from_config(cfg)
    dm_key       = cfg.dm_param
    snr_key      = cfg.snr_param
    epoch        = cfg.epoch_mjd
    smearing     = cfg.dm_smearing
    do_noise     = cfg.add_noise
    noise_model  = cfg.noise_model
    npol         = cfg.npol
    t_obs        = cfg.t_obs_s

    for _ in range(n_sims):
        if n_sims > 1 and _ * 100 % n_sims == 0:
            print(f"{_/(n_sims-1)*100:.1f}% simulations done", end='\r')

        # --- 1. parameter draws (inferred then nuisance, YAML order) ---
        sampled    = cfg.sample()
        scattering = cfg.scattering_values(sampled)

        components = build_components(cfg, sampled)

        # --- 1b. per-channel effects: scintillation gain, Tsys factor (Nones when off) ---
        chan_fx = _draw_channel_effects(cfg, sampled, sim)

        # --- 2. portrait + noise realisation ---
        result = sim.generate_portrait(
            components,
            epoch_mjd        = epoch,
            dm               = sampled[dm_key],
            tau_scatter_ms   = scattering["tau_ms"],
            scattering_index = scattering["spectral_index"],
            dm_smearing      = smearing,
            add_noise        = do_noise,
            snr              = sampled[snr_key],
            noise_model      = noise_model,
            npol             = npol,
            t_obs_s          = t_obs,
            **chan_fx,
        )

        # --- 3. bandpass, normalisation, summary features (traps 3 and 4) ---
        portrait, flux_per_channel, centroids, mean_profile = _finalise_portrait(
            result["portrait"], cfg
        )

        yield {
            'theta':            cfg.theta_vector(sampled),
            'sampled':          sampled,
            'portrait':         portrait,
            'flux_per_channel': flux_per_channel,
            'centroids':        centroids,
            'mean_profile':     mean_profile,
        }

def get_device():
    # CUDA (cluster GPUs)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon (MPS)
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")

    # Fallback
    return torch.device("cpu")

# ------------------------------
# 2. Simple CNN Encoder
# ------------------------------
class EncoderCNN_old(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 32x1024 → 16x512

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 16x512 → 8x256
        )

        # Automatically compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 400, 512)
            flat_dim = self.conv(dummy).numel()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class EncoderCNN(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(1),  # Only pool in phase dimension

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(1),  # Only pool in phase dimension

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(1),  # Only pool in phase dimension
        )

        # Global average pooling to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        return self.fc(x)







class MultiBranchEncoder(nn.Module):
    """Fixed 2D CNN over the portrait, plus flux and centroid branches.

    Input layout (must match sim_maker.py's x assembly):
        portrait.flatten() | flux_per_channel | centroids
    Output: 128-dim embedding.
    """

    def __init__(self, n_chan: int = 40, n_phase: int = 512):
        super().__init__()

        # Three MaxPool2d(2) layers halve the frequency axis each time, so a
        # config with fewer than 8 channels collapses it to zero width and
        # fails deep inside the conv stack with an unhelpful shape error.
        if n_chan < 8:
            raise ValueError(
                f"MultiBranchEncoder needs n_chan >= 8 (got {n_chan}): the CNN "
                f"branch halves the frequency axis three times. Use "
                f"ChannelSetEncoder for narrow-band or few-channel setups."
            )

        self.n_chan  = n_chan
        self.n_phase = n_phase

        # -----------------------------
        # CNN branch (portrait)
        # -----------------------------
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((5,4)),
            nn.Flatten(),

            nn.Linear(32*5*4, 128),
            nn.ReLU()
        )

        # -----------------------------
        # Flux branch (weaker)
        # -----------------------------
        self.mlp_branch = nn.Sequential(
            nn.Linear(n_chan, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # -----------------------------
        # Centroid branch (toa + DM)
        # -----------------------------
        self.centroid_branch = nn.Sequential(
            nn.Linear(n_chan + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # -----------------------------
        # Combined representation — 128 + 32 + 32 = 192 → 128 output
        # mean_profile branch removed: frequency-averaging encodes spectral indices
        # (γ_lead, γ_main) into the profile shape, contaminating ToA inference
        # -----------------------------
        self.combined_fc = nn.Sequential(
            nn.Linear(128 + 32 + 32, 128),
            nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, **overrides) -> "MultiBranchEncoder":
        """Shape the encoder to the config's instrument setup."""
        return cls(n_chan=cfg.n_chan, n_phase=cfg.n_bins, **overrides)

    def forward(self, x):

        x_flat = x.view(x.shape[0], -1)

        N, P  = self.n_chan, self.n_phase
        split = N * P

        portrait = x_flat[:, :split].view(-1, 1, N, P)
        flux     = x_flat[:, split     : split + N]
        centroid = x_flat[:, split + N : split + 2 * N]

        cnn_out      = self.cnn_branch(portrait)
        mlp_out      = self.mlp_branch(flux)
        centroid_out = self.centroid_branch(torch.cat([centroid, cnn_out], dim=1))

        combined = torch.cat([cnn_out, mlp_out, centroid_out], dim=1)
        return self.combined_fc(combined)




class ChannelSetEncoder(nn.Module):
    """Channel-set attention encoder for pulsar portraits.

    Replaces the fixed 2D CNN in MultiBranchEncoder with an architecture
    that treats each frequency channel as an independent token, making
    the channel frequency explicit rather than implicit in grid position.
    This enables generalisation to multi-telescope data with different
    channel counts and frequency grids.
    """

    def __init__(
        self,
        n_chan:        int   = 40,
        n_phase:       int   = 512,
        d_token:       int   = 64,
        d_model:       int   = 128,
        n_heads:       int   = 4,
        n_attn_layers: int   = 2,
        freqs_mhz:     Optional[np.ndarray] = None,
        ref_freq_mhz:  Optional[float] = None,
        mask_dead_tokens: bool = False,
    ):
        super().__init__()

        self.n_chan  = n_chan
        self.n_phase = n_phase
        self.mask_dead_tokens = bool(mask_dead_tokens)

        if freqs_mhz is None or ref_freq_mhz is None:
            raise ValueError(
                "ChannelSetEncoder needs both freqs_mhz and ref_freq_mhz — build "
                "it with ChannelSetEncoder.from_config(cfg) so the channel grid "
                "matches the simulator's."
            )
        freqs = torch.as_tensor(np.asarray(freqs_mhz, dtype=np.float32))
        if freqs.numel() != n_chan:
            raise ValueError(
                f"freqs_mhz has {freqs.numel()} entries but n_chan={n_chan}"
            )
        log_nu = torch.log(freqs / ref_freq_mhz)   # (n_chan,)
        self.register_buffer("log_nu", log_nu)

        # ── Per-channel 1D profile encoder (shared weights across channels) ──
        # Processes (B*n_chan, 1, n_phase) in one batched pass.
        # 512 → MaxPool4 → 128 → AdaptiveAvgPool8 → 8 → flatten 256 → d_token
        self.profile_enc = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),                         # 512 → 128
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),                 # → 8
            nn.Flatten(),                            # 32 * 8 = 256
            nn.Linear(256, d_token),
            nn.ReLU(),
        )

        # ── Token projection ─────────────────────────────────────────────
        # [profile_feat (d_token) | log(ν/ν_ref) | flux | centroid] → d_model
        self.token_proj = nn.Sequential(
            nn.Linear(d_token + 3, d_model),
            nn.ReLU(),
        )

        # ── Transformer encoder: self-attention across n_chan tokens ──────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_attn_layers)

        # ── Output projection → 128-dim (matches MultiBranchEncoder) ─────
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, **overrides) -> "ChannelSetEncoder":
        overrides.setdefault("mask_dead_tokens", cfg.dead_channels is not None)
        return cls(
            n_chan       = cfg.n_chan,
            n_phase      = cfg.n_bins,
            freqs_mhz    = cfg.freqs_mhz,
            ref_freq_mhz = cfg.ref_freq_mhz,
            **overrides,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, n_chan*n_phase + n_chan + n_chan)
            Flat vector in the same order as MultiBranchEncoder:
            portrait pixels | flux_per_channel | centroids.
        Returns : (B, 128)
        """
        B     = x.shape[0]
        N     = self.n_chan
        P     = self.n_phase
        split = N * P

        portrait  = x[:, :split].view(B, N, P)            # (B, N, P)
        flux      = x[:, split      : split + N]           # (B, N)
        centroids = x[:, split + N  : split + 2 * N]       # (B, N)

        # ── 1D profile encoding: all channels in one batched Conv1d pass ─
        prof_in   = portrait.reshape(B * N, 1, P)          # (B*N, 1, P)
        prof_feat = self.profile_enc(prof_in)              # (B*N, d_token)
        prof_feat = prof_feat.view(B, N, -1)               # (B, N, d_token)

        # ── Assemble tokens with side features ───────────────────────────
        log_nu_exp = self.log_nu.view(1, N, 1).expand(B, N, 1)   # (B, N, 1)
        flux_exp   = flux.unsqueeze(-1)                           # (B, N, 1)
        cent_exp   = centroids.unsqueeze(-1)                      # (B, N, 1)

        tokens = torch.cat([prof_feat, log_nu_exp, flux_exp, cent_exp], dim=-1)
        tokens = self.token_proj(tokens)                   # (B, N, d_model)

        # ── Dead channels: drop them from attention and from the pool ────
        # A dead channel's row is exactly zero; every live row carries noise.
        # The mask comes from the ROW, not from flux == 0 — on real data live
        # channels reach +0.0002 and even negative flux. Batches with no dead
        # rows fall through to the unmasked path below.
        if getattr(self, "mask_dead_tokens", False):
            valid = portrait.abs().amax(dim=-1) > 0                    # (B, N)
            if not bool(valid.all()):
                valid = valid | ~valid.any(dim=1, keepdim=True)        # all-dead guard
                tokens = self.transformer(tokens, src_key_padding_mask=~valid)
                w = valid.unsqueeze(-1).to(tokens.dtype)               # (B, N, 1)
                embedding = (tokens * w).sum(dim=1) / w.sum(dim=1)     # masked mean
                return self.out_proj(embedding)

        # ── Self-attention across channels ───────────────────────────────
        tokens = self.transformer(tokens)                  # (B, N, d_model)

        # ── Mean pool across channels → fixed-size embedding ─────────────
        embedding = tokens.mean(dim=1)                     # (B, d_model)

        return self.out_proj(embedding)                    # (B, 128)


ENCODERS = {
    "multibranch": MultiBranchEncoder,
    "channelset":  ChannelSetEncoder,
}


def build_encoder(name: str, cfg, **overrides) -> nn.Module:
    if name not in ENCODERS:
        raise SystemExit(
            f"Unknown encoder '{name}'. Choose from: {', '.join(sorted(ENCODERS))}"
        )
    return ENCODERS[name].from_config(cfg, **overrides)


def encoder_param_count(encoder: nn.Module) -> int:
    return sum(p.numel() for p in encoder.parameters())


if __name__ == "__main__":
    print("This module is intended for import and use in other scripts, not direct execution.")
    exit(0)

