"""
pulsar_config.py
----------------
Config-driven parameter management for the SBI pulsar pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml


# --------------------------------------------------
# Parameter specification
# --------------------------------------------------
@dataclass
class ParamSpec:
    name:  str
    low:   float
    high:  float
    log:   bool = False
    label: Optional[str] = None
    unit:  str = ""
    # How to present this parameter in plots, converting FROM theta space:
    #   None           theta value as-is
    #   "exp"          np.exp(theta)      — the default when log: true
    #   "phase_to_us"  theta * period_s * 1e6, for phase-unit params like ToA
    # Purely cosmetic: never affects sampling, training or the theta vector.
    display: Optional[str] = None

    _DISPLAY_KINDS = (None, "exp", "phase_to_us")

    def __post_init__(self):
        if self.high <= self.low:
            raise ValueError(f"Parameter '{self.name}': high ({self.high}) must exceed low ({self.low})")
        if self.log and self.low <= 0:
            raise ValueError(f"Parameter '{self.name}': log=True requires low > 0 (got {self.low})")
        if self.display is None and self.log:
            self.display = "exp"
        if self.display not in ParamSpec._DISPLAY_KINDS:
            raise ValueError(
                f"Parameter '{self.name}': display '{self.display}' must be one of "
                f"{[k for k in ParamSpec._DISPLAY_KINDS]}"
            )

    @property
    def theta_name(self) -> str:
        """Name as it appears in the theta vector."""
        return f"log_{self.name}" if self.log else self.name

    @property
    def theta_low(self) -> float:
        return float(np.log(self.low)) if self.log else float(self.low)

    @property
    def theta_high(self) -> float:
        return float(np.log(self.high)) if self.log else float(self.high)

    def draw(self) -> float:
        """Draw one sample in LINEAR units (the simulator's units)."""
        if self.log:
            return float(np.exp(np.random.uniform(np.log(self.low), np.log(self.high))))
        return float(np.random.uniform(self.low, self.high))

    def to_theta(self, value: float) -> float:
        """Convert a linear-units value to its theta representation."""
        return float(np.log(value)) if self.log else float(value)


# --------------------------------------------------
# Component specification
# --------------------------------------------------
@dataclass
class ComponentSpec:
    """One von Mises profile component.

    Fields typed Union[float, str] accept either a literal number or a
    '$param_name' reference to a sampled inferred/nuisance parameter.
    """
    name:                  str
    amplitude_ref:         Union[float, str]
    phase_center_ref:      float
    concentration_ref:     Union[float, str]
    spectral_index_amp:    Union[float, str] = 0.0
    spectral_index_width:  Union[float, str] = 0.0
    spectral_index_center: Union[float, str] = 0.0
    sigma_broadening_bins: Union[float, str] = 0.0
    phase_offset_param:    Optional[str] = "toa"   # added to phase_center_ref; None to disable

    REF_FIELDS = (
        "amplitude_ref", "concentration_ref", "spectral_index_amp",
        "spectral_index_width", "spectral_index_center", "sigma_broadening_bins",
    )


# --------------------------------------------------
# Full configuration
# --------------------------------------------------
@dataclass
class PulsarConfig:
    path:         str
    pulsar:       Dict[str, Any]
    observatory:  Dict[str, Any]
    instrument:   Dict[str, Any]
    simulation:   Dict[str, Any]
    scattering:   Dict[str, Any]
    inferred:     List[ParamSpec]
    nuisance:     List[ParamSpec]
    components:   List[ComponentSpec]
    reference_component: str
    raw:          Dict[str, Any] = field(default_factory=dict)
    text:         str = ""

    # ---------- pulsar ----------
    @property
    def name(self) -> str:
        return self.pulsar["name"]

    @property
    def period_ms(self) -> float:
        return float(self.pulsar["period_ms"])

    @property
    def period_s(self) -> float:
        return self.period_ms / 1000.0

    @property
    def f0_par_hz(self) -> float:
        if "f0_par_hz" in self.pulsar:
            return float(self.pulsar["f0_par_hz"])
        return 1000.0 / self.period_ms

    @property
    def dm_header(self) -> float:
        """DM of the reference epoch used to derive the SBI templates (was SBI_DM_REF)."""
        return float(self.pulsar["dm_header"])

    @property
    def dm_conv(self) -> float:
        """Reference DM used in conventional timing (was CONV_DM_REF)."""
        return float(self.pulsar["dm_conv"])

    @property
    def raj(self) -> str:
        return str(self.pulsar["raj"])

    @property
    def decj(self) -> str:
        return str(self.pulsar["decj"])

    # ---------- observatory ----------
    # Post-inference only (timeseries.py). Deliberately excluded from the
    # fingerprint: changing station coordinates does NOT invalidate a trained
    # model, because Doppler correction never touches training.
    @property
    def obs_name(self) -> str:
        return str(self.observatory.get("name", "unknown"))

    @property
    def geodetic(self) -> Dict[str, float]:
        """lon_deg / lat_deg / height_m, for astropy EarthLocation.from_geodetic."""
        missing = [k for k in ("lon_deg", "lat_deg", "height_m") if k not in self.observatory]
        if missing:
            raise KeyError(
                f"config '{self.path}': observatory section is missing {missing} — "
                f"required for the Doppler DM correction in timeseries.py"
            )
        return {
            "lon_deg":  float(self.observatory["lon_deg"]),
            "lat_deg":  float(self.observatory["lat_deg"]),
            "height_m": float(self.observatory["height_m"]),
        }

    @property
    def n_chan(self) -> int:
        return int(self.instrument["n_freq_channels"])

    @property
    def n_bins(self) -> int:
        return int(self.instrument["n_phase_bins"])

    @property
    def freq_low_mhz(self) -> float:
        return float(self.instrument["freq_low_mhz"])

    @property
    def freq_high_mhz(self) -> float:
        return float(self.instrument["freq_high_mhz"])

    @property
    def ref_freq_mhz(self) -> float:
        return float(self.instrument["reference_freq_mhz"])

    @property
    def sigma_bandpass(self) -> float:
        return float(self.instrument.get("sigma_bandpass", 0.1))

    @property
    def tsys_scatter(self) -> float:
        """sd of ln(per-channel noise level), the system-temperature term (Stage S6b).

        Unlike sigma_bandpass, which multiplies signal AND noise (so it cannot
        change a channel's S/N), this multiplies the noise only. 0 (default) =>
        no draw, unchanged fingerprint.
        """
        return float(self.instrument.get("tsys_scatter", 0.0))

    @property
    def freq_tol_mhz(self) -> float:
        """Tolerance on each band edge when validating a real archive."""
        return float(self.instrument.get("freq_tol_mhz", 2.0))

    @property
    def centroid_mode(self) -> str:
        return str(self.instrument.get("centroid_mode", "linear"))

    @property
    def freqs_mhz(self) -> np.ndarray:
        explicit = self.instrument.get("channel_freqs_mhz")
        if explicit is not None:
            return np.asarray(explicit, dtype=float)
        return np.linspace(self.freq_low_mhz, self.freq_high_mhz, self.n_chan)

    @property
    def has_explicit_freqs(self) -> bool:
        return self.instrument.get("channel_freqs_mhz") is not None

    @cached_property
    def dead_channels(self) -> Optional[Dict[str, Any]]:
        block = self.raw.get("dead_channels")
        if block is None:
            return None
        spec, errs = _parse_dead_channels(block, self.freqs_mhz)
        if errs:
            raise ValueError("Invalid dead_channels block:\n  - " + "\n  - ".join(errs))
        return spec

    @cached_property
    def scintillation(self) -> Optional[Dict[str, Any]]:
        block = self.raw.get("scintillation")
        if block is None:
            return None
        spec, errs = _parse_scintillation(block)
        if errs:
            raise ValueError("Invalid scintillation block:\n  - " + "\n  - ".join(errs))
        return None if spec["model"] == "none" else spec

    @property
    def x_dim(self) -> int:
        """Feature-vector width: portrait.flatten() | flux_per_channel | centroids."""
        return self.n_chan * self.n_bins + 2 * self.n_chan

    # ---------- simulation ----------
    # Defaults below MUST equal the values the hardcoded pipeline used, or the
    # bit-for-bit reproduction check in verify_refactor.py fails.
    @property
    def epoch_mjd(self) -> float:
        return float(self.simulation.get("epoch_mjd", 59500))

    @property
    def add_noise(self) -> bool:
        return bool(self.simulation.get("add_noise", True))

    @property
    def dm_smearing(self) -> bool:
        return bool(self.simulation.get("dm_smearing", False))

    @property
    def noise_model(self) -> str:
        return str(self.simulation.get("noise_model", "avg_peak"))

    @property
    def npol(self) -> int:
        return int(self.simulation.get("npol", 2))

    @property
    def t_obs_s(self) -> float:
        return float(self.simulation.get("t_obs_s", 30.0))

    # DM and S/N are structural arguments to generate_portrait() rather than
    # component fields, so unlike components they cannot use a '$name' ref.
    # These name which sampled parameter drives each, so a config is free to
    # rename them (or drive DM from a nuisance parameter) without the simulator
    # hardcoding the strings. _validate() checks both resolve.
    @property
    def dm_param(self) -> str:
        return str(self.simulation.get("dm_param", "DM"))

    @property
    def snr_param(self) -> str:
        return str(self.simulation.get("snr_param", "snr"))

    @property
    def param_names(self) -> List[str]:
        """Inferred parameter names as they appear in theta."""
        return [p.theta_name for p in self.inferred]

    @property
    def param_labels(self) -> List[str]:
        return [(p.label or p.theta_name) for p in self.inferred]

    @property
    def prior_low(self) -> np.ndarray:
        return np.array([p.theta_low for p in self.inferred], dtype=np.float32)

    @property
    def prior_high(self) -> np.ndarray:
        return np.array([p.theta_high for p in self.inferred], dtype=np.float32)

    @property
    def reference_phase(self) -> float:
        """Phase centre of the reference (brightest) component, in [0,1)."""
        for c in self.components:
            if c.name == self.reference_component:
                return float(c.phase_center_ref)
        raise KeyError(f"reference_component '{self.reference_component}' not found in components")

    @property
    def reference_phase_bin(self) -> int:
        """Sentinel phase bin for dead/flagged channels (was MAIN_PHASE_BIN)."""
        return int(self.reference_phase * self.n_bins)

    # ---------- fingerprint ----------
    @property
    def fingerprint(self) -> str:
        payload = {
            "pulsar": {
                "name":      self.pulsar.get("name"),
                "period_ms": float(self.pulsar["period_ms"]),
            },
            "instrument": {
                "n_freq_channels":    self.n_chan,
                "n_phase_bins":       self.n_bins,
                "freq_low_mhz":       self.freq_low_mhz,
                "freq_high_mhz":      self.freq_high_mhz,
                "reference_freq_mhz": self.ref_freq_mhz,
                "sigma_bandpass":     self.sigma_bandpass,
                # The channel grid itself, so switching between explicit centres
                # and a linspace changes the fingerprint — it changes where the
                # simulated DM sweep lands, so the model is not interchangeable.
                "channel_freqs_mhz":  [float(v) for v in self.freqs_mhz],
                # Changes the meaning of the centroid block in x, so a model
                # trained under one mode must never be used with the other.
                "centroid_mode":      self.centroid_mode,
            },
            "simulation": {
                "epoch_mjd":   self.epoch_mjd,
                "add_noise":   self.add_noise,
                "dm_smearing": self.dm_smearing,
                "noise_model": self.noise_model,
                "npol":        self.npol,
                "t_obs_s":     self.t_obs_s,
                "dm_param":    self.dm_param,
                "snr_param":   self.snr_param,
            },
            # Order matters for inferred (it IS the theta ordering) and for
            # nuisance (it fixes the RNG draw sequence), so these stay lists.
            "inferred": [[p.name, p.low, p.high, p.log] for p in self.inferred],
            "nuisance": [[p.name, p.low, p.high, p.log] for p in self.nuisance],
            "components": [
                [c.name, c.amplitude_ref, c.phase_center_ref, c.concentration_ref,
                 c.spectral_index_amp, c.spectral_index_width, c.spectral_index_center,
                 c.sigma_broadening_bins, c.phase_offset_param]
                for c in self.components
            ],
            "reference_component": self.reference_component,
            "scattering": self.scattering,
        }
        # Added ONLY when present, so every config without the block keeps the
        # fingerprint it had before dead-channel injection existed. The mask
        # model changes what the network learned, so it must invalidate a dataset
        # when it changes. Site order is included because it is draw order.
        dc = self.dead_channels
        if dc is not None:
            payload["dead_channels"] = {
                "site_channels": [[s["mhz"], s["p"]] for s in dc["site_channels"]],
                "bad_day_sigma": dc["bad_day_sigma"],
                "generic":       dc["generic"],
                "max_frac":      dc["max_frac"],
                "wipeout":       dc["wipeout"],
            }
        # Same rule for scintillation (Stage S): only when present, so configs
        # without the block keep their fingerprints. Values are recorded as
        # written (a literal or a '$param' name); the referenced priors are
        # already in the inferred/nuisance lists above.
        sc = self.scintillation
        if sc is not None:
            payload["scintillation"] = {k: sc[k] for k in ("model",) + _SCINT_VALUE_FIELDS
                                        + ("tie_scattering",)}
        if self.tsys_scatter > 0:
            payload["instrument"]["tsys_scatter"] = self.tsys_scatter
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    # ---------- sampling ----------
    def sample(self) -> Dict[str, float]:
        """Draw all inferred + nuisance parameters. Values are in LINEAR units."""
        out = {}
        for p in self.inferred:
            out[p.name] = p.draw()
        for p in self.nuisance:
            out[p.name] = p.draw()
        return out

    def theta_vector(self, sampled: Dict[str, float]) -> List[float]:
        """Ordered inferred values, converted to theta representation (log applied)."""
        return [p.to_theta(sampled[p.name]) for p in self.inferred]

    # ---------- display / plotting ----------
    def display_specs(self) -> List[Dict[str, Any]]:
        p0s = self.period_s
        out = []
        for p in self.inferred:
            if p.display == "exp":
                fn = np.exp
            elif p.display == "phase_to_us":
                fn = lambda x, _s=p0s * 1e6: np.asarray(x) * _s
            else:
                fn = lambda x: np.asarray(x) if not np.isscalar(x) else x

            out.append({
                "name":  p.theta_name,
                "label": p.label or p.theta_name,
                "unit":  p.unit,
                "fn":    fn,
                "low":   float(fn(p.theta_low)),
                "high":  float(fn(p.theta_high)),
            })
        return out

    # ---------- component construction ----------
    def _resolve(self, value: Union[float, str], sampled: Dict[str, float], ctx: str) -> float:
        """Resolve a literal or '$param' reference against the sampled draws."""
        if isinstance(value, str):
            if not value.startswith("$"):
                raise ValueError(f"{ctx}: string value '{value}' must start with '$' to reference a parameter")
            key = value[1:]
            if key not in sampled:
                raise KeyError(f"{ctx}: '${key}' is not a declared inferred or nuisance parameter")
            return float(sampled[key])
        return float(value)

    def build_component_kwargs(self, sampled: Dict[str, float]) -> List[Dict[str, Any]]:
        """Return ProfileComponent kwargs dicts with all $refs resolved."""
        out = []
        for c in self.components:
            ctx = f"component '{c.name}'"
            kw: Dict[str, Any] = {"name": c.name}

            for f in ComponentSpec.REF_FIELDS:
                kw[f] = self._resolve(getattr(c, f), sampled, ctx)

            phase = float(c.phase_center_ref)
            if c.phase_offset_param:
                if c.phase_offset_param not in sampled:
                    raise KeyError(
                        f"{ctx}: phase_offset_param '{c.phase_offset_param}' is not a declared parameter"
                    )
                phase += float(sampled[c.phase_offset_param])
            kw["phase_center_ref"] = phase

            out.append(kw)
        return out

    # ---------- scattering ----------
    def scattering_values(self, sampled: Dict[str, float]) -> Dict[str, float]:
        """Resolve scattering tau (ms) and spectral index."""
        def _val(entry, default):
            if entry is None:
                return default
            if isinstance(entry, dict):
                if entry.get("type") == "free":
                    return float(sampled[entry["param"]])
                return float(entry.get("value", default))
            return float(entry)

        sc = self.scintillation
        if sc is not None and sc["tie_scattering"]:
            # One screen: tau_d = 1 / (2 pi dnu_d) at the reference frequency,
            # scaling as nu^-bandwidth_index. _validate() rejects a scattering
            # block that would otherwise set tau itself.
            from .scintillation import scattering_tau_ms
            v = self.scintillation_values(sampled)
            return {"tau_ms": scattering_tau_ms(v["bandwidth_mhz"]),
                    "spectral_index": -v["bandwidth_index"]}
        return {
            "tau_ms":         _val(self.scattering.get("tau_ms"), 0.0),
            "spectral_index": _val(self.scattering.get("spectral_index"), -4.4),
        }

    # ---------- scintillation ----------
    def scintillation_values(self, sampled: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Resolve the scintillation block against one draw; None when there is none.

        t_obs_s falls back to simulation.t_obs_s when the block does not set it.
        """
        sc = self.scintillation
        if sc is None:
            return None
        out = {f: self._resolve(sc[f], sampled, "scintillation")
               for f in _SCINT_VALUE_FIELDS if sc[f] is not None}
        out.setdefault("t_obs_s", self.t_obs_s)
        return out

    # ---------- summary ----------
    def summary(self) -> str:
        lines = [
            "=" * 62,
            f"  Config: {os.path.basename(self.path)}",
            "=" * 62,
            f"  Pulsar        : {self.name}   P0 = {self.period_ms:.6f} ms",
            f"  Observatory   : {self.observatory.get('name', '?')}  "
            f"(lon {self.observatory.get('lon_deg')}, lat {self.observatory.get('lat_deg')})",
            f"  Instrument    : {self.n_chan} ch x {self.n_bins} bins,  "
            f"{self.freq_low_mhz:.3f}-{self.freq_high_mhz:.3f} MHz",
            f"  Reference freq: {self.ref_freq_mhz:.3f} MHz",
            f"  Components    : {len(self.components)}  "
            f"({', '.join(c.name for c in self.components)})",
            f"  Reference comp: {self.reference_component}  "
            f"(phase {self.reference_phase:.6f} -> bin {self.reference_phase_bin})",
            f"  Noise model   : {self.noise_model}  "
            f"(noise {'on' if self.add_noise else 'off'}, "
            f"smearing {'on' if self.dm_smearing else 'off'})",
            f"  Centroid mode : {self.centroid_mode}",
            f"  Dead channels : {_dead_channels_summary(self.dead_channels)}",
            f"  Scintillation : {_scintillation_summary(self.scintillation)}",
            f"  Channel gains : bandpass sigma {self.sigma_bandpass:g} (signal+noise), "
            f"Tsys scatter {self.tsys_scatter:g} (noise only)",
            f"  Feature width : {self.x_dim}  "
            f"({self.n_chan}x{self.n_bins} portrait + {self.n_chan} flux + {self.n_chan} centroids)",
            f"  Fingerprint   : {self.fingerprint}",
            "-" * 62,
            f"  Inferred ({len(self.inferred)}):",
        ]
        for p in self.inferred:
            lo, hi = (p.theta_low, p.theta_high)
            tag = "  [log]" if p.log else ""
            lines.append(f"    {p.theta_name:<16} U[{lo:>9.4f}, {hi:>9.4f}]{tag}")
        lines.append(f"  Nuisance ({len(self.nuisance)}):")
        for p in self.nuisance:
            lines.append(f"    {p.name:<16} U[{p.low:>9.4f}, {p.high:>9.4f}]")
        lines.append("=" * 62)
        return "\n".join(lines)


# --------------------------------------------------
# Loading and validation
# --------------------------------------------------
def _parse_params(raw: Optional[List[Dict]], section: str) -> List[ParamSpec]:
    if not raw:
        return []
    out = []
    for entry in raw:
        if "name" not in entry:
            raise ValueError(f"{section}: every entry needs a 'name'")
        known = {"name", "low", "high", "log", "label", "unit", "display"}
        unknown = set(entry) - known
        if unknown:
            raise ValueError(
                f"{section}: parameter '{entry['name']}' has unknown field(s) "
                f"{sorted(unknown)}. Known: {sorted(known)}"
            )
        out.append(ParamSpec(
            name    = entry["name"],
            low     = float(entry["low"]),
            high    = float(entry["high"]),
            log     = bool(entry.get("log", False)),
            label   = entry.get("label"),
            unit    = entry.get("unit", ""),
            display = entry.get("display"),
        ))
    return out


def _parse_components(raw: List[Dict]) -> List[ComponentSpec]:
    if not raw:
        raise ValueError("config must declare at least one component")
    out = []
    for entry in raw:
        known = {f.name for f in ComponentSpec.__dataclass_fields__.values()}
        unknown = set(entry) - known
        if unknown:
            raise ValueError(f"component '{entry.get('name','?')}': unknown field(s) {sorted(unknown)}")
        out.append(ComponentSpec(**entry))
    return out


# --------------------------------------------------
# Dead-channel (RFI) injection model
# --------------------------------------------------
_DEAD_KEYS         = {"site_channels", "bad_day_sigma", "generic", "max_frac", "wipeout"}
_DEAD_GENERIC_KEYS = {"p_any", "n_mean", "modes", "block_width_mhz"}
_DEAD_MODES        = ("random", "block")

# Scintillation block (Stage S). Each value field is a literal or a '$param'.
_SCINT_VALUE_FIELDS = ("bandwidth_mhz", "bandwidth_index", "timescale_s", "timescale_index", "t_obs_s")
_SCINT_KEYS         = {"model", "tie_scattering", *_SCINT_VALUE_FIELDS}
_SCINT_MODELS       = ("none", "synthesis")
_SCINT_DEFAULTS     = {"bandwidth_index": 4.4, "timescale_index": 1.2, "t_obs_s": None}
_SCINT_POSITIVE     = ("bandwidth_mhz", "timescale_s", "t_obs_s")


def _expit(t):
    """Logistic function, overflow-free. Same form used in _draw_dead_mask."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(t, dtype=float)))


def _solve_bad_day_logits(p: np.ndarray, sigma: float, n_nodes: int = 64) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return p
    if sigma == 0.0:
        return np.log(p / (1.0 - p))
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)   # weight exp(-x²/2)
    w = w / w.sum()
    lo = np.full(p.shape, -60.0)
    hi = np.full(p.shape,  60.0)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        too_high = (_expit(mid[:, None] + sigma * x[None, :]) * w).sum(axis=1) > p
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return 0.5 * (lo + hi)


def _parse_dead_channels(block: Any, freqs: np.ndarray):
    errs: List[str] = []
    where = "dead_channels"
    if not isinstance(block, dict):
        return None, [f"{where}: must be a mapping, got {type(block).__name__}"]
    unknown = set(block) - _DEAD_KEYS
    if unknown:
        errs.append(f"{where}: unknown key(s) {sorted(unknown)}. Known: {sorted(_DEAD_KEYS)}")

    freqs = np.asarray(freqs, dtype=float)
    n = freqs.size

    # ---- site channels: MHz-addressed, snapped to the nearest channel ----
    site_in = block.get("site_channels") or []
    if not isinstance(site_in, list):
        errs.append(f"{where}.site_channels: must be a list of {{mhz, p}}")
        site_in = []
    site, idx = [], []
    for j, e in enumerate(site_in):
        w = f"{where}.site_channels[{j}]"
        if not isinstance(e, dict) or set(e) != {"mhz", "p"}:
            errs.append(f"{w}: must be {{mhz: <float>, p: <float>}}, got {e!r}")
            continue
        mhz, p = float(e["mhz"]), float(e["p"])
        i = int(np.argmin(np.abs(freqs - mhz)))
        gaps = [abs(freqs[i] - freqs[k]) for k in (i - 1, i + 1) if 0 <= k < n]
        tol = 0.5 * min(gaps) if gaps else np.inf
        if abs(freqs[i] - mhz) > tol:
            errs.append(
                f"{w}: {mhz} MHz is {abs(freqs[i] - mhz):.3f} MHz from the nearest "
                f"channel ({freqs[i]:.3f} MHz, idx {i}), more than half a channel "
                f"spacing ({tol:.3f} MHz). Was it copied from another telescope's grid?")
            continue
        if not (0.0 < p < 1.0):
            errs.append(
                f"{w}: p = {p} must lie strictly inside (0, 1). p = 1 removes "
                f"'channel alive' from training altogether, so any real epoch with "
                f"it alive becomes out-of-distribution — give the measured "
                f"fraction instead (e.g. 0.9995).")
            continue
        if i in idx:
            errs.append(f"{w}: {mhz} MHz resolves to channel {i}, which is already listed")
            continue
        site.append({"mhz": mhz, "p": p})
        idx.append(i)

    sigma = float(block.get("bad_day_sigma", 0.0))
    if sigma < 0.0:
        errs.append(f"{where}.bad_day_sigma: {sigma} must be >= 0")
        sigma = 0.0

    # ---- generic: support beyond the measured channels ----
    g_in = block.get("generic") or {}
    if not isinstance(g_in, dict):
        errs.append(f"{where}.generic: must be a mapping")
        g_in = {}
    unknown = set(g_in) - _DEAD_GENERIC_KEYS
    if unknown:
        errs.append(f"{where}.generic: unknown key(s) {sorted(unknown)}. "
                    f"Known: {sorted(_DEAD_GENERIC_KEYS)}")
    p_any  = float(g_in.get("p_any", 0.1))
    n_mean = float(g_in.get("n_mean", 2.0))
    modes  = g_in.get("modes", {"random": 0.5, "block": 0.5})
    bw     = g_in.get("block_width_mhz", [2.0, 10.0])
    if not 0.0 <= p_any <= 1.0:
        errs.append(f"{where}.generic.p_any: {p_any} outside [0, 1]")
    if n_mean < 1.0:
        errs.append(f"{where}.generic.n_mean: {n_mean} must be >= 1 (count is 1 + Poisson(n_mean - 1))")
    if not isinstance(modes, dict) or set(modes) - set(_DEAD_MODES):
        errs.append(f"{where}.generic.modes: must map a subset of {list(_DEAD_MODES)} to weights")
        modes = {"random": 1.0}
    modes = {m: float(modes.get(m, 0.0)) for m in _DEAD_MODES}
    if any(v < 0 for v in modes.values()) or (p_any > 0 and sum(modes.values()) <= 0):
        errs.append(f"{where}.generic.modes: weights must be >= 0 and not all zero")
    if not (isinstance(bw, (list, tuple)) and len(bw) == 2 and 0 < float(bw[0]) <= float(bw[1])):
        errs.append(f"{where}.generic.block_width_mhz: must be [lo, hi] with 0 < lo <= hi")
        bw = [2.0, 10.0]

    # ---- cap on ordinary (non-wipeout) dead channels ----
    max_frac = float(block.get("max_frac", 0.25))
    max_dead = int(np.floor(max_frac * n))
    if not 0.0 < max_frac < 1.0 or max_dead < 1:
        errs.append(f"{where}.max_frac: {max_frac} must be in (0, 1) and allow at least "
                    f"one channel ({n} channels)")
    elif len(site) > max_dead:
        errs.append(f"{where}: {len(site)} site_channels exceed max_frac x n_chan = "
                    f"{max_dead}; raise max_frac")

    # ---- wipeout: rare near-total band loss ----
    wo_in = block.get("wipeout") or {}
    if not isinstance(wo_in, dict) or set(wo_in) - {"p", "frac"}:
        errs.append(f"{where}.wipeout: must be {{p: <float>, frac: [lo, hi]}}")
        wo_in = {}
    wo_p    = float(wo_in.get("p", 0.0))
    wo_frac = wo_in.get("frac", [0.3, 0.7])
    if not 0.0 <= wo_p <= 1.0:
        errs.append(f"{where}.wipeout.p: {wo_p} outside [0, 1]")
    if not (isinstance(wo_frac, (list, tuple)) and len(wo_frac) == 2
            and 0 < float(wo_frac[0]) <= float(wo_frac[1]) < 1):
        errs.append(f"{where}.wipeout.frac: must be [lo, hi] with 0 < lo <= hi < 1")
        wo_frac = [0.3, 0.7]

    if errs:
        return None, errs
    p_arr = np.array([s["p"] for s in site], dtype=float)
    return {
        "site_channels": site,                       # as declared (fingerprint, summary)
        "site_idx":      np.array(idx, dtype=int),
        "site_p":        p_arr,
        "site_logit":    _solve_bad_day_logits(p_arr, sigma),
        "bad_day_sigma": sigma,
        "generic": {"p_any": p_any, "n_mean": n_mean, "modes": modes,
                    "block_width_mhz": [float(bw[0]), float(bw[1])]},
        "max_frac":      max_frac,
        "max_dead":      max_dead,
        "wipeout":       {"p": wo_p, "frac": [float(wo_frac[0]), float(wo_frac[1])]},
    }, []


def _dead_channels_summary(dc: Optional[Dict[str, Any]]) -> str:
    if dc is None:
        return "off"
    g, wo = dc["generic"], dc["wipeout"]
    return (f"{len(dc['site_channels'])} site ch (bad_day_sigma {dc['bad_day_sigma']:g}, "
            f"mean {dc['site_p'].sum():.2f}/epoch), generic p_any {g['p_any']:g}, "
            f"wipeout p {wo['p']:g}, cap {dc['max_dead']}")


def _parse_scintillation(block: Any):
    errs: List[str] = []
    where = "scintillation"
    if not isinstance(block, dict):
        return None, [f"{where}: must be a mapping, got {type(block).__name__}"]
    unknown = set(block) - _SCINT_KEYS
    if unknown:
        errs.append(f"{where}: unknown key(s) {sorted(unknown)}. Known: {sorted(_SCINT_KEYS)}")
    model = block.get("model", "synthesis")
    if model not in _SCINT_MODELS:
        errs.append(f"{where}.model: '{model}' must be one of {list(_SCINT_MODELS)}")
    tie = block.get("tie_scattering", False)
    if not isinstance(tie, bool):
        errs.append(f"{where}.tie_scattering: must be true or false, got {tie!r}")
        tie = False
    spec: Dict[str, Any] = {"model": model, "tie_scattering": tie}
    for f in _SCINT_VALUE_FIELDS:
        v = block.get(f, _SCINT_DEFAULTS.get(f))
        if v is None:
            if f in ("bandwidth_mhz", "timescale_s") and model != "none":
                errs.append(f"{where}.{f}: required (a literal or a '$param' reference)")
            spec[f] = None
            continue
        if isinstance(v, str):
            if not v.startswith("$"):
                errs.append(f"{where}.{f}: '{v}' must be a number or start with '$'")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            errs.append(f"{where}.{f}: {v!r} must be a finite number or a '$param' reference")
        elif f in _SCINT_POSITIVE and v <= 0:
            errs.append(f"{where}.{f}: {v} must be > 0")
        else:
            v = float(v)
        spec[f] = v
    return (None if errs else spec), errs


def _scintillation_summary(sc: Optional[Dict[str, Any]]) -> str:
    if sc is None:
        return "off"
    t_obs = sc["t_obs_s"] if sc["t_obs_s"] is not None else "simulation.t_obs_s"
    return (f"{sc['model']}, dnu_d {sc['bandwidth_mhz']} MHz (nu^{sc['bandwidth_index']}), "
            f"dt_d {sc['timescale_s']} s (nu^{sc['timescale_index']}), t_obs {t_obs}, "
            f"tie_scattering {'on' if sc['tie_scattering'] else 'off'}")


def _validate(cfg: PulsarConfig) -> None:
    """Fail loudly on the mistakes that would otherwise corrupt a training run."""
    errs: List[str] = []

    # Duplicate parameter names
    all_names = [p.name for p in cfg.inferred] + [p.name for p in cfg.nuisance]
    dupes = {n for n in all_names if all_names.count(n) > 1}
    if dupes:
        errs.append(f"duplicate parameter name(s): {sorted(dupes)}")

    declared = set(all_names)

    # Every $ref must resolve
    for c in cfg.components:
        for f in ComponentSpec.REF_FIELDS:
            v = getattr(c, f)
            if isinstance(v, str):
                if not v.startswith("$"):
                    errs.append(f"component '{c.name}'.{f}: '{v}' must start with '$'")
                elif v[1:] not in declared:
                    errs.append(f"component '{c.name}'.{f}: '${v[1:]}' is not declared")
        if c.phase_offset_param and c.phase_offset_param not in declared:
            errs.append(
                f"component '{c.name}': phase_offset_param '{c.phase_offset_param}' is not declared"
            )
        if not (0.0 <= c.phase_center_ref < 1.0):
            errs.append(f"component '{c.name}': phase_center_ref {c.phase_center_ref} outside [0,1)")

    # Reference component must exist
    if cfg.reference_component not in {c.name for c in cfg.components}:
        errs.append(f"reference_component '{cfg.reference_component}' is not a declared component")

    # Instrument sanity
    if cfg.freq_high_mhz <= cfg.freq_low_mhz:
        errs.append(f"freq_high_mhz ({cfg.freq_high_mhz}) must exceed freq_low_mhz ({cfg.freq_low_mhz})")
    # Explicit channel centres, when given, must be self-consistent
    if cfg.has_explicit_freqs:
        fr = cfg.freqs_mhz
        if fr.size != cfg.n_chan:
            errs.append(
                f"instrument.channel_freqs_mhz has {fr.size} entries but "
                f"n_freq_channels is {cfg.n_chan}"
            )
        if fr.size > 1 and np.any(np.diff(fr) <= 0):
            errs.append(
                "instrument.channel_freqs_mhz must be strictly ascending — the "
                "simulator and real_archive_loader.py both use ascending channel "
                "order (row 0 = lowest frequency)"
            )
        if fr.size and (fr.min() < cfg.freq_low_mhz or fr.max() > cfg.freq_high_mhz):
            errs.append(
                f"instrument.channel_freqs_mhz spans {fr.min():.4f}-{fr.max():.4f} MHz, "
                f"outside the declared band [{cfg.freq_low_mhz}, {cfg.freq_high_mhz}]. "
                f"freq_low_mhz/freq_high_mhz are band EDGES and must enclose the "
                f"channel centres."
            )

    if not (cfg.freq_low_mhz <= cfg.ref_freq_mhz <= cfg.freq_high_mhz):
        errs.append(
            f"reference_freq_mhz ({cfg.ref_freq_mhz}) outside band "
            f"[{cfg.freq_low_mhz}, {cfg.freq_high_mhz}] — component amplitude_ref/concentration_ref "
            f"are defined AT this frequency, so this is almost certainly wrong"
        )

    if cfg.centroid_mode not in ("linear", "circular"):
        errs.append(
            f"instrument.centroid_mode '{cfg.centroid_mode}' must be 'linear' or "
            f"'circular'"
        )


    if cfg.centroid_mode == "linear" and cfg.inferred:
        try:
            _dm = next(p for p in cfg.inferred if p.name == cfg.dm_param)
            _k = 4.1488064239e6
            _sweep = (_k * abs(_dm.high) *
                      (1.0 / cfg.freqs_mhz.min() ** 2 - 1.0 / cfg.freqs_mhz.max() ** 2)
                      / cfg.period_ms)
            _toa = next((p for p in cfg.inferred if p.name == "toa"), None)
            _reach = _sweep + (abs(_toa.high) if _toa else 0.0)
            _lo = cfg.reference_phase - _reach
            _hi = cfg.reference_phase + _reach
            if _lo < 0.0 or _hi > 1.0:
                errs.append(
                    f"centroid_mode is 'linear' but the pulse reaches phase "
                    f"[{_lo:.3f}, {_hi:.3f}] once the DM sweep ({_sweep:.3f}) and the "
                    f"ToA prior are added to reference_phase {cfg.reference_phase:.3f}. "
                    f"That crosses the phase wrap, where the linear centroid is a "
                    f"linear statistic on a circular quantity: it becomes "
                    f"non-monotonic in DM and inverts the ToA response, destroying "
                    f"the DM/ToA discrimination the centroids exist for. Use "
                    f"centroid_mode: circular."
                )
        except StopIteration:
            pass

    # DM / SNR driver parameters must resolve
    for prop, key in (("dm_param", "dm_param"), ("snr_param", "snr_param")):
        pname = getattr(cfg, prop)
        if pname not in declared:
            errs.append(
                f"simulation.{key}: '{pname}' is not a declared inferred or "
                f"nuisance parameter (it drives generate_portrait, so it must exist)"
            )

    # Scattering free-param references
    for key in ("tau_ms", "spectral_index"):
        entry = cfg.scattering.get(key)
        if isinstance(entry, dict) and entry.get("type") == "free":
            if "param" not in entry:
                errs.append(f"scattering.{key}: type 'free' requires a 'param' field")
            elif entry["param"] not in declared:
                errs.append(f"scattering.{key}: param '{entry['param']}' is not declared")

    # Dead-channel injection model
    dead_block = cfg.raw.get("dead_channels")
    if dead_block is not None:
        errs.extend(_parse_dead_channels(dead_block, cfg.freqs_mhz)[1])
        if cfg.centroid_mode == "linear":
            errs.append(
                "dead_channels: requires instrument.centroid_mode: circular. In "
                "linear mode an all-zero channel gets centroid 0 instead of the "
                "sentinel, so it reads as a pulse ~reference_phase_bin bins from "
                "the real one (REFACTOR_PLAN.md R2b)."
            )

    if cfg.tsys_scatter < 0:
        errs.append(f"instrument.tsys_scatter: {cfg.tsys_scatter} must be >= 0")

    # Scintillation model (Stage S)
    scint_block = cfg.raw.get("scintillation")
    if scint_block is not None:
        spec, scint_errs = _parse_scintillation(scint_block)
        errs.extend(scint_errs)
        if spec is not None and spec["model"] != "none":
            specs = {p.name: p for p in list(cfg.inferred) + list(cfg.nuisance)}
            for f in _SCINT_VALUE_FIELDS:
                v = spec[f]
                if isinstance(v, str):
                    if v[1:] not in declared:
                        errs.append(f"scintillation.{f}: '{v}' is not a declared inferred or nuisance parameter")
                    elif f in _SCINT_POSITIVE and specs[v[1:]].low <= 0:
                        errs.append(f"scintillation.{f}: prior of '{v}' must be > 0 (low is {specs[v[1:]].low})")
            if spec["tie_scattering"]:
                tau = cfg.scattering.get("tau_ms")
                if isinstance(tau, dict) and (tau.get("type") == "free" or float(tau.get("value", 0.0)) != 0.0) \
                        or (tau is not None and not isinstance(tau, dict) and float(tau) != 0.0):
                    errs.append("scintillation.tie_scattering: the scattering block also sets tau_ms; "
                                "with the tie, tau comes from bandwidth_mhz. Remove tau_ms or set it to fixed 0.")

    if errs:
        raise ValueError(
            "Invalid config '%s':\n  - %s" % (cfg.path, "\n  - ".join(errs))
        )


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: str, _seen: Optional[List[str]] = None) -> Dict[str, Any]:
    path  = os.path.abspath(path)
    _seen = list(_seen or [])
    if path in _seen:
        chain = " -> ".join(os.path.basename(p) for p in _seen + [path])
        raise ValueError(f"Circular 'extends' in config chain: {chain}")
    _seen.append(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config '{path}' must be a YAML mapping at top level")

    parent = raw.pop("extends", None)
    if parent is None:
        return raw

    parent_path = parent if os.path.isabs(parent) else os.path.join(
        os.path.dirname(path), parent)
    if not os.path.isfile(parent_path):
        raise FileNotFoundError(
            f"Config '{os.path.basename(path)}' extends '{parent}', which was not "
            f"found at {parent_path} (paths are relative to the extending file)"
        )
    return _deep_merge(_load_raw(parent_path, _seen), raw)


@lru_cache(maxsize=16)
def load_config(path: str) -> PulsarConfig:
    """Load and validate a YAML config. Cached per path (safe across mp workers)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        own_text = fh.read()
    raw = _load_raw(path)

    # `text` is embedded verbatim in metadata.pt so a dataset directory stays
    # self-describing. With `extends:` the file on disk is only a fragment, so
    # embed the RESOLVED config — otherwise a copied dataset would reference a
    # base file that is not there.
    if "extends:" in own_text:
        text = (
            f"# Resolved config for {os.path.basename(path)}\n"
            f"# (this file uses 'extends:'; the merged result is recorded here so\n"
            f"#  the dataset stays self-describing without its base file)\n"
            + yaml.safe_dump(raw, sort_keys=False, default_flow_style=False)
        )
    else:
        text = own_text

    for section in ("pulsar", "instrument", "components"):
        if section not in raw:
            raise ValueError(f"Config '{path}' is missing required section '{section}'")

    cfg = PulsarConfig(
        path        = path,
        pulsar      = raw["pulsar"],
        observatory = raw.get("observatory", {}),
        instrument  = raw["instrument"],
        simulation  = raw.get("simulation", {}),
        scattering  = raw.get("scattering", {}),
        inferred    = _parse_params(raw.get("inferred"),  "inferred"),
        nuisance    = _parse_params(raw.get("nuisance"),  "nuisance"),
        components  = _parse_components(raw["components"]),
        reference_component = raw.get("reference_component",
                                      raw["components"][0]["name"]),
        raw         = raw,
        text        = text,   # embedded verbatim in metadata.pt — keeps a
                              # dataset directory self-describing after a move
    )
    _validate(cfg)
    return cfg


# --------------------------------------------------
# Default config resolution
# --------------------------------------------------
DEFAULT_CONFIG_ENV = "SBI_PULSAR_CONFIG"
def resolve_config_path(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get(DEFAULT_CONFIG_ENV)
    if env:
        return env
    raise ValueError(f"No config given: pass --config <path/to/file.yaml> "
                     f"or set ${DEFAULT_CONFIG_ENV}.")


def get_config(explicit: Optional[str] = None) -> PulsarConfig:
    return load_config(resolve_config_path(explicit))


def recorded_config_path(meta: Dict[str, Any], meta_dir: str,
                         explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    path = meta.get("config_path")
    if path and os.path.isfile(path):
        return path
    text = meta.get("config_yaml")
    if not text:
        raise FileNotFoundError(
            f"No --config given, and the metadata in '{meta_dir or '.'}' records "
            f"neither a usable config path ({path}) nor an embedded config. "
            f"Pass --config with the config this was generated/trained with.")
    local = os.path.abspath(os.path.join(meta_dir, "config_embedded.yaml"))
    if not (os.path.isfile(local) and open(local).read() == text):
        with open(local + ".tmp", "w") as fh:
            fh.write(text)
        os.replace(local + ".tmp", local)
    print(f"NOTE: config '{path}' not found; using the copy embedded in the "
          f"metadata, written to {local}")
    return local


# --------------------------------------------------
# Fingerprint enforcement
# --------------------------------------------------
def check_fingerprint(cfg: PulsarConfig,
                      stored: Optional[str],
                      what: str,
                      force: bool = False) -> None:
    if stored is None:
        print(f"WARNING: {what} carries no config fingerprint (predates this "
              f"field). Cannot verify it matches '{os.path.basename(cfg.path)}'.")
        return

    if stored == cfg.fingerprint:
        return

    msg = (
        f"Config mismatch against {what}.\n"
        f"    expected fingerprint : {stored}   (from {what})\n"
        f"    this config          : {cfg.fingerprint}   ({cfg.path})\n"
        f"  The instrument setup, priors, or components differ. Continuing would "
        f"mix incompatible simulations or train on the wrong feature layout."
    )
    if force:
        print(f"WARNING: {msg}\n  --force given; continuing anyway.")
        return
    raise SystemExit(f"ERROR: {msg}\n  Pass --force to override deliberately.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validate and summarise a pulsar config.")
    ap.add_argument("config", nargs="?", default=None, help="Path to YAML config")
    a = ap.parse_args()

    cfg = get_config(a.config)
    print(cfg.summary())

    print("\nOne sample draw:")
    s = cfg.sample()
    for k, v in s.items():
        print(f"  {k:<16} = {v:.6f}")
    print("\nTheta vector:")
    for n, v in zip(cfg.param_names, cfg.theta_vector(s)):
        print(f"  {n:<16} = {v:.6f}")
    print("\nComponent kwargs:")
    for kw in cfg.build_component_kwargs(s):
        print(f"  {kw}")
