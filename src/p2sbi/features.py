"""
features.py
-----------
THE single definition of the SBI feature vector's components.

    x = portrait.flatten() | flux_per_channel | centroids

Imported by BOTH the simulator (pulsarDMvary.py) and the real-data path
(real_archive_loader.py). That is the entire point of this module: the dead-channel sentinel and the portrait normalisation — were
cross-file contracts maintained by hand, each implemented twice, with nothing
detecting drift. Here they are implemented once.

Deliberately depends on numpy only, so the psrchive environment can import it
without pulling in torch or sbi.
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# TRAP 4 — normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_portrait(portrait: np.ndarray) -> np.ndarray:
    """Max-normalise. Simulator and archive loader MUST agree; they call this."""
    return portrait / (portrait.max() + 1e-12)


def flux_per_channel(portrait: np.ndarray) -> np.ndarray:
    """Mean flux in each channel."""
    return portrait.mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# TRAP 3 — per-channel pulse position
# ─────────────────────────────────────────────────────────────────────────────

def _linear_centroids(portrait: np.ndarray, sentinel_bin: float) -> np.ndarray:
    """Flux-weighted mean bin index, the pre-2026 definition.

    Kept so the I-LOFAR configuration and its trained 20M-sim model remain
    reproducible bit-for-bit. Do not use for new configurations: it treats a
    CIRCULAR quantity with a LINEAR statistic, so a profile whose pulse
    straddles phase 0 gets a centroid in the middle of the array, nowhere near
    the pulse. See centroid_mode: circular below.
    """
    n_bins = portrait.shape[1]
    phase_axis = np.arange(n_bins)
    c = (portrait * phase_axis).sum(axis=1) / (portrait.sum(axis=1) + 1e-12)
    # At low S/N noise drives the ratio out of range (values like -1463 were
    # seen on real dead channels). Replace with the reference bin so the network
    # meets the same value in training and inference.
    bad = (c < 0.0) | (c >= n_bins)
    c[bad] = float(sentinel_bin)
    return c


def _circular_centroids(portrait: np.ndarray, reference_bin: float,
                        dead_frac: float = 1e-3) -> np.ndarray:
    """Circular mean position, as a wrapped OFFSET from `reference_bin`.

    Two fixes over the linear version, both measurable at infinite S/N on a
    profile that straddles phase 0:

      * the first Fourier moment is the correct mean of a circular quantity, so
        the estimate tracks the pulse instead of the middle of the array;
      * returning a wrapped offset from the reference phase makes the feature
        continuous across the wrap AND centres it on zero.

    Why it matters: the centroid block exists to separate DM from ToA — DM
    appears as the slope of centroid against 1/nu^2, ToA as the intercept. With
    the linear definition on the German configuration the ToA response comes out
    at -2200 bins per unit phase against an expected +512 (wrong sign, 4x wrong
    size) and the DM response is not even monotonic. With this definition they
    are +511.9 and exactly linear.

    Dead channels: atan2 is bounded, so the catastrophic out-of-range values the
    linear version produced cannot occur. A channel carrying negligible flux
    still has an arbitrary phase, so it is set to offset 0 — the circular
    analogue of the old sentinel, which was reference_phase_bin, i.e. offset 0.
    """
    n_bins = portrait.shape[1]
    theta = 2.0 * np.pi * np.arange(n_bins) / n_bins
    w = np.clip(portrait, 0.0, None)           # negative noise carries no position
    s = (w * np.sin(theta)).sum(axis=1)
    c = (w * np.cos(theta)).sum(axis=1)

    ang = np.arctan2(s, c) % (2.0 * np.pi)
    pos = ang * n_bins / (2.0 * np.pi)
    off = (pos - reference_bin + n_bins / 2.0) % n_bins - n_bins / 2.0

    tot = w.sum(axis=1)
    dead = tot <= dead_frac * max(float(tot.max()), 1e-30)
    off[dead] = 0.0
    return off


def channel_centroids(portrait: np.ndarray, cfg) -> np.ndarray:
    """Per-channel pulse position, per the config's `centroid_mode`.

    linear   — flux-weighted mean bin, absolute, with the out-of-range sentinel.
               The pre-2026 behaviour; keeps existing datasets/models valid.
    circular — first-Fourier-moment position as a wrapped offset from
               reference_phase_bin. Correct for a circular quantity.
    """
    mode = cfg.centroid_mode
    if mode == "linear":
        return _linear_centroids(portrait, cfg.reference_phase_bin)
    if mode == "circular":
        return _circular_centroids(portrait, cfg.reference_phase_bin)
    raise ValueError(f"unknown centroid_mode '{mode}'")


def build_features(portrait: np.ndarray, cfg):
    """Normalise, then derive every feature. Returns (portrait, flux, centroids).

    The one call both the simulator and the archive loader make, so the feature
    definition cannot diverge between training and inference.
    """
    p = normalise_portrait(portrait)
    return p, flux_per_channel(p), channel_centroids(p, cfg)
