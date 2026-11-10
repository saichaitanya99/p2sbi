r"""
natastro_style.py — Shared matplotlib style for Nature Astronomy submission.

Uses matplotlib's built-in STIX fonts (Times-like, no LaTeX installation
needed).  Math symbols in axis labels still work via matplotlib's mathtext
renderer — just write $\delta$, $\gamma$, etc. inside dollar signs as usual.

Usage:
    from p2sbi.natastro_style import set_style, NAT_W_SINGLE, NAT_W_DOUBLE

    set_style()
    fig, ax = plt.subplots(figsize=(NAT_W_DOUBLE, 3.5))

Design rules:
  - No figure titles (caption carries that information)
  - STIX serif font via matplotlib mathtext (text.usetex always False)
  - Ticks inward on all 4 sides with minor ticks visible
  - No grid, no legend box border
  - savefig.dpi = 300 (Nature minimum for raster)
  - Figure widths follow Nature column sizes (88 mm / 180 mm)
"""

import matplotlib as mpl

# Nature Astronomy column widths in inches (88 mm / 180 mm)
NAT_W_SINGLE = 3.46
NAT_W_DOUBLE = 7.09
NAT_H_MAX    = 9.21   # maximum page height (234 mm)


def set_style() -> bool:
    """
    Apply Nature Astronomy rcParams.

    Always uses text.usetex=False with the STIX font set — no LaTeX
    installation required.  Returns False so callers can select plain
    unicode / mathtext labels without LaTeX-specific syntax.
    """
    params = {
        # ── Font (STIX: Times-like, bundled with matplotlib) ──────────────────
        "text.usetex"         : False,
        "font.family"         : "serif",
        "font.serif"          : ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset"    : "stix",
        "font.size"           : 8,
        "axes.labelsize"      : 9,
        "axes.titlesize"      : 9,
        "xtick.labelsize"     : 8,
        "ytick.labelsize"     : 8,
        # ── Legend ────────────────────────────────────────────────────────────
        "legend.fontsize"     : 8,
        "legend.frameon"      : False,
        "legend.borderpad"    : 0.3,
        "legend.handlelength" : 1.5,
        "legend.labelspacing" : 0.3,
        # ── Axes ──────────────────────────────────────────────────────────────
        "axes.linewidth"      : 0.8,
        "axes.grid"           : False,
        # ── Ticks: inward on all 4 sides, major + minor ───────────────────────
        "xtick.direction"     : "in",
        "ytick.direction"     : "in",
        "xtick.top"           : True,
        "ytick.right"         : True,
        "xtick.major.width"   : 0.6,
        "ytick.major.width"   : 0.6,
        "xtick.minor.visible" : True,
        "ytick.minor.visible" : True,
        "xtick.minor.width"   : 0.4,
        "ytick.minor.width"   : 0.4,
        "xtick.major.size"    : 4.0,
        "ytick.major.size"    : 4.0,
        "xtick.minor.size"    : 2.0,
        "ytick.minor.size"    : 2.0,
        # ── Lines ─────────────────────────────────────────────────────────────
        "lines.linewidth"     : 1.2,
        # ── Save ──────────────────────────────────────────────────────────────
        "figure.facecolor"    : "white",
        "savefig.facecolor"   : "white",
        "savefig.dpi"         : 300,
        "savefig.bbox"        : "tight",
    }
    mpl.rcParams.update(params)
    return False   # usetex is always off; callers use plain/mathtext labels
