#!/usr/bin/env python3
"""
val_loss_plot.py — Validation loss comparison across SBI_prods_* runs.

Reads training.log from --dataset (and any sibling SBI_prods_* directories
in the same parent folder), parses per-chunk best validation losses, and
produces a single comparison plot saved to <dataset>/validation_losses.png.

Can be run standalone or invoked by run_pipeline.py --val-loss-plot.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .natastro_style import set_style, NAT_W_DOUBLE

# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot validation loss curves across SBI_prods_* training runs."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Primary dataset directory (the 'current' run).")
    parser.add_argument("--training_log", type=str, default=None,
                        help="Override path to training.log for the primary run.")
    args = parser.parse_args(argv)

    log_path = args.training_log or os.path.join(args.dataset, "training.log")

    # ──────────────────────────────────────────────────────────────────────────────
    # Parsing helpers
    # ──────────────────────────────────────────────────────────────────────────────
    _PATTERNS = [
        re.compile(r"Best validation performance:\s*(-?[\d.]+)"),
        re.compile(r"best_validation_loss['\"]?\s*:\s*(-?[\d.]+)"),
        re.compile(r"Validation loss:\s*(-?[\d.]+)"),
        re.compile(r"val[_ ]loss[:\s]+(-?[\d.]+)"),
    ]


    def _parse_val_losses(path):
        losses = []
        with open(path) as fh:
            for line in fh:
                for pat in _PATTERNS:
                    m = pat.search(line)
                    if m:
                        losses.append(float(m.group(1)))
                        break
        return losses


    def _run_label(dirpath):
        """SBI_prods_6000000 → '6M sims', SBI_prods_400000 → '400k sims'."""
        name = os.path.basename(dirpath.rstrip("/\\"))
        try:
            n = int(name.split("_")[-1])
            if n >= 1_000_000:
                frac = n / 1_000_000
                return f"{int(frac)}M sims" if frac == int(frac) else f"{frac:.1f}M sims"
            elif n >= 1_000:
                return f"{n // 1_000}k sims"
            return f"{n} sims"
        except ValueError:
            return name


    # ──────────────────────────────────────────────────────────────────────────────
    # Load primary run
    # ──────────────────────────────────────────────────────────────────────────────
    if not os.path.isfile(log_path):
        print(f"ERROR: training.log not found at {log_path}")
        sys.exit(1)

    val_losses = _parse_val_losses(log_path)
    if not val_losses:
        print(f"WARNING: no validation loss entries matched in {log_path}")
        print("  First 5 lines:")
        with open(log_path) as fh:
            for i, line in enumerate(fh):
                if i >= 5:
                    break
                print(f"    {line.rstrip()}")
        sys.exit(1)

    cur_label = _run_label(args.dataset)
    print(f"\nPrimary run : {cur_label}  ({len(val_losses)} chunks, best = {min(val_losses):.4f})")

    print("\n── Validation losses per chunk ──────────────────────")
    for i, v in enumerate(val_losses, 1):
        print(f"  Chunk {i:>3d} : {v:.4f}")
    print(f"  Best overall : {min(val_losses):.4f}  (chunk {val_losses.index(min(val_losses)) + 1})")
    print("─────────────────────────────────────────────────────")

    # ──────────────────────────────────────────────────────────────────────────────
    # Discover sibling SBI_prods_* directories
    # ──────────────────────────────────────────────────────────────────────────────
    parent_dir   = os.path.dirname(os.path.abspath(args.dataset))
    sibling_dirs = sorted(glob.glob(os.path.join(parent_dir, "SBI_prods_*")))
    cur_abs      = os.path.abspath(args.dataset)

    all_series = {cur_label: val_losses}   # current run first

    for d in sibling_dirs:
        if os.path.abspath(d) == cur_abs:
            continue
        lp = os.path.join(d, "training.log")
        if not os.path.isfile(lp):
            continue
        losses = _parse_val_losses(lp)
        if losses:
            label = _run_label(d)
            all_series[label] = losses
            print(f"Sibling run : {label}  ({len(losses)} chunks, best = {min(losses):.4f})")

    # ──────────────────────────────────────────────────────────────────────────────
    # Plot
    # ──────────────────────────────────────────────────────────────────────────────
    set_style()

    _COLORS = [
        "#1F3A5F", "#C0392B", "#27AE60", "#E67E22",
        "#8E44AD", "#16A085", "#D35400", "#2980B9",
        "#7F8C8D", "#F39C12",
    ]

    fig, ax = plt.subplots(figsize=(NAT_W_DOUBLE, 3.0))

    for idx, (label, losses) in enumerate(all_series.items()):
        col    = _COLORS[idx % len(_COLORS)]
        is_cur = (label == cur_label)
        ax.plot(range(1, len(losses) + 1), losses,
                marker="o", markersize=2.5,
                lw=1.5,
                alpha=1.0,
                color=col, zorder=3 if is_cur else 2,
                label=f"{label}  (best = {min(losses):.4f})")
        ax.axhline(min(losses), color=col, lw=0.7, ls="--", alpha=0.45, zorder=1)

    ax.set_xlabel("Chunk")
    ax.set_ylabel("Best validation loss")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(pad=0.5)

    os.makedirs(args.dataset, exist_ok=True)
    out = os.path.join(args.dataset, "validation_losses.png")
    fig.savefig(out)
    plt.close()
    print(f"\nValidation loss plot saved → {out}")


if __name__ == "__main__":
    main()
