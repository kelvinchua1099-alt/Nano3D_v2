#!/usr/bin/env python3
"""
Visualize V_delta (V_tar - V_src) heatmaps saved during flow editing.

Usage:
    python visualize_vdelta.py --vdelta_dir <output_dir>/vdelta --output_dir <output_dir>/vdelta_vis

Each saved file has shape [1, 8, 16, 16, 16].
For each step we compute the per-voxel L2 norm across 8 channels → [16, 16, 16],
then show 3 orthogonal max-projections (XY / XZ / YZ) as heatmaps.

Output:
    - One PNG per step:  step_NNN_t0.XXXX.png
    - One summary grid:  summary.png  (all steps in a single figure)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vdelta_dir(vdelta_dir: str):
    """
    Load all step_*.npy files in order.
    Returns list of dicts: {"step": int, "t": float, "v_delta": np.ndarray [1,8,16,16,16]}
    """
    files = sorted(Path(vdelta_dir).glob("step_*.npy"))
    if len(files) == 0:
        raise FileNotFoundError(f"No step_*.npy files found in {vdelta_dir}")

    records = []
    for f in files:
        # filename: step_012_t0.7600.npy
        stem = f.stem                      # step_012_t0.7600
        parts = stem.split("_")
        step_idx = int(parts[1])
        t_val = float(parts[2][1:])        # strip leading 't'
        v = np.load(str(f))                # [1, 8, 16, 16, 16]
        records.append({"step": step_idx, "t": t_val, "v_delta": v})

    print(f"Loaded {len(records)} V_delta tensors from {vdelta_dir}")
    return records


def compute_magnitude(v_delta: np.ndarray) -> np.ndarray:
    """
    v_delta: [1, 8, 16, 16, 16]
    Returns L2 norm across channel dim: [16, 16, 16]
    """
    v = v_delta[0]                   # [8, 16, 16, 16]
    mag = np.sqrt((v ** 2).sum(axis=0))   # [16, 16, 16]
    return mag


def max_project(vol: np.ndarray):
    """
    vol: [D, H, W]
    Returns three max-projections: XY (top), XZ (front), YZ (side)
    """
    xy = vol.max(axis=0)   # project along D → [H, W]
    xz = vol.max(axis=1)   # project along H → [D, W]
    yz = vol.max(axis=2)   # project along W → [D, H]
    return xy, xz, yz


def plot_step(record: dict, vmax: float, output_path: str):
    """
    Draw one figure (3 projection heatmaps + channel grid) for a single step.
    """
    mag = compute_magnitude(record["v_delta"])   # [16, 16, 16]
    xy, xz, yz = max_project(mag)

    v = record["v_delta"][0]   # [8, 16, 16, 16]

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(
        f"V_delta  |  step={record['step']}  t={record['t']:.4f}",
        fontsize=13, fontweight="bold"
    )

    # ---- Row 1: 3 projections of L2 magnitude ----
    proj_labels = ["XY  (top view,  max over Z)",
                   "XZ  (front view, max over Y)",
                   "YZ  (side view,  max over X)"]
    proj_data   = [xy, xz, yz]
    for col, (label, data) in enumerate(zip(proj_labels, proj_data)):
        ax = fig.add_subplot(2, 8, col + 1)
        im = ax.imshow(data, cmap="hot", vmin=0, vmax=vmax,
                       interpolation="nearest", origin="upper")
        ax.set_title(label, fontsize=7)
        ax.set_xlabel("axis-1"); ax.set_ylabel("axis-0")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Row 2: per-channel middle-slice heatmaps (z=8) ----
    for ch in range(8):
        ax = fig.add_subplot(2, 8, 9 + ch)
        slice_data = v[ch, 8, :, :]   # middle slice along Z
        vmax_ch    = np.abs(v[ch]).max()
        norm       = mcolors.TwoSlopeNorm(vmin=-vmax_ch, vcenter=0, vmax=vmax_ch)
        im = ax.imshow(slice_data, cmap="RdBu_r", norm=norm,
                       interpolation="nearest", origin="upper")
        ax.set_title(f"ch {ch}  (z=8)", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary(records: list, output_path: str):
    """
    Summary grid: rows = steps, cols = 3 projections (L2 magnitude).
    """
    n = len(records)
    mags = [compute_magnitude(r["v_delta"]) for r in records]
    vmax = max(m.max() for m in mags)   # global scale

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]   # ensure 2D indexing

    proj_titles = ["XY (max over Z)", "XZ (max over Y)", "YZ (max over X)"]

    for row, (record, mag) in enumerate(zip(records, mags)):
        xy, xz, yz = max_project(mag)
        for col, (title, data) in enumerate(zip(proj_titles, [xy, xz, yz])):
            ax = axes[row, col]
            im = ax.imshow(data, cmap="hot", vmin=0, vmax=vmax,
                           interpolation="nearest", origin="upper")
            if row == 0:
                ax.set_title(title, fontsize=8)
            ax.set_ylabel(f"step={record['step']}\nt={record['t']:.3f}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("V_delta  L2 magnitude — all editing steps", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary saved to: {output_path}")


def plot_magnitude_curve(records: list, output_path: str):
    """
    Line chart: mean and max L2 magnitude per step.
    """
    steps = [r["step"] for r in records]
    t_vals = [r["t"] for r in records]
    means  = [compute_magnitude(r["v_delta"]).mean() for r in records]
    maxes  = [compute_magnitude(r["v_delta"]).max()  for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(steps, means, "o-", color="steelblue",  label="mean |V_delta|", linewidth=2)
    ax1.plot(steps, maxes, "s--", color="tomato",    label="max  |V_delta|",  linewidth=2)
    ax1.set_xlabel("denoising step index")
    ax1.set_ylabel("L2 magnitude")
    ax1.set_title("V_delta magnitude over denoising steps")
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(steps[::max(1, len(steps)//8)])
    ax2.set_xticklabels([f"{t:.2f}" for t in t_vals[::max(1, len(t_vals)//8)]], fontsize=7)
    ax2.set_xlabel("t value")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Magnitude curve saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize V_delta heatmaps")
    parser.add_argument(
        "--vdelta_dir",
        type=str,
        required=True,
        help="Directory containing step_*.npy files (e.g. <output>/vdelta)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save visualization PNGs"
    )
    parser.add_argument(
        "--per_step",
        action="store_true",
        help="Also save one detailed PNG per step (in addition to summary)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_vdelta_dir(args.vdelta_dir)

    # global vmax for consistent color scale across steps
    mags  = [compute_magnitude(r["v_delta"]) for r in records]
    vmax  = max(m.max() for m in mags)
    print(f"Global V_delta magnitude range: 0 ~ {vmax:.4f}")

    # 1. Summary grid
    plot_summary(records, os.path.join(args.output_dir, "summary.png"))

    # 2. Magnitude curve
    plot_magnitude_curve(records, os.path.join(args.output_dir, "magnitude_curve.png"))

    # 3. Per-step detailed figures (optional)
    if args.per_step:
        per_step_dir = os.path.join(args.output_dir, "per_step")
        os.makedirs(per_step_dir, exist_ok=True)
        for r in records:
            fname = f"step_{r['step']:03d}_t{r['t']:.4f}.png"
            plot_step(r, vmax, os.path.join(per_step_dir, fname))
        print(f"Per-step figures saved to: {per_step_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
