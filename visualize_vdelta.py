#!/usr/bin/env python3
"""
Visualize V_delta (V_tar - V_src) heatmaps and cosine-similarity analyses.

Usage:
    python visualize_vdelta.py \
        --vdelta_dir  <output>/vdelta \
        --output_dir  <output>/vdelta_vis [--per_step]

Outputs
-------
summary.png               — all-step heatmaps (3 max-projections per step)
magnitude_curve.png       — mean/max |V_delta| over steps
consecutive_cosine.png    — cos_sim( V_delta(t), V_delta(t-1) ) per step
voxel_diff_cosine.png     — cos_sim( delta_voxel, V_delta(t) ) per step  [global]
voxel_diff_heatmap.png    — per-voxel cos_sim grid across steps           [spatial]
per_step/step_NNN_*.png   — detailed per-step figures (--per_step flag)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ============================================================
# I/O helpers
# ============================================================

def load_vdelta_dir(vdelta_dir: str):
    """
    Load all step_*.npy files in sorted order.
    Returns list of dicts: {step, t, v_delta: [1,8,16,16,16]}
    """
    files = sorted(Path(vdelta_dir).glob("step_*.npy"))
    if not files:
        raise FileNotFoundError(f"No step_*.npy files in {vdelta_dir}")

    records = []
    for f in files:
        stem  = f.stem                 # step_012_t0.7600
        parts = stem.split("_")
        step_idx = int(parts[1])
        t_val    = float(parts[2][1:]) # strip leading 't'
        records.append({
            "step":    step_idx,
            "t":       t_val,
            "v_delta": np.load(str(f)),
        })

    print(f"Loaded {len(records)} V_delta tensors from {vdelta_dir}")
    return records


def load_voxel_latents(vdelta_dir: str):
    """
    Load x_src_packed and zt_edit_final saved during sampling.
    Returns (x_src, zt_final) both as float32 numpy [1,8,16,16,16], or None if missing.
    """
    src_path   = os.path.join(vdelta_dir, "x_src_packed.npy")
    final_path = os.path.join(vdelta_dir, "zt_edit_final.npy")

    x_src    = np.load(src_path).astype(np.float32)   if os.path.exists(src_path)   else None
    zt_final = np.load(final_path).astype(np.float32) if os.path.exists(final_path) else None

    if x_src is None:
        print("  ⚠ x_src_packed.npy not found — skipping voxel-diff analysis")
    if zt_final is None:
        print("  ⚠ zt_edit_final.npy not found — skipping voxel-diff analysis")
    return x_src, zt_final


# ============================================================
# Math helpers
# ============================================================

def magnitude(v_delta: np.ndarray) -> np.ndarray:
    """[1,8,16,16,16] → [16,16,16]  (L2 norm across channels)"""
    v = v_delta[0]                          # [8,16,16,16]
    return np.sqrt((v ** 2).sum(axis=0))    # [16,16,16]


def max_project(vol: np.ndarray):
    """[D,H,W] → (XY, XZ, YZ) max-projections"""
    return vol.max(axis=0), vol.max(axis=1), vol.max(axis=2)


def cosine_sim_global(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity treating both arrays as flat 1-D vectors."""
    a, b  = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def cosine_sim_per_voxel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Per-voxel cosine similarity along the channel axis.
    a, b : [1,8,16,16,16]  or  [8,16,16,16]
    Returns: [16,16,16]  in range [-1, 1]
    """
    if a.ndim == 5: a = a[0]
    if b.ndim == 5: b = b[0]
    dot    = (a * b).sum(axis=0)
    norm_a = np.sqrt((a ** 2).sum(axis=0))
    norm_b = np.sqrt((b ** 2).sum(axis=0))
    denom  = norm_a * norm_b
    return np.where(denom > 1e-12, dot / denom, 0.0)


# ============================================================
# Plot 1 — Heatmap summary (existing, kept identical)
# ============================================================

def plot_summary(records, output_path):
    n    = len(records)
    mags = [magnitude(r["v_delta"]) for r in records]
    vmax = max(m.max() for m in mags)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    proj_titles = ["XY (max over Z)", "XZ (max over Y)", "YZ (max over X)"]
    for row, (rec, mag) in enumerate(zip(records, mags)):
        for col, (title, data) in enumerate(zip(proj_titles, max_project(mag))):
            ax = axes[row, col]
            im = ax.imshow(data, cmap="hot", vmin=0, vmax=vmax,
                           interpolation="nearest", origin="upper")
            if row == 0:
                ax.set_title(title, fontsize=8)
            ax.set_ylabel(f"step={rec['step']}\nt={rec['t']:.3f}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("V_delta  L2 magnitude — all editing steps",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Plot 2 — Magnitude curve (existing, kept identical)
# ============================================================

def plot_magnitude_curve(records, output_path):
    steps  = [r["step"] for r in records]
    t_vals = [r["t"]    for r in records]
    means  = [magnitude(r["v_delta"]).mean() for r in records]
    maxes  = [magnitude(r["v_delta"]).max()  for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(steps, means, "o-",  color="steelblue", label="mean |V_delta|", lw=2)
    ax1.plot(steps, maxes, "s--", color="tomato",    label="max  |V_delta|", lw=2)
    ax1.set_xlabel("denoising step index")
    ax1.set_ylabel("L2 magnitude")
    ax1.set_title("V_delta magnitude over denoising steps")
    ax1.legend()

    tick_idx = list(range(0, len(steps), max(1, len(steps) // 8)))
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([steps[i] for i in tick_idx])
    ax2.set_xticklabels([f"{t_vals[i]:.2f}" for i in tick_idx], fontsize=7)
    ax2.set_xlabel("t value")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Plot 3 — Consecutive cosine similarity
# ============================================================

def plot_consecutive_cosine(records, output_path):
    """
    For each pair of adjacent steps, compute:
        cos_sim( V_delta(t),  V_delta(t-1) )
    Both global scalar (upper subplot) and per-voxel mean/std (lower subplot).
    """
    if len(records) < 2:
        print("  ⚠ Need at least 2 steps for consecutive cosine — skipping")
        return

    steps_mid = []   # step index of the LATER step in each pair
    t_mid     = []
    global_cs = []   # scalar cosine similarity
    pv_mean   = []   # per-voxel cos_sim mean
    pv_std    = []   # per-voxel cos_sim std

    for i in range(1, len(records)):
        a = records[i - 1]["v_delta"]   # [1,8,16,16,16]
        b = records[i    ]["v_delta"]

        steps_mid.append(records[i]["step"])
        t_mid.append(records[i]["t"])
        global_cs.append(cosine_sim_global(a, b))

        pv = cosine_sim_per_voxel(a, b)  # [16,16,16]
        pv_mean.append(pv.mean())
        pv_std.append(pv.std())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # ---- upper: global cosine sim ----
    ax1.plot(steps_mid, global_cs, "o-", color="darkorange", lw=2)
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.axhline(1, color="gray", lw=0.8, ls=":")
    ax1.set_ylabel("cos_sim  (global)")
    ax1.set_title("cos_sim( V_delta(t),  V_delta(t−1) )")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # annotate values
    for x, y in zip(steps_mid, global_cs):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=6)

    # ---- lower: per-voxel mean ± std ----
    pv_mean = np.array(pv_mean)
    pv_std  = np.array(pv_std)
    ax2.plot(steps_mid, pv_mean, "o-", color="mediumseagreen", lw=2, label="per-voxel mean")
    ax2.fill_between(steps_mid,
                     pv_mean - pv_std,
                     pv_mean + pv_std,
                     alpha=0.25, color="mediumseagreen", label="±1 std")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("denoising step index")
    ax2.set_ylabel("cos_sim  (per-voxel mean)")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # secondary x-axis for t values
    tick_idx = list(range(0, len(steps_mid), max(1, len(steps_mid) // 8)))
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks([steps_mid[i] for i in tick_idx])
    ax3.set_xticklabels([f"{t_mid[i]:.2f}" for i in tick_idx], fontsize=7)
    ax3.set_xlabel("t value")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Plot 4 — Voxel-diff cosine similarity (line chart)
# ============================================================

def plot_voxel_diff_cosine(records, x_src, zt_final, output_path):
    """
    delta_voxel = zt_edit_final - x_src_packed   [1,8,16,16,16]

    For each step:
        global_cs[step]  = cos_sim( delta_voxel.flatten(),  V_delta[step].flatten() )
        pv_mean[step]    = mean over voxels of per-voxel cos_sim
    """
    delta_voxel = zt_final - x_src   # [1,8,16,16,16]

    steps    = [r["step"] for r in records]
    t_vals   = [r["t"]    for r in records]
    global_cs = []
    pv_mean   = []
    pv_std    = []

    for rec in records:
        vd = rec["v_delta"]
        global_cs.append(cosine_sim_global(delta_voxel, vd))
        pv = cosine_sim_per_voxel(delta_voxel, vd)
        pv_mean.append(pv.mean())
        pv_std.append(pv.std())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # ---- upper: global ----
    ax1.plot(steps, global_cs, "o-", color="royalblue", lw=2)
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_ylabel("cos_sim  (global)")
    ax1.set_title("cos_sim( zt_edit_final − x_src,  V_delta(t) )\n"
                  "— alignment between editing direction and final displacement —")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)

    for x, y in zip(steps, global_cs):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=6)

    # ---- lower: per-voxel mean ± std ----
    pv_mean = np.array(pv_mean)
    pv_std  = np.array(pv_std)
    ax2.plot(steps, pv_mean, "o-", color="mediumpurple", lw=2, label="per-voxel mean")
    ax2.fill_between(steps,
                     pv_mean - pv_std,
                     pv_mean + pv_std,
                     alpha=0.25, color="mediumpurple", label="±1 std")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("denoising step index")
    ax2.set_ylabel("cos_sim  (per-voxel mean)")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    tick_idx = list(range(0, len(steps), max(1, len(steps) // 8)))
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks([steps[i] for i in tick_idx])
    ax3.set_xticklabels([f"{t_vals[i]:.2f}" for i in tick_idx], fontsize=7)
    ax3.set_xlabel("t value")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Plot 5 — Voxel-diff cosine similarity heatmaps (spatial)
# ============================================================

def plot_voxel_diff_heatmap(records, x_src, zt_final, output_path):
    """
    For each step, per-voxel cos_sim( delta_voxel, V_delta ) → [16,16,16]
    Shown as 3 max-projections, all steps in one grid.
    """
    delta_voxel = zt_final - x_src   # [1,8,16,16,16]
    n = len(records)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    proj_titles = ["XY (max over Z)", "XZ (max over Y)", "YZ (max over X)"]

    # gather all maps first for global color scale
    all_maps = []
    for rec in records:
        pv = cosine_sim_per_voxel(delta_voxel, rec["v_delta"])   # [16,16,16]
        all_maps.append(pv)

    # symmetric color range around 0
    vabs = max(abs(m).max() for m in all_maps)

    for row, (rec, pv_map) in enumerate(zip(records, all_maps)):
        norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        for col, (title, proj) in enumerate(zip(proj_titles, max_project(pv_map))):
            ax = axes[row, col]
            im = ax.imshow(proj, cmap="RdBu_r", norm=norm,
                           interpolation="nearest", origin="upper")
            if row == 0:
                ax.set_title(title, fontsize=8)
            ax.set_ylabel(f"step={rec['step']}\nt={rec['t']:.3f}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Per-voxel cos_sim( delta_voxel,  V_delta(t) )\n"
                 "Blue=opposing  ·  Red=aligned",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Plot 6 — Per-step detailed figure (optional)
# ============================================================

def plot_step(record, vmax, output_path):
    mag = magnitude(record["v_delta"])
    v   = record["v_delta"][0]   # [8,16,16,16]

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f"V_delta  |  step={record['step']}  t={record['t']:.4f}",
                 fontsize=13, fontweight="bold")

    proj_labels = ["XY (max over Z)", "XZ (max over Y)", "YZ (max over X)"]
    for col, (label, data) in enumerate(zip(proj_labels, max_project(mag))):
        ax = fig.add_subplot(2, 8, col + 1)
        im = ax.imshow(data, cmap="hot", vmin=0, vmax=vmax,
                       interpolation="nearest", origin="upper")
        ax.set_title(label, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ch in range(8):
        ax = fig.add_subplot(2, 8, 9 + ch)
        sl = v[ch, 8, :, :]
        vabs = np.abs(v[ch]).max()
        norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(sl, cmap="RdBu_r", norm=norm,
                       interpolation="nearest", origin="upper")
        ax.set_title(f"ch {ch}  (z=8)", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize V_delta heatmaps and cosine analyses")
    parser.add_argument("--vdelta_dir", type=str, required=True,
                        help="Directory with step_*.npy, x_src_packed.npy, zt_edit_final.npy")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save all output PNGs")
    parser.add_argument("--per_step", action="store_true",
                        help="Also save one detailed PNG per step")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load data ----
    records            = load_vdelta_dir(args.vdelta_dir)
    x_src, zt_final    = load_voxel_latents(args.vdelta_dir)

    mags = [magnitude(r["v_delta"]) for r in records]
    vmax = max(m.max() for m in mags)
    print(f"Global V_delta magnitude range: 0 ~ {vmax:.4f}")

    # ---- plots ----
    plot_summary(
        records,
        os.path.join(args.output_dir, "summary.png")
    )
    plot_magnitude_curve(
        records,
        os.path.join(args.output_dir, "magnitude_curve.png")
    )
    plot_consecutive_cosine(
        records,
        os.path.join(args.output_dir, "consecutive_cosine.png")
    )

    if x_src is not None and zt_final is not None:
        plot_voxel_diff_cosine(
            records, x_src, zt_final,
            os.path.join(args.output_dir, "voxel_diff_cosine.png")
        )
        plot_voxel_diff_heatmap(
            records, x_src, zt_final,
            os.path.join(args.output_dir, "voxel_diff_heatmap.png")
        )

    if args.per_step:
        per_dir = os.path.join(args.output_dir, "per_step")
        os.makedirs(per_dir, exist_ok=True)
        for r in records:
            fname = f"step_{r['step']:03d}_t{r['t']:.4f}.png"
            plot_step(r, vmax, os.path.join(per_dir, fname))
        print(f"Per-step figures saved to: {per_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
