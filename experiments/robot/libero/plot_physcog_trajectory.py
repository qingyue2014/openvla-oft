"""
plot_physcog_trajectory.py

Top-down (XY) plot of recorded PhysCogSafe episode trajectories, for quickly
judging whether carry paths are geometrically sensible without watching every
video.

Draws, per episode: the policy-phase EEF path colored by time, the tracked
target-object path, start/end markers, initial positions of all tracked
bodies, and (optionally) a threshold circle around a hazard body.

Usage:
  python -m experiments.robot.libero.plot_physcog_trajectory \
    rollouts/libero_10/L2-B1-cream-cheese-stove-beside-plate-carry/trajectories/*.npz \
    --hazard_body flat_stove_1_burner --threshold 0.10 \
    --target_body cream_cheese_1_main \
    --out l2b1_er_paths.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.robot.libero.physcog_trajectory import load_trajectory


def plot_episode(ax, traj: dict, target_body: str = "", hazard_body: str = "", threshold: float = 0.0):
    meta = traj["metadata"]
    policy = traj["phases"] == "policy"
    eef = traj["eef_pos"][policy]
    eef = eef[~np.isnan(eef).any(axis=1)]

    if eef.shape[0] >= 2:
        points = ax.scatter(
            eef[:, 0], eef[:, 1], c=np.linspace(0, 1, eef.shape[0]),
            cmap="viridis", s=6, zorder=3, label="EEF (dark→light = time)",
        )
        ax.plot(eef[0, 0], eef[0, 1], "g^", markersize=10, zorder=4, label="EEF start")
        ax.plot(eef[-1, 0], eef[-1, 1], "rv", markersize=10, zorder=4, label="EEF end")

    target_key = f"body_pos__{target_body}" if target_body else ""
    if target_key and target_key in traj:
        tp = traj[target_key][policy]
        ax.plot(tp[:, 0], tp[:, 1], "-", color="tab:orange", linewidth=2, zorder=2,
                label=f"{target_body} path")

    for key in traj:
        if not key.startswith("body_pos__"):
            continue
        name = key[len("body_pos__"):]
        pos0 = traj[key][0]
        marker = "s"
        color = "tab:red" if name == hazard_body else "tab:gray"
        ax.plot(pos0[0], pos0[1], marker, color=color, markersize=9, zorder=2)
        ax.annotate(name, (pos0[0], pos0[1]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
        if name == hazard_body and threshold > 0:
            ax.add_patch(plt.Circle((pos0[0], pos0[1]), threshold, fill=False,
                                    color="tab:red", linestyle="--", linewidth=1,
                                    zorder=1, label=f"threshold {threshold} m"))

    status = []
    status.append("success" if meta.get("success") else "fail")
    if meta.get("violated"):
        status.append(f"VIOLATED@{meta.get('violation_step')}")
    ax.set_title(f"ep{meta.get('episode_idx', '?')} ({', '.join(status)})", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3)


def main():
    parser = argparse.ArgumentParser(description="Top-down XY plot of PhysCog episode trajectories")
    parser.add_argument("npz", nargs="+", help="Episode .npz file(s)")
    parser.add_argument("--target_body", default="", help="Tracked body to draw as the carried-object path")
    parser.add_argument("--hazard_body", default="", help="Tracked body to highlight with a threshold circle")
    parser.add_argument("--threshold", type=float, default=0.0, help="Radius of the hazard circle in metres")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--out", default="physcog_trajectories.png")
    args = parser.parse_args()

    n = len(args.npz)
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)

    for idx, path in enumerate(sorted(args.npz)):
        ax = axes[idx // cols][idx % cols]
        plot_episode(ax, load_trajectory(path), args.target_body, args.hazard_body, args.threshold)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle(os.path.dirname(os.path.commonpath(args.npz)) or args.npz[0], fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out} ({n} episodes)")


if __name__ == "__main__":
    main()
