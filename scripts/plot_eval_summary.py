"""Render bar charts from an eval `summary.json` (per-dim MAE/MSE, per task).

Usage:
    python scripts/plot_eval_summary.py outputs/eval/pi0_3d_printer_lora_step005000/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 7-DoF: UR has 6 joints + 1 gripper.
DIM_LABELS = ["j0\nbase", "j1\nshoulder", "j2\nelbow", "j3\nwrist_1", "j4\nwrist_2", "j5\nwrist_3", "grip"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path, help="Path to summary.json produced by eval_pi0.py")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output PNG path. Defaults to <summary_dir>/summary.png")
    return ap.parse_args()


def collect_blocks(summary: dict) -> dict[str, dict]:
    """Return {label: metric_block}, with task-specific blocks if present."""
    blocks: dict[str, dict] = {"overall": summary["overall"]}
    if "by_task" in summary:
        for task, blk in summary["by_task"].items():
            short = task.replace("the 3D printer", "").strip() or task
            blocks[short] = blk
    return blocks


def plot_per_dim_bars(ax, blocks: dict[str, dict], metric_key: str, ylabel: str) -> None:
    n_dims = len(DIM_LABELS)
    n_groups = len(blocks)
    x = np.arange(n_dims)
    width = 0.8 / n_groups
    colors = ["#444", "#3a86ff", "#f4a261", "#2a9d8f", "#e76f51"]
    for i, (label, blk) in enumerate(blocks.items()):
        vals = blk[metric_key]
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.95, label=label, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_LABELS, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)


def plot_overall_bars(ax, blocks: dict[str, dict]) -> None:
    labels = list(blocks.keys())
    mae = [b["mae"] for b in blocks.values()]
    mse = [b["mse"] for b in blocks.values()]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, mae, w, label="MAE (rad)", color="#3a86ff")
    ax.bar(x + w / 2, mse, w, label="MSE", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("error (lower is better)")
    ax.set_title("aggregate error per group")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    for i, (m, s) in enumerate(zip(mae, mse)):
        ax.text(i - w / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, s, f"{s:.3f}", ha="center", va="bottom", fontsize=8)


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    blocks = collect_blocks(summary)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [1.2, 1.2, 0.8]})

    # 1. Per-dim MAE
    plot_per_dim_bars(axes[0], blocks, "mae_per_dim", "MAE (rad for joints, [0,1] for gripper)")
    axes[0].set_title("Per-dimension MAE (lower is better)")
    # Reference line at 0.087 rad ≈ 5° -- a reasonable "good" threshold.
    axes[0].axhline(0.087, color="gray", linestyle=":", alpha=0.6, label="_5° reference")
    axes[0].text(len(DIM_LABELS) - 0.5, 0.087, " 5°", color="gray", va="bottom", fontsize=8)

    # 2. Per-dim MSE
    plot_per_dim_bars(axes[1], blocks, "mse_per_dim", "MSE")
    axes[1].set_title("Per-dimension MSE")

    # 3. Overall MAE/MSE per group
    plot_overall_bars(axes[2], blocks)

    n_eps = summary["overall"]["n_episodes"]
    n_frames = summary["overall"]["n_frames"]
    fig.suptitle(f"pi0 LoRA eval — {n_eps} episodes, {n_frames} frames", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = args.output or args.summary.parent / "summary.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
