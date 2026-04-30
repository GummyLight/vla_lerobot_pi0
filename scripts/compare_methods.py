"""Run eval_pi0.py against every trained method and print a comparison table.

Looks under outputs/train/pi0_3d_printer_<method>/checkpoints/last/pretrained_model
for each known method, runs offline eval on the open_3d_printer_test dataset,
then aggregates the per-run summary.json files into one table written as
outputs/eval/comparison.csv (and printed to stdout).

Usage:
    # Evaluate all methods that have a `last` checkpoint
    python scripts/compare_methods.py

    # Force re-run eval even if summary.json already exists
    python scripts/compare_methods.py --force

    # Restrict to specific methods
    python scripts/compare_methods.py --methods full lora
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = REPO_ROOT / "outputs" / "train"
EVAL_ROOT = REPO_ROOT / "outputs" / "eval"
EVAL_DATASET = REPO_ROOT / "datasets" / "open_3d_printer_test"

METHODS = ["full", "lora", "frozen"]


def find_checkpoint(method: str) -> Path | None:
    """Return path to the `last` pretrained_model dir for a method, if present."""
    p = TRAIN_ROOT / f"pi0_3d_printer_{method}" / "checkpoints" / "last" / "pretrained_model"
    return p if p.exists() else None


def run_eval(ckpt: Path, run_name: str, force: bool) -> Path:
    """Run scripts/eval_pi0.py and return the directory holding summary.json."""
    out_dir = EVAL_ROOT / run_name
    summary = out_dir / "summary.json"
    if summary.exists() and not force:
        print(f"  [skip] {run_name}: summary.json already exists (use --force to re-run)")
        return out_dir

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_pi0.py"),
        "--policy-path", str(ckpt),
        "--dataset-root", str(EVAL_DATASET),
    ]
    print(f"  [eval] {run_name}")
    print("    + " + " ".join(cmd))
    subprocess.check_call(cmd)
    return out_dir


def render_table(rows: list[dict]) -> str:
    """Plain-text table of method-level metrics."""
    if not rows:
        return "(no results)"

    header = ["method", "n_eps", "n_frames", "mae", "mse"]
    dim_count = len(rows[0].get("mae_per_dim", []))
    header += [f"mae[{i}]" for i in range(dim_count)]

    lines = [" | ".join(f"{h:>10}" for h in header)]
    lines.append("-+-".join("-" * 10 for _ in header))
    for r in rows:
        cells = [
            r["method"],
            str(r["n_episodes"]),
            str(r["n_frames"]),
            f"{r['mae']:.5f}",
            f"{r['mse']:.5f}",
        ]
        cells += [f"{v:.4f}" for v in r.get("mae_per_dim", [])]
        lines.append(" | ".join(f"{c:>10}" for c in cells))
    return "\n".join(lines)


METHOD_COLORS = {
    "full":   "#2E86AB",  # ocean blue
    "lora":   "#E07A5F",  # warm coral
    "frozen": "#81B29A",  # sage green
}
METHOD_LABELS = {"full": "Full SFT", "lora": "LoRA", "frozen": "Frozen vision"}


def plot_comparison(rows: list[dict], path: Path) -> None:
    """Render a clean side-by-side comparison figure."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#222",
        "xtick.color": "#444",
        "ytick.color": "#444",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
    })

    methods = [r["method"] for r in rows]
    colors = [METHOD_COLORS.get(m, "#888") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    mae = np.array([r["mae"] for r in rows])
    mse = np.array([r["mse"] for r in rows])
    per_dim = np.array([r["mae_per_dim"] for r in rows])  # (n_methods, n_dim)
    n_dim = per_dim.shape[1]

    fig = plt.figure(figsize=(13, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 5)
    ax_overall = fig.add_subplot(gs[0, 0:2])
    ax_perdim = fig.add_subplot(gs[0, 2:5])

    # --- Overall MAE / MSE (grouped bars) ---
    x = np.arange(len(methods))
    w = 0.36
    bars_mae = ax_overall.bar(x - w / 2, mae, w, color=colors, label="MAE", edgecolor="white", linewidth=1.2)
    bars_mse = ax_overall.bar(x + w / 2, mse, w, color=colors, alpha=0.55, label="MSE", edgecolor="white", linewidth=1.2, hatch="//")

    for bar, val in zip(bars_mae, mae):
        ax_overall.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}",
                        ha="center", va="bottom", fontsize=8, color="#222")
    for bar, val in zip(bars_mse, mse):
        ax_overall.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}",
                        ha="center", va="bottom", fontsize=8, color="#222")

    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(labels)
    ax_overall.set_ylabel("error")
    ax_overall.set_title("Overall action error (lower is better)")
    ax_overall.grid(axis="y", linestyle=":", alpha=0.5)
    ax_overall.set_axisbelow(True)
    # Custom legend (MAE = solid, MSE = hatched)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#888", edgecolor="white", label="MAE"),
        Patch(facecolor="#888", edgecolor="white", alpha=0.55, hatch="//", label="MSE"),
    ]
    ax_overall.legend(handles=legend_handles, loc="upper right", frameon=False)

    # --- Per-dim MAE (grouped bars) ---
    dim_x = np.arange(n_dim)
    group_w = 0.8
    bar_w = group_w / max(len(methods), 1)
    for i, (m, lab, c) in enumerate(zip(methods, labels, colors)):
        offset = (i - (len(methods) - 1) / 2) * bar_w
        ax_perdim.bar(dim_x + offset, per_dim[i], bar_w, color=c, label=lab,
                      edgecolor="white", linewidth=1.0)

    ax_perdim.set_xticks(dim_x)
    dim_names = ["j0", "j1", "j2", "j3", "j4", "j5", "grip"][:n_dim]
    if len(dim_names) < n_dim:
        dim_names += [f"d{i}" for i in range(len(dim_names), n_dim)]
    ax_perdim.set_xticklabels(dim_names)
    ax_perdim.set_xlabel("action dim")
    ax_perdim.set_ylabel("MAE")
    ax_perdim.set_title("Per-dimension MAE")
    ax_perdim.grid(axis="y", linestyle=":", alpha=0.5)
    ax_perdim.set_axisbelow(True)
    ax_perdim.legend(loc="upper right", frameon=False)

    fig.suptitle("pi0 fine-tune comparison on open_3d_printer_test",
                 fontsize=14, fontweight="bold", color="#222")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        path.write_text("")
        return
    dim_count = len(rows[0].get("mae_per_dim", []))
    fieldnames = ["method", "n_episodes", "n_frames", "mae", "mse"] + [
        f"mae_dim_{i}" for i in range(dim_count)
    ] + [f"mse_dim_{i}" for i in range(dim_count)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r[k] for k in ("method", "n_episodes", "n_frames", "mae", "mse")}
            for i, v in enumerate(r.get("mae_per_dim", [])):
                row[f"mae_dim_{i}"] = v
            for i, v in enumerate(r.get("mse_per_dim", [])):
                row[f"mse_dim_{i}"] = v
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS,
                    choices=METHODS, help="Subset of methods to evaluate.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run eval even if summary.json already exists.")
    args = ap.parse_args()

    rows: list[dict] = []
    missing: list[str] = []

    for method in args.methods:
        ckpt = find_checkpoint(method)
        if ckpt is None:
            missing.append(method)
            continue

        run_name = f"pi0_3d_printer_{method}"
        eval_dir = run_eval(ckpt, run_name, args.force)
        summary_path = eval_dir / "summary.json"
        if not summary_path.exists():
            print(f"  [warn] {method}: eval ran but no summary.json found")
            continue
        summary = json.loads(summary_path.read_text())
        summary["method"] = method
        rows.append(summary)

    if missing:
        print(f"\nNo `last` checkpoint for: {', '.join(missing)}  (skipped)")

    print("\n=== comparison ===\n")
    print(render_table(rows))

    csv_path = EVAL_ROOT / "comparison.csv"
    write_csv(rows, csv_path)
    print(f"\nwrote {csv_path}")

    fig_path = EVAL_ROOT / "comparison.png"
    plot_comparison(rows, fig_path)
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
