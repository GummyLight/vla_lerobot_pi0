"""Launch pi0 fine-tuning on the local open_3d_printer_diversified dataset.

Three training paradigms are supported via --method:

  full     Full-parameter SFT. All ~3B params get gradients. Highest memory,
           highest ceiling. (Default.)
  lora     LoRA SFT. Frozen base + low-rank adapters. ~1-5% trainable params.
           Friendly to 16-24 GB GPUs and small datasets.
  frozen   Full SFT but the vision encoder is frozen. Compromise between
           full and lora.

Each method writes to its own output_dir so you can train all three and then
compare with scripts/compare_methods.py.

Examples
--------
    # Default: full fine-tune
    python scripts/train_pi0.py

    # LoRA
    python scripts/train_pi0.py --method=lora

    # Frozen vision tower
    python scripts/train_pi0.py --method=frozen

    # Forward extra args straight to lerobot's train CLI
    python scripts/train_pi0.py --method=lora --steps=10000 --batch_size=4 --wandb.enable=true
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "datasets" / "open_3d_printer_diversified"


METHODS = {
    "full": {
        "extra_args": [],
        "default_steps": 30000,
        "default_batch_size": 8,
        "default_lr": 2.5e-5,
    },
    "lora": {
        "extra_args": [
            # PEFT enabled via top-level --peft.* config; do NOT set
            # --policy.use_peft=true because that tells make_policy to load an
            # already-trained adapter from --policy.pretrained_path. We want to
            # load the base pi0 weights and let lerobot_train.py wrap them with
            # a fresh LoRA adapter via `policy.wrap_with_peft(...)`.
            "--peft.method_type=lora",
            "--peft.r=16",
            # bf16 by default for LoRA — saves a lot of memory at almost no cost
            "--policy.dtype=bfloat16",
        ],
        "default_steps": 20000,
        "default_batch_size": 4,
        "default_lr": 1e-4,  # LoRA tolerates higher LR
    },
    "frozen": {
        "extra_args": [
            "--policy.freeze_vision_encoder=true",
            "--policy.dtype=bfloat16",
        ],
        "default_steps": 25000,
        "default_batch_size": 6,
        "default_lr": 2.5e-5,
    },
}


def ensure_pi0_weights() -> Path:
    """Snapshot-download `lerobot/pi0` to the local HF cache and return its path.

    lerobot types `policy.pretrained_path` as `Path`, which on Windows converts
    forward-slash repo ids ("lerobot/pi0") into backslash WindowsPaths
    ("lerobot\\pi0") that HF Hub then rejects. Resolving to a local snapshot
    directory sidesteps that.
    """
    from huggingface_hub import snapshot_download
    local_dir = Path(snapshot_download(repo_id="lerobot/pi0"))
    return local_dir


def build_args(method: str, policy_type: str = "pi0") -> tuple[list[str], Path]:
    cfg = METHODS[method]
    output_dir = REPO_ROOT / "outputs" / "train" / f"pi0_3d_printer_{method}"
    pi0_path = ensure_pi0_weights()

    base = [
        # policy
        f"--policy.type={policy_type}",
        f"--policy.discover_packages_path=lerobot.policies.{policy_type}",
        f"--policy.pretrained_path={pi0_path}",
        "--policy.push_to_hub=false",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        # dataset (local)
        "--dataset.repo_id=local/open_3d_printer_diversified",
        f"--dataset.root={DATASET_ROOT}",
        # training (per-method defaults)
        f"--steps={cfg['default_steps']}",
        f"--batch_size={cfg['default_batch_size']}",
        "--num_workers=4",
        "--save_freq=5000",
        "--log_freq=100",
        "--optimizer.type=adamw",
        f"--optimizer.lr={cfg['default_lr']}",
        f"--output_dir={output_dir}",
        f"--job_name=pi0_3d_printer_{method}",
        "--wandb.enable=false",
    ]
    return base + cfg["extra_args"], output_dir


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(
        "--method",
        choices=list(METHODS.keys()),
        default="full",
        help="Fine-tuning paradigm: full | lora | frozen.",
    )
    ap.add_argument("-h", "--help", action="store_true")
    known, passthrough = ap.parse_known_args()

    if known.help:
        print(__doc__)
        print("\nAny extra flags are forwarded to `python -m lerobot.scripts.train`.")
        print("Run `python -m lerobot.scripts.train --help` to see the full list.")
        return 0

    # HF datasets builds long cache filenames (260+ chars on Windows) by
    # embedding the absolute source path as a hash key. If HF_HOME lives under
    # this already-deep project root, lockfile creation fails with FileNotFound
    # and lerobot falls back to remote lookup -> 401. Pin a short path.
    if sys.platform == "win32":
        os.environ.setdefault("HF_HOME", r"D:\.hfcache")
    else:
        os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
    # Detect policy type from passthrough or default to pi0
    policy_type = "pi0"
    for i, arg in enumerate(passthrough):
        if arg.startswith("--policy.type="):
            policy_type = arg.split("=", 1)[1]
        elif arg == "--policy.type" and i + 1 < len(passthrough):
            policy_type = passthrough[i+1]

    args, output_dir = build_args(known.method, policy_type=policy_type)
    # lerobot refuses to start if output_dir exists and resume=false. If the
    # directory is empty (e.g. left behind by a prior crash), remove it; if it
    # has real artifacts, bail out so the user can decide.
    if output_dir.exists():
        if any(output_dir.iterdir()):
            print(f"error: {output_dir} is non-empty. Delete it or pass --resume=true.", file=sys.stderr)
            return 1
        output_dir.rmdir()

    # New lerobot (≥ 0.5? — package re-org) renamed the train entrypoint to
    # `lerobot.scripts.lerobot_train`. Older versions used `lerobot.scripts.train`.
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train", *args, *passthrough]
    print(f"==> method: {known.method}")
    print(f"==> output: {output_dir}")
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
