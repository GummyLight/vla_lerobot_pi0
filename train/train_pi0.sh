#!/usr/bin/env bash
# Launch pi0 fine-tuning. First arg can be a method (full|lora|frozen);
# all remaining args are forwarded to lerobot's train CLI.
#
#   bash train/train_pi0.sh                 # full SFT (default)
#   bash train/train_pi0.sh lora            # LoRA
#   bash train/train_pi0.sh frozen --steps=10000
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Defer everything to the Python launcher so method-specific defaults stay in
# one place.
METHOD="full"
if [[ $# -ge 1 && "$1" =~ ^(full|lora|frozen)$ ]]; then
    METHOD="$1"; shift
fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
exec python train/train_pi0.py --method="$METHOD" "$@"
# (train_pi0.py shells out to `python -m lerobot.scripts.lerobot_train`.)
