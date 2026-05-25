#!/usr/bin/env bash
# Clean CC4 red PPO launcher.
# Algorithm: ippo    Checkpoint: red_checkpoints/ppo/ippo

# Run from repo root so all relative paths resolve correctly.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

set -euo pipefail

python scripts/train/train_red_ppo.py \
  --RL_TRAIN_ALG="ippo" \
  --RL_NUM_ROLLOUT_WORKERS=4 \
  --RL_NUM_ITERATIONS=100 \
  \
  "$@"
