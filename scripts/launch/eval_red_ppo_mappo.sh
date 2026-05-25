#!/usr/bin/env bash
# Clean CC4 red PPO eval launcher.
# Algorithm: mappo    Default checkpoint: red_checkpoints/ppo/mappo/best

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

set -euo pipefail

RED_POLICY_TAG="${RED_POLICY_TAG:-best}"
RED_POLICY_DIR="${RED_POLICY_DIR:-red_checkpoints/ppo/mappo/${RED_POLICY_TAG}}"
EPISODES="${EPISODES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_output/red_ppo_mappo_${RED_POLICY_TAG}}"
BLUE_OPPONENT="${BLUE_OPPONENT:-marl}"
BLUE_CHECKPOINT_ROOT="${BLUE_CHECKPOINT_ROOT:-Hierarchical-MARL-main/marl-1policy/models/train_marl}"
BLUE_CHECKPOINT_ITER="${BLUE_CHECKPOINT_ITER:-iter_199}"

python scripts/eval/eval_red_ppo.py \
  --RL_TRAIN_ALG mappo \
  --red-policy-dir "${RED_POLICY_DIR}" \
  --episodes "${EPISODES}" \
  --output-dir "${OUTPUT_DIR}" \
  --blue-opponent "${BLUE_OPPONENT}" \
  --blue-checkpoint-root "${BLUE_CHECKPOINT_ROOT}" \
  --blue-checkpoint-iter "${BLUE_CHECKPOINT_ITER}" \
  "$@"
