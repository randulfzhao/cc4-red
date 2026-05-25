#!/usr/bin/env bash

set -euo pipefail

# Run from the blue trainer directory so its checkpoints land beside the
# train_marl.py module, where Ray_BlueAgent expects them by default.
cd "$(dirname "${BASH_SOURCE[0]}")/../../Hierarchical-MARL-main/marl-1policy" || exit 1

python train_marl.py "$@"
