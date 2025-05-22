#!/usr/bin/env bash
# Launch the torch-xla baseline. Requires torch + torch-xla installed.
set -euo pipefail

CONFIG="${1:-configs/torch_xla_v4_8.yaml}"

# torch-xla env recommended defaults for TPU v4
export PJRT_DEVICE=TPU
export XLA_USE_BF16=0

python -m src.training.train_torch_xla --config "$CONFIG"
