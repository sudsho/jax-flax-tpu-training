#!/usr/bin/env bash
# Launch the JAX training loop with the tpu v4-8 config.
set -euo pipefail

CONFIG="${1:-configs/jax_tpu_v4_8.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-jax-vit-tpu}"

python -m src.training.train_jax \
    --config "$CONFIG" \
    --out-dir outputs \
    --seed 42 \
    --wandb "$@"
