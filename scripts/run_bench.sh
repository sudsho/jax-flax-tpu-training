#!/usr/bin/env bash
# Run both JAX and torch-xla throughput benches, save under benchmarks/.
set -euo pipefail

STEPS="${STEPS:-100}"
BATCH="${BATCH:-512}"
PER_CORE="${PER_CORE:-64}"
CORES="${CORES:-8}"

mkdir -p benchmarks

echo ">>> JAX bench (steps=$STEPS batch=$BATCH)"
python -m src.bench.throughput \
    --framework jax --batch-size "$BATCH" --steps "$STEPS" \
    --out "benchmarks/jax_$(date +%Y%m%d_%H%M%S).json"

echo ">>> torch-xla bench (steps=$STEPS per-core=$PER_CORE cores=$CORES)"
python -m src.bench.throughput \
    --framework torch-xla \
    --batch-size-per-core "$PER_CORE" \
    --num-cores "$CORES" \
    --steps "$STEPS" \
    --out "benchmarks/torch_xla_$(date +%Y%m%d_%H%M%S).json"

echo ">>> done. see benchmarks/results.md for the summary table"
