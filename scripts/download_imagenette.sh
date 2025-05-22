#!/usr/bin/env bash
# Pre-download imagenette so the training loop doesn't wait on tfds on first run.
set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/tensorflow_datasets}"

python - <<PY
import os
os.environ.setdefault("TFDS_DATA_DIR", "$DATA_DIR")
import tensorflow_datasets as tfds
print("downloading imagenette/320px-v2 to $DATA_DIR")
b = tfds.builder("imagenette/320px-v2", data_dir="$DATA_DIR")
b.download_and_prepare()
info = b.info
print("splits:", {k: v.num_examples for k, v in info.splits.items()})
PY

echo "done. dataset lives at $DATA_DIR"
