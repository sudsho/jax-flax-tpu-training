#!/usr/bin/env bash
# Provision a TPU v4-8 VM and install jax + repo deps.
#
# Prereqs: gcloud auth + project set, and $TPU_NAME / $ZONE exported.
set -euo pipefail

: "${TPU_NAME:?set TPU_NAME (e.g. vit-jax-v4-8)}"
: "${ZONE:?set ZONE (e.g. us-central2-b)}"
ACCELERATOR="${ACCELERATOR:-v4-8}"
RUNTIME="${RUNTIME:-tpu-vm-v4-base}"

echo ">>> creating TPU VM $TPU_NAME ($ACCELERATOR) in $ZONE"
gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$ZONE" \
    --accelerator-type="$ACCELERATOR" \
    --version="$RUNTIME"

echo ">>> installing deps on the TPU VM"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command "
set -e
sudo apt-get update -y
sudo apt-get install -y git python3-venv
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install 'jax[tpu]==0.4.35' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax==0.10.2 optax==0.2.4 chex==0.1.87 \
            orbax-checkpoint==0.10.1 tensorflow-datasets==4.9.7 \
            wandb==0.19.6 numpy==2.2.4 pillow==11.1.0 gcsfs==2025.2.0 \
            tqdm==4.67.1 pyyaml pytest==8.3.4
python -c 'import jax; print(\"devices:\", jax.devices())'
"

echo ">>> done. rsync the repo with:"
echo "gcloud compute tpus tpu-vm scp --recurse --zone $ZONE . $TPU_NAME:~/jax-flax-tpu-training"
