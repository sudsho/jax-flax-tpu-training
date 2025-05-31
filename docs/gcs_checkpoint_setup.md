# GCS checkpoint setup

Notes for running orbax checkpoints against a Google Cloud Storage bucket.

## 1. Make the bucket

```bash
export PROJECT=my-gcp-project
export BUCKET=vit-jax-checkpoints
export REGION=us-central1

gcloud storage buckets create gs://$BUCKET \
    --project=$PROJECT \
    --location=$REGION \
    --uniform-bucket-level-access
```

Same-region as your TPU slice matters. Cross-region writes cost bandwidth
and add latency to every checkpoint step.

## 2. Grant the TPU service account write access

TPU VMs run as a service account of the form
`service-{PROJECT_NUMBER}@cloud-tpus.iam.gserviceaccount.com`. Grant it
`roles/storage.objectAdmin` on the bucket:

```bash
gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
    --member=serviceAccount:service-${PROJECT_NUMBER}@cloud-tpus.iam.gserviceaccount.com \
    --role=roles/storage.objectAdmin
```

Alternatively use a user-managed SA and attach it with
`--service-account=...` when creating the TPU VM.

## 3. Install gcsfs on the TPU VM

`orbax-checkpoint` uses `gcsfs` when the path starts with `gs://`. Already
in `requirements.txt`, but if you set up the environment by hand:

```bash
pip install gcsfs==2025.2.0
```

## 4. Use gs:// paths directly in code

```python
from src.checkpoint.orbax_ckpt import build_checkpoint_manager

mgr = build_checkpoint_manager(
    "gs://vit-jax-checkpoints/exp-2025-05-13",
    keep_last=3,
    save_interval_steps=1000,
)
mgr.save(step, {"state": state, "opt": opt_state, "ema": ema_state})
```

## 5. Lifecycle rules to keep costs down

Add a lifecycle rule that deletes objects older than 30 days from
uncleaned experiment prefixes:

```bash
cat > /tmp/lifecycle.json <<'EOF'
{
  "rule": [
    {"action": {"type": "Delete"},
     "condition": {"age": 30, "matchesPrefix": ["experiments/scratch/"]}}
  ]
}
EOF

gcloud storage buckets update gs://$BUCKET --lifecycle-file=/tmp/lifecycle.json
```

## 5b. Multi-host slice caveat

On a v4-32 or larger slice orbax writes from every host process. Point
every host at the same `gs://` prefix; orbax coordinates writes with an
internal barrier. If you set a *local* directory instead, each host
writes to its own worker disk, and restore silently loads whichever
happens to be on rank 0 which is almost never what you want.

## 6. Cost note

Standard storage at $0.020/GB-month. A single ViT-B/16 fp32 checkpoint is
~350 MB (~90M params x 4 bytes). Three checkpoints for 12 experiments
= 12 GB, about $0.25/month. Cheap.
