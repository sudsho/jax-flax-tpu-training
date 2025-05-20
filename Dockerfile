# Multi-target Dockerfile.
#
# CPU dev image:
#     docker build -t jax-vit-cpu --target cpu .
#
# TPU-VM image (jax with libtpu):
#     docker build -t jax-vit-tpu --target tpu .

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# ---- CPU target ------------------------------------------------------------
FROM base AS cpu
RUN pip install -r requirements.txt
CMD ["python", "-m", "src.training.train_jax", "--config", "configs/single_host_cpu.yaml"]

# ---- TPU-VM target ---------------------------------------------------------
FROM base AS tpu
RUN pip install -r requirements.txt \
    && pip install "jax[tpu]==0.4.35" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
ENV LIBTPU_INIT_ARGS=""
CMD ["python", "-m", "src.training.train_jax", "--config", "configs/jax_tpu_v4_8.yaml"]
