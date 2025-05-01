# jax-flax-tpu-training

JAX + Flax reimplementation of ViT-B/16 pretraining on TPU v4.

Just getting started. Goal: reproduce ViT-B/16 training with Flax NNX on TPU
v4-8 with pjit mesh sharding, checkpointed to GCS, and compare throughput
against a PyTorch/XLA baseline on the same slice.
