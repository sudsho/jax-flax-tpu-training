.PHONY: help install install-tpu install-torch-xla test lint fmt bench train-jax train-torch clean

help:
	@echo "targets:"
	@echo "  install         core deps (cpu jax)"
	@echo "  install-tpu     jax with TPU wheels"
	@echo "  install-torch-xla  install torch + torch-xla for baseline"
	@echo "  test            run pytest"
	@echo "  lint            ruff check"
	@echo "  fmt             ruff format"
	@echo "  train-jax       run jax training loop (single host cpu by default)"
	@echo "  train-torch     run torch/xla baseline"
	@echo "  bench           run throughput comparison"

install:
	pip install -e ".[dev]"

install-tpu:
	pip install "jax[tpu]==0.4.35" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
	pip install -e ".[dev]"

install-torch-xla:
	pip install -e ".[torch-xla,dev]"

test:
	pytest tests/ -q

lint:
	ruff check src tests

fmt:
	ruff format src tests

train-jax:
	python -m src.training.train_jax --config configs/single_host_cpu.yaml

train-torch:
	python -m src.training.train_torch_xla --config configs/torch_xla_v4_8.yaml

bench:
	bash scripts/run_bench.sh

clean:
	rm -rf __pycache__ .pytest_cache dist build *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
