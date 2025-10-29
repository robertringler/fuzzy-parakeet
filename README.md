# Fuzzy Parakeet Superintelligence Toolkit

This repository provides a research-oriented PyTorch implementation of a
Transformer-style language model featuring rotary embeddings, root-mean-square
normalisation, and a sparse mixture-of-experts feed-forward network. The goal is
to offer a clean and easily-extensible codebase rather than claim any specific
capabilities.

## Requirements

Install PyTorch (CPU or CUDA build) alongside the minimal Python dependencies:

```bash
pip install -r requirements.txt
```

Refer to the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
if you need wheels for specific CUDA versions.

## Getting started

```bash
python -m superintelligence.cli --steps 5 --sequence-length 128 --device cpu
```

This command will run a short synthetic training loop to verify the plumbing
between the model, data iterator, and trainer components.

## Benchmarking

To capture throughput, latency, and memory metrics for the reference model,
use the scripts under `benchmarks/`:

```bash
# GPU sweep
python benchmarks/bench_pytorch.py --device cuda:0 --steps 2000 \
  --seq 128 512 2048 --batch 8 32 --precision bf16 --out _bench

# CPU sanity check
python benchmarks/bench_pytorch.py --device cpu --steps 200 --seq 128 \
  --batch 4 --precision fp32 --out _bench_cpu
```

If you are operating in a restricted environment without PyTorch wheels,
append `--sandbox` to emit deterministic placeholder metrics:

```bash
python benchmarks/bench_pytorch.py --sandbox --device cpu --seq 128 --batch 4 --steps 10
```

To bundle the generated artifacts into a zip archive, include `--zip`. The
archive (for example `_bench.zip`) is written next to the output directory.

```bash
python benchmarks/bench_pytorch.py --sandbox --device cpu --seq 128 --batch 4 --steps 10 --zip
```

Additional details, including the optional JAX adapter and output schemas, are
documented in `benchmarks/README.md`.
