# Benchmark Suite

Drop-in scripts to benchmark the `superintelligence` PyTorch model (and optional JAX adapter) for throughput, latency, and memory. Results are emitted as JSONL/CSV artifacts that can be merged into existing metric registries.

## What is measured?

- Training and inference tokens/sec, ms/step, and latency sweeps.
- Allocated/reserved/peak memory alongside parameter counts.
- Precision, sequence length, and batch size sweeps.
- Simple numerical stability signals (loss evolution, grad norms, NaN detection) can be layered on top of the produced artifacts.

## Repository layout

```
benchmarks/
  bench_pytorch.py
  bench_jax.py
  utils.py
  README.md
```

## Environment matrix (example)

```yaml
hardware:
  gpu: ["A100 80GB", "RTX 4090 24GB", "T4 16GB", "CPU-only"]
software:
  torch: ">=2.2"
  cuda: ">=12.1"
  jax: ">=0.4.30"
  python: ">=3.10"
```

## Running the PyTorch benchmarks

Install dependencies first (PyTorch and optional `pynvml` for VRAM stats):

```bash
pip install -r ../requirements.txt
```

```bash
# GPU sweep
python benchmarks/bench_pytorch.py --device cuda:0 --steps 2000 \
  --seq 128 512 2048 --batch 8 32 --precision bf16 --out _bench

# CPU sanity
python benchmarks/bench_pytorch.py --device cpu --steps 200 --seq 128 \
  --batch 4 --precision fp32 --out _bench_cpu
```

To generate synthetic placeholder metrics (for environments without PyTorch),
add the `--sandbox` flag:

```bash
python benchmarks/bench_pytorch.py --sandbox --device cpu --seq 128 --batch 4 --steps 10
```

Add `--zip` to compress the output directory (e.g. `_bench.zip`) once logging
completes:

```bash
python benchmarks/bench_pytorch.py --sandbox --device cpu --seq 128 --batch 4 --steps 10 --zip
```

## Running the JAX placeholder benchmarks

Replace `simple_forward` with your QuASIM/Flax model call when ready.

```bash
python benchmarks/bench_jax.py --platform gpu --steps 2000 \
  --seq 128 512 2048 --batch 8 32 --precision bf16 --out _bench_jax
```

## Output integration

Append or merge the generated `bench_results.jsonl` file into your V24.x metrics registry. A suggested bridge entry:

```yaml
- id: metric:fp_train_tps
  name: Tokens/sec (train)
  compare: higher_is_better
  source: benchmarks/bench_pytorch.py
  tags: [fuzzy-parakeet, v24.x, pytorch]
```

To surface live metrics inside QVR HUDs, tail `bench_results.jsonl` and expose toggle hotkeys for `batch`, `seq`, and `precision` variants.
