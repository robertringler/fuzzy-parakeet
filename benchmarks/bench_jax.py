from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp


def simple_forward(x: jnp.ndarray) -> jnp.ndarray:
    w1 = jnp.ones((x.shape[-1], x.shape[-1]), dtype=x.dtype) * 0.01
    w2 = jnp.ones((x.shape[-1], x.shape[-1]), dtype=x.dtype) * 0.01
    return jnp.tanh(x @ w1) @ w2


@jax.jit
def step(x: jnp.ndarray) -> jnp.ndarray:
    return simple_forward(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX placeholder benchmark")
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--batch", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out", default="_bench_jax")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", args.platform)

    for seq_len in args.seq:
        for batch_size in args.batch:
            dtype = jnp.bfloat16 if args.precision == "bf16" else jnp.float32
            x = jnp.ones((batch_size, seq_len, 256), dtype=dtype)
            step(x).block_until_ready()
            timings: list[float] = []
            for _ in range(args.steps):
                t0 = time.perf_counter()
                _ = step(x).block_until_ready()
                timings.append((time.perf_counter() - t0) * 1e3)
            ms = sum(timings) / len(timings)
            tokens = batch_size * seq_len
            print(f"jax,seq={seq_len},batch={batch_size},ms={ms:.3f},tok/s={tokens / (ms / 1000):.1f}")


if __name__ == "__main__":
    main()
