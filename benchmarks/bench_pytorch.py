from __future__ import annotations

import argparse
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
import math
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in sandbox mode
    torch = None  # type: ignore[assignment]

from benchmarks.utils import BenchRecord, Logger, gpu_mem_mb

if torch is not None:
    from superintelligence.config import ModelConfig
    from superintelligence.dataset import SyntheticTextDataset, chunk_iterator
    from superintelligence.model import SuperIntelligenceModel


def count_params(model: "torch.nn.Module") -> int:
    return sum(p.numel() for p in model.parameters())


def run_pass(
    model: "torch.nn.Module",
    inputs: "torch.Tensor",
    targets: "torch.Tensor" | None = None,
    train: bool = False,
    loss_fn: "torch.nn.Module" | None = None,
    optim: "torch.optim.Optimizer" | None = None,
) -> tuple[float, float | None]:
    if train and optim is not None:
        optim.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    logits = model(inputs)
    loss_val: float | None = None
    if train and targets is not None and loss_fn is not None and optim is not None:
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optim.step()
        loss_val = float(loss.item())
    if inputs.is_cuda:
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3
    return dt_ms, loss_val


def autocast_context(device: "torch.device", dtype: "torch.dtype"):
    if device.type == "cuda" and dtype == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)  # type: ignore[attr-defined]
    return nullcontext()


@dataclass
class SandboxArgs:
    device: str
    precision: str
    seq: list[int]
    batch: list[int]
    steps: int
    warmup: int
    out: str
    zip_results: bool


def sandbox_run(args: SandboxArgs) -> None:
    """Emit deterministic, fake benchmark numbers when PyTorch is unavailable."""

    logger = Logger(args.out)
    rng = random.Random(1337)

    for seq_len in args.seq:
        # Pretend parameter count grows linearly with sequence length.
        params_m = 650 + 0.12 * (seq_len / 128)

        for batch_size in args.batch:
            tokens = batch_size * seq_len
            base_ms = 12.5 + math.log1p(seq_len) * 3 + math.log1p(batch_size) * 5
            jitter = rng.uniform(-0.8, 0.8)

            for mode, scale in ("train", 1.35), ("infer", 0.85):
                ms = (base_ms + jitter) * scale
                tok_per_sec = tokens / (ms / 1000)
                logger.log(
                    BenchRecord(
                        framework="pytorch-sandbox",
                        device=args.device,
                        precision=args.precision,
                        mode=mode,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        steps=args.steps,
                        tokens_per_step=tokens,
                        ms_per_step=ms,
                        tokens_per_sec=tok_per_sec,
                        mem_alloc_mb=0.0,
                        mem_reserved_mb=0.0,
                        peak_mem_mb=0.0,
                        params_millions=params_m,
                        notes="sandboxed synthetic metrics",
                    )
                )

    if args.zip_results:
        archive_path = logger.archive(Path(args.out).with_suffix(".zip"))
        print(f"Sandbox metrics archived to {archive_path}")
    else:
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch SuperIntelligence model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--batch", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", default="_bench")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Emit deterministic synthetic metrics without requiring PyTorch",
    )
    parser.add_argument(
        "--zip",
        dest="zip_results",
        action="store_true",
        help="Archive the output directory into a .zip after the run completes",
    )
    args = parser.parse_args()

    if args.sandbox:
        sandbox_run(
            SandboxArgs(
                device=args.device,
                precision=args.precision,
                seq=list(args.seq),
                batch=list(args.batch),
                steps=args.steps,
                warmup=args.warmup,
                out=args.out,
                zip_results=args.zip_results,
            )
        )
        return

    if torch is None:
        parser.error(
            "PyTorch is not installed. Install the dependency or rerun with --sandbox for stub metrics."
        )

    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" and device.type == "cuda" else torch.float32

    logger = Logger(args.out)

    for seq_len in args.seq:
        cfg = ModelConfig(max_sequence_length=seq_len)
        model = SuperIntelligenceModel(cfg).to(device)
        params_m = count_params(model) / 1e6

        for batch_size in args.batch:
            dataset = SyntheticTextDataset(cfg.vocab_size, seq_len, device=device)
            iterator = chunk_iterator(iter(dataset), batch_size)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            loss_fn = torch.nn.CrossEntropyLoss()

            model.train()
            for _ in range(args.warmup):
                batch = next(iterator)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                with autocast_context(device, amp_dtype):
                    _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            timings: list[float] = []
            for _ in range(args.steps):
                batch = next(iterator)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                with autocast_context(device, amp_dtype):
                    dt_ms, _ = run_pass(
                        model,
                        inputs,
                        targets,
                        train=True,
                        loss_fn=loss_fn,
                        optim=optimizer,
                    )
                timings.append(dt_ms)
            ms = statistics.fmean(timings)
            tokens = batch_size * seq_len
            tok_per_sec = tokens / (ms / 1000)
            alloc, reserved, peak = gpu_mem_mb()
            logger.log(
                BenchRecord(
                    framework="pytorch",
                    device=str(device),
                    precision=args.precision,
                    mode="train",
                    seq_len=seq_len,
                    batch_size=batch_size,
                    steps=args.steps,
                    tokens_per_step=tokens,
                    ms_per_step=ms,
                    tokens_per_sec=tok_per_sec,
                    mem_alloc_mb=alloc,
                    mem_reserved_mb=reserved,
                    peak_mem_mb=peak,
                    params_millions=params_m,
                )
            )

            model.eval()
            timings = []
            with torch.no_grad():
                for _ in range(args.steps):
                    batch = next(iterator)
                    inputs = batch[:, :-1]
                    with autocast_context(device, amp_dtype):
                        dt_ms, _ = run_pass(model, inputs, train=False)
                    timings.append(dt_ms)
            ms = statistics.fmean(timings)
            tok_per_sec = tokens / (ms / 1000)
            alloc, reserved, peak = gpu_mem_mb()
            logger.log(
                BenchRecord(
                    framework="pytorch",
                    device=str(device),
                    precision=args.precision,
                    mode="infer",
                    seq_len=seq_len,
                    batch_size=batch_size,
                    steps=args.steps,
                    tokens_per_step=tokens,
                    ms_per_step=ms,
                    tokens_per_sec=tok_per_sec,
                    mem_alloc_mb=alloc,
                    mem_reserved_mb=reserved,
                    peak_mem_mb=peak,
                    params_millions=params_m,
                )
            )

    if args.zip_results:
        archive_path = logger.archive(Path(args.out).with_suffix(".zip"))
        print(f"Benchmark metrics archived to {archive_path}")
    else:
        logger.close()


if __name__ == "__main__":
    main()
