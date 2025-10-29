from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil

try:
    import torch  # type: ignore
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional deps
    torch = None  # type: ignore
    pynvml = None  # type: ignore


@dataclass
class BenchRecord:
    framework: str
    device: str
    precision: str
    mode: str  # train|infer|forward
    seq_len: int
    batch_size: int
    steps: int
    tokens_per_step: int
    ms_per_step: float
    tokens_per_sec: float
    mem_alloc_mb: float
    mem_reserved_mb: float
    peak_mem_mb: float
    params_millions: float
    notes: str = ""


class Logger:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = Path(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.jsonl_path = self.out_dir / "bench_results.jsonl"
        self.jsonl = open(self.jsonl_path, "a", buffering=1, encoding="utf-8")
        self.csv_path = self.out_dir / "bench_results.csv"
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "framework,device,precision,mode,seq_len,batch_size,steps,tokens_per_step,ms_per_step,"
                    "tokens_per_sec,mem_alloc_mb,mem_reserved_mb,peak_mem_mb,params_millions,notes\n"
                )

    def log(self, rec: BenchRecord) -> None:
        self.jsonl.write(json.dumps(asdict(rec)) + "\n")
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{rec.framework},{rec.device},{rec.precision},{rec.mode},{rec.seq_len},{rec.batch_size},{rec.steps},"
                f"{rec.tokens_per_step},{rec.ms_per_step:.4f},{rec.tokens_per_sec:.2f},{rec.mem_alloc_mb:.2f},"
                f"{rec.mem_reserved_mb:.2f},{rec.peak_mem_mb:.2f},{rec.params_millions:.2f},{rec.notes}\n"
            )

    def close(self) -> None:
        if not self.jsonl.closed:
            self.jsonl.close()

    def archive(self, destination: str | os.PathLike[str] | None = None) -> Path:
        """Close log files and package the output directory into a zip archive."""

        self.close()
        base_name: Path
        if destination is None:
            base_name = self.out_dir
        else:
            base_name = Path(destination)
        if base_name.suffix == ".zip":
            archive_base = base_name.with_suffix("")
        else:
            archive_base = base_name
        archive_base.parent.mkdir(parents=True, exist_ok=True)
        archive_path = Path(
            shutil.make_archive(str(archive_base), "zip", root_dir=self.out_dir)
        )
        if base_name.suffix == ".zip":
            return archive_path.with_suffix(".zip")
        return archive_path


def gpu_mem_mb() -> tuple[float, float, float]:
    if torch is None or not torch.cuda.is_available():  # type: ignore[union-attr]
        return (0.0, 0.0, 0.0)
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return (alloc, reserved, peak)
