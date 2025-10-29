"""Utilities for loading synthetic data for experimentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import torch


@dataclass
class SyntheticTextDataset:
    """Infinite iterator that yields random token sequences.

    Each yielded tensor has ``sequence_length + 1`` tokens so that callers can
    perform next-token prediction by shifting the batch once for inputs and
    once for targets without having to manually account for the off-by-one.
    """

    vocab_size: int
    sequence_length: int
    device: torch.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        length = self.sequence_length + 1
        while True:
            yield torch.randint(
                low=0,
                high=self.vocab_size,
                size=(length,),
                device=self.device,
            )


def chunk_iterator(
    iterator: Iterable[torch.Tensor],
    batch_size: int,
) -> Iterator[torch.Tensor]:
    """Groups elements of an iterator into fixed-size batches."""

    batch: list[torch.Tensor] = []
    for element in iterator:
        batch.append(element)
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0)
            batch.clear()
