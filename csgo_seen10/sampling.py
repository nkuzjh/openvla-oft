"""Optimizer-update batches with GPU-independent sample exposure."""

from __future__ import annotations

import hashlib

import torch
from torch.utils.data import Sampler


def event_steps(max_steps: int, interval: int) -> list[int]:
    """Include the final update even when it is not a multiple of interval."""
    if max_steps <= 0 or interval <= 0:
        raise ValueError("max_steps and event interval must be positive")
    return sorted(set(range(interval, max_steps + 1, interval)) | {max_steps})


class GlobalUpdateSampler(Sampler):
    """Form full global batches first, then shard each batch across ranks.

    Yields (record_index, epoch) so augmentation is independent of worker
    lifetime/prefetching. Each rank receives a contiguous slice of every
    logical update. No duplicate padding, partial update, or tail forward.
    """

    def __init__(self, dataset, *, effective_batch_size: int, microbatch_size: int,
                 accumulation_steps: int, rank: int, world_size: int, seed: int):
        self.size = len(dataset)
        self.effective_batch_size = int(effective_batch_size)
        self.rank, self.world_size = int(rank), int(world_size)
        self.seed, self.epoch = int(seed), 0
        self.microbatch_size = int(microbatch_size)
        self.accumulation_steps = int(accumulation_steps)
        self.start_batch = 0
        if min(self.effective_batch_size, microbatch_size, accumulation_steps, self.world_size) <= 0:
            raise ValueError("Batch, accumulation and world size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid rank")
        if microbatch_size * accumulation_steps * self.world_size != self.effective_batch_size:
            raise ValueError("microbatch * accumulation * world_size must equal effective_batch_size")
        self.updates_per_epoch = self.size // self.effective_batch_size
        if not self.updates_per_epoch:
            raise ValueError("Dataset is smaller than one complete global update")
        self.used_per_epoch = self.updates_per_epoch * self.effective_batch_size
        self.dropped_per_epoch = self.size - self.used_per_epoch
        self.local_update_size = self.effective_batch_size // self.world_size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)
        self.start_batch = 0

    def set_start_batch(self, batch: int) -> None:
        """Skip completed updates before dataset access, preserving epoch order.

        Checkpoints store a per-rank microbatch offset into the *full* epoch.
        Augmentation is keyed by epoch/sample rather than iterator consumption.
        """
        full_batches = self.used_per_epoch // self.world_size // self.microbatch_size
        if batch < 0 or batch > full_batches or batch % self.accumulation_steps:
            raise ValueError(
                f"Resume batch {batch} must be an update boundary in [0, {full_batches}] "
                f"(accumulation={self.accumulation_steps})"
            )
        self.start_batch = int(batch)

    def global_indices(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.size, generator=generator)[:self.used_per_epoch]

    def epoch_audit(self) -> dict:
        indices = self.global_indices().numpy().astype("<i8", copy=False)
        return {"epoch": self.epoch, "updates": self.updates_per_epoch,
                "used": self.used_per_epoch, "dropped": self.dropped_per_epoch,
                "global_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}

    def __iter__(self):
        batches = self.global_indices().reshape(-1, self.effective_batch_size)
        start = self.rank * self.local_update_size
        local = batches[:, start:start + self.local_update_size].reshape(-1).tolist()
        return iter((index, self.epoch) for index in local[self.start_batch * self.microbatch_size:])

    def __len__(self):
        return self.used_per_epoch // self.world_size - self.start_batch * self.microbatch_size
