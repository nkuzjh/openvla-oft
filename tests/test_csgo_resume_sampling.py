"""CPU checks for aligned Seen-10 sampler resume without replaying images."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from csgo_seen10.augmentations import PHOTOMETRIC_POLICY, augment_image
from csgo_seen10.sampling import GlobalUpdateSampler


class CountingDataset(Dataset):
    def __init__(self, size: int):
        self.size = size
        self.reads: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: tuple[int, int]) -> tuple[int, int]:
        self.reads.append(index)
        return index


def make_sampler(dataset: Dataset, *, rank: int, world_size: int,
                 microbatch: int, accumulation: int, epoch: int = 3) -> GlobalUpdateSampler:
    sampler = GlobalUpdateSampler(
        dataset,
        effective_batch_size=microbatch * accumulation * world_size,
        microbatch_size=microbatch,
        accumulation_steps=accumulation,
        rank=rank,
        world_size=world_size,
        seed=17,
    )
    sampler.set_epoch(epoch)
    return sampler


class ResumeSamplingTests(unittest.TestCase):
    def test_dataloader_suffix_and_no_historical_access(self) -> None:
        # Includes the actual aligned single-rank microbatch/accumulation layout.
        layouts = ((1, 128, 1), (1, 3, 4), (2, 2, 3), (4, 1, 3))
        for world_size, microbatch, accumulation in layouts:
            for rank in range(world_size):
                with self.subTest(world_size=world_size, rank=rank,
                                  microbatch=microbatch, accumulation=accumulation):
                    dataset = CountingDataset(257)
                    sampler = make_sampler(dataset, rank=rank, world_size=world_size,
                                           microbatch=microbatch, accumulation=accumulation)
                    collations: list[tuple[tuple[int, int], ...]] = []

                    def collate(samples):
                        batch = tuple(samples)
                        collations.append(batch)
                        return batch

                    loader = DataLoader(dataset, batch_size=microbatch, sampler=sampler,
                                        collate_fn=collate, num_workers=0)
                    full_batches = list(loader)
                    full_length = len(loader)
                    offset = (full_length // accumulation // 2) * accumulation
                    self.assertGreater(offset, 0)
                    expected = full_batches[offset:]
                    expected_rows = [row for batch in expected for row in batch]

                    dataset.reads.clear()
                    collations.clear()
                    sampler.set_start_batch(offset)
                    self.assertEqual(len(loader), full_length - offset)
                    self.assertEqual(list(loader), expected)
                    self.assertEqual(dataset.reads, expected_rows)
                    self.assertEqual(collations, expected)
                    self.assertEqual(sampler.epoch_audit()["used"], sampler.used_per_epoch)

                    # A new epoch must restart at batch zero, including on a reused loader.
                    sampler.set_epoch(4)
                    self.assertEqual(sampler.start_batch, 0)
                    self.assertEqual(len(loader), full_length)
                    next_epoch = list(loader)
                    self.assertEqual(len(next_epoch), full_length)
                    self.assertEqual({row[1] for batch in next_epoch for row in batch}, {4})

    def test_offset_validation_and_epoch_tail(self) -> None:
        dataset = CountingDataset(257)
        sampler = make_sampler(dataset, rank=1, world_size=2, microbatch=2,
                               accumulation=3)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler,
                            collate_fn=tuple, num_workers=0)
        full_batches = len(loader)
        self.assertEqual(full_batches % 3, 0)
        sampler.set_start_batch(full_batches - 3)
        self.assertEqual(len(loader), 3)
        self.assertEqual(len(list(loader)), 3)
        sampler.set_start_batch(full_batches)
        self.assertEqual(len(loader), 0)
        self.assertEqual(list(loader), [])
        for invalid in (-3, 1, full_batches + 3):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sampler.set_start_batch(invalid)

    def test_augmentation_is_epoch_keyed_and_does_not_advance_torch_rng(self) -> None:
        pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        image = Image.fromarray(pixels, mode="RGB")
        torch.manual_seed(91)
        rng_before = torch.get_rng_state().clone()
        first = augment_image(image, policy=PHOTOMETRIC_POLICY, seed=17,
                              epoch=3, sample_id="sample-5", view="fpv")
        same = augment_image(image, policy=PHOTOMETRIC_POLICY, seed=17,
                             epoch=3, sample_id="sample-5", view="fpv")
        later = augment_image(image, policy=PHOTOMETRIC_POLICY, seed=17,
                              epoch=4, sample_id="sample-5", view="fpv")
        self.assertEqual(first.tobytes(), same.tobytes())
        self.assertNotEqual(first.tobytes(), later.tobytes())
        sampler = make_sampler(CountingDataset(257), rank=0, world_size=1,
                               microbatch=3, accumulation=4)
        sampler.set_start_batch(12)
        list(sampler)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_dropout_adamw_resume_matches_replayed_loader(self) -> None:
        dataset = CountingDataset(48)
        torch.manual_seed(99)
        seed_model = nn.Sequential(nn.Linear(2, 8), nn.Dropout(0.35), nn.Linear(8, 1))
        seed_optimizer = torch.optim.AdamW(seed_model.parameters(), lr=0.01)
        seed_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            seed_optimizer, milestones=[3], gamma=0.1,
        )

        def loader_at(start_batch: int) -> DataLoader:
            sampler = make_sampler(dataset, rank=0, world_size=1,
                                   microbatch=2, accumulation=2, epoch=3)
            sampler.set_start_batch(start_batch)

            def collate(rows):
                indices = [index for index, _ in rows]
                x = torch.tensor([[index / 48, (index % 7) / 7] for index in indices],
                                 dtype=torch.float32)
                y = torch.tensor([[index / 48] for index in indices], dtype=torch.float32)
                return x, y

            generator = torch.Generator().manual_seed(1234)
            return DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=collate,
                              num_workers=0, generator=generator)

        def update(model, optimizer, scheduler, batches):
            self.assertEqual(len(batches), 2)
            optimizer.zero_grad(set_to_none=True)
            for x, y in batches:
                loss = (model(x) - y).square().mean()
                (loss / 2).backward()
            optimizer.step()
            scheduler.step()

        initial_loader = iter(loader_at(0))
        for _ in range(2):
            update(seed_model, seed_optimizer, seed_scheduler,
                   [next(initial_loader), next(initial_loader)])
        model_state = copy.deepcopy(seed_model.state_dict())
        optimizer_state = copy.deepcopy(seed_optimizer.state_dict())
        scheduler_state = copy.deepcopy(seed_scheduler.state_dict())
        resumed_rng = torch.get_rng_state().clone()

        results = []
        full_epoch_batches = len(loader_at(0))
        self.assertEqual(full_epoch_batches, 24)
        for replay in (True, False):
            model = nn.Sequential(nn.Linear(2, 8), nn.Dropout(0.35), nn.Linear(8, 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[3], gamma=0.1,
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            scheduler.load_state_dict(copy.deepcopy(scheduler_state))
            torch.set_rng_state(resumed_rng)
            loader = loader_at(0 if replay else 4)
            pending = []
            positions = []
            first_update_inputs = []
            first_forward_rng = None
            for batch_index, batch in enumerate(loader, start=0 if replay else 4):
                if replay and batch_index < 4:
                    continue
                if first_forward_rng is None:
                    first_forward_rng = torch.get_rng_state().clone()
                if len(first_update_inputs) < 2:
                    first_update_inputs.append(batch)
                pending.append(batch)
                if len(pending) < 2:
                    continue
                update(model, optimizer, scheduler, pending)
                pending = []
                next_batch = batch_index + 1
                positions.append((4, 0) if next_batch >= full_epoch_batches else (3, next_batch))
            self.assertEqual(pending, [])
            self.assertEqual(positions[-1], (4, 0))
            results.append((copy.deepcopy(model.state_dict()),
                            copy.deepcopy(optimizer.state_dict()),
                            copy.deepcopy(scheduler.state_dict()),
                            torch.get_rng_state().clone(), positions,
                            first_update_inputs, first_forward_rng))

        for old_batch, new_batch in zip(results[0][5], results[1][5]):
            for first, second in zip(old_batch, new_batch):
                self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(results[0][6], results[1][6]))
        for key in results[0][0]:
            self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]), key)
        old_optimizer, new_optimizer = results[0][1], results[1][1]
        self.assertEqual(old_optimizer["param_groups"], new_optimizer["param_groups"])
        for parameter_id, values in old_optimizer["state"].items():
            for key, value in values.items():
                other = new_optimizer["state"][parameter_id][key]
                self.assertTrue(torch.equal(value, other), (parameter_id, key))
        self.assertEqual(results[0][2], results[1][2])
        self.assertTrue(torch.equal(results[0][3], results[1][3]))
        self.assertEqual(results[0][4], results[1][4])


if __name__ == "__main__":
    unittest.main()
