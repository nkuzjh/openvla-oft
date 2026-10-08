"""CPU checks for single-rank expansion; no pretrained model or GPU loading."""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from csgo_seen10.resume import remap_resume_position, validate_resume_settings
from csgo_seen10.sampling import GlobalUpdateSampler
from csgo_seen10.runner import (
    _restore_rng_state,
    _restore_single_rank_rng_state,
    _save_rng_state,
)


POLICY = "global_full_update_batches"


def settings(world=1, micro=128, accumulation=1):
    return {
        "world_size": world, "batch_size": micro,
        "grad_accumulation_steps": accumulation, "seed": 42,
        "learning_rate": 0.0005, "num_steps_before_decay": 100000,
        "event_every": 4000, "smoke": False,
    }


class TopologySettingsTests(unittest.TestCase):
    def test_expansion_requires_equal_actual_products(self):
        old = settings()
        for new in (settings(2, 64, 1), settings(2, 32, 2), settings(4, 16, 2)):
            self.assertTrue(validate_resume_settings(old, new, sampler_policy=POLICY))
        with self.assertRaisesRegex(ValueError, "effective batch"):
            validate_resume_settings(old, settings(2, 128, 1), sampler_policy=POLICY)

    def test_single_rank_and_same_topology_remain_strict(self):
        for old in (settings(), settings(2, 32, 2)):
            self.assertFalse(validate_resume_settings(old, old.copy(), sampler_policy=POLICY))
            new = dict(old, batch_size=old["batch_size"] // 2,
                       grad_accumulation_steps=old["grad_accumulation_steps"] * 2)
            with self.assertRaisesRegex(ValueError, "training settings differ"):
                validate_resume_settings(old, new, sampler_policy=POLICY)
        with self.assertRaisesRegex(ValueError, "training settings differ"):
            validate_resume_settings(settings(2, 64, 1), settings(), sampler_policy=POLICY)

    def test_other_training_identity_is_not_relaxed(self):
        for key, value in (("seed", 43), ("learning_rate", 0.001),
                           ("event_every", 3900), ("num_steps_before_decay", 5000),
                           ("smoke", True)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_resume_settings(settings(), dict(settings(2, 32, 2), **{key: value}),
                                         sampler_policy=POLICY)
        with self.assertRaisesRegex(ValueError, "global_full_update_batches"):
            validate_resume_settings(settings(), settings(2, 32, 2), sampler_policy="legacy")
        self.assertFalse(validate_resume_settings(settings(), settings(), sampler_policy="legacy"))

    def test_bad_topology_values_rejected(self):
        for key in ("world_size", "batch_size", "grad_accumulation_steps"):
            for value in (0, -1, None, True, 2.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_resume_settings(settings(), dict(settings(2, 32, 2), **{key: value}),
                                             sampler_policy=POLICY)


class TopologyPositionTests(unittest.TestCase):
    def test_mid_epoch_remaps_accumulation_in_both_directions(self):
        for old, new, old_batch, new_batch in (
            (settings(), settings(2, 32, 2), 100, 200),
            (settings(1, 16, 8), settings(2, 64, 1), 800, 100),
            (settings(), settings(2, 64, 1), 100, 100),
        ):
            state = {"step": 4000, "data_position": {"epoch": 10, "next_batch": old_batch}}
            untouched = copy.deepcopy(state)
            result = remap_resume_position(state, old, new, dataset_size=50000)
            self.assertEqual(result["new_position"], {"epoch": 10, "next_batch": new_batch})
            self.assertEqual(result["updates_per_epoch"], 390)
            self.assertEqual(result["global_step"], 4000)
            self.assertEqual(state, untouched)

    def test_epoch_end_and_final_checkpoint(self):
        for step, old_pos, expected in (
            (0, {"epoch": 0, "next_batch": 0}, {"epoch": 0, "next_batch": 0}),
            (4290, {"epoch": 10, "next_batch": 390}, {"epoch": 11, "next_batch": 0}),
            (19500, {"epoch": 50, "next_batch": 0}, {"epoch": 50, "next_batch": 0}),
        ):
            result = remap_resume_position({"step": step, "data_position": old_pos},
                                           settings(), settings(2, 32, 2), dataset_size=50000)
            self.assertEqual(result["new_position"], expected)

    def test_corrupt_positions_and_partial_updates_are_rejected(self):
        good = {"step": 4000, "data_position": {"epoch": 10, "next_batch": 200}}
        old, new = settings(1, 64, 2), settings(2, 32, 2)
        variants = [
            dict(good, step=4001), dict(good, step=-1), dict(good, step=True),
            dict(good, data_position=None),
            dict(good, data_position={"epoch": -1, "next_batch": 200}),
            dict(good, data_position={"epoch": 10, "next_batch": 199}),
            dict(good, data_position={"epoch": 10, "next_batch": 782}),
            dict(good, data_position={"epoch": 10, "next_batch": "200"}),
        ]
        for state in variants:
            with self.subTest(state=state), self.assertRaises(ValueError):
                remap_resume_position(state, old, new, dataset_size=50000)

    def test_global_update_suffix_matches_across_ranks_and_epochs(self):
        # 49,920 samples/epoch, with a frozen 80-row tail; exact published size.
        dataset = range(50000)
        for world, micro, accum in ((2, 64, 1), (2, 32, 2), (4, 16, 2)):
            old = GlobalUpdateSampler(dataset, effective_batch_size=128, microbatch_size=128,
                                      accumulation_steps=1, rank=0, world_size=1, seed=42)
            new = [GlobalUpdateSampler(dataset, effective_batch_size=128, microbatch_size=micro,
                                       accumulation_steps=accum, rank=r, world_size=world, seed=42)
                   for r in range(world)]
            for epoch, completed_updates in ((10, 100), (11, 0)):
                old.set_epoch(epoch)
                old.set_start_batch(completed_updates)
                expected = list(old)
                shards = []
                for sampler in new:
                    sampler.set_epoch(epoch)
                    sampler.set_start_batch(completed_updates * accum)
                    shards.append(list(sampler))
                local_update = micro * accum
                actual = []
                for start in range(0, len(shards[0]), local_update):
                    for rank in range(world):
                        actual.extend(shards[rank][start:start + local_update])
                self.assertEqual(actual, expected)


class TopologyRNGTests(unittest.TestCase):
    def setUp(self):
        cpu, numpy, python = torch.get_rng_state(), np.random.get_state(), random.getstate()
        self.addCleanup(torch.set_rng_state, cpu)
        self.addCleanup(np.random.set_state, numpy)
        self.addCleanup(random.setstate, python)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        torch.manual_seed(321)
        np.random.seed(321)
        random.seed(321)
        with patch("torch.cuda.is_available", return_value=False):
            _save_rng_state(self.directory, 0)

    def read_payload(self):
        return torch.load(self.directory / "rng_state_rank_0.pt", weights_only=False)

    def write_payload(self, payload):
        torch.save(payload, self.directory / "rng_state_rank_0.pt")

    def test_all_new_ranks_copy_original_cpu_streams(self):
        samples = []
        for rank in range(3):
            torch.manual_seed(900 + rank)
            np.random.seed(900 + rank)
            random.seed(900 + rank)
            _restore_single_rank_rng_state(self.directory, torch.device("cpu"))
            samples.append((torch.rand(5).tolist(), np.random.rand(5).tolist(), random.random()))
        self.assertEqual(samples[0], samples[1])
        self.assertEqual(samples[0], samples[2])

    def test_cuda_state_targets_each_active_device_and_legacy_defaults_to_zero(self):
        for source in (None, 2):
            payload = self.read_payload()
            payload["cuda"] = [torch.tensor([i], dtype=torch.uint8) for i in range(3)]
            if source is None:
                payload.pop("cuda_device_index", None)
            else:
                payload["cuda_device_index"] = source
            self.write_payload(payload)
            for rank in (0, 1):
                device = torch.device("cuda", rank)
                with patch("torch.cuda.set_rng_state") as one, patch("torch.cuda.set_rng_state_all") as all_devices:
                    _restore_single_rank_rng_state(self.directory, device)
                one.assert_called_once()
                self.assertTrue(torch.equal(one.call_args.args[0], payload["cuda"][source or 0]))
                self.assertEqual(one.call_args.kwargs["device"], device)
                all_devices.assert_not_called()

    def test_missing_cuda_state_or_bad_active_index_fails(self):
        base = self.read_payload()
        variants = [base, dict(base, cuda=[]),
                    dict(base, cuda=[torch.zeros(1)], cuda_device_index=1),
                    dict(base, cuda=[torch.zeros(1)], cuda_device_index=-1)]
        for payload in variants:
            self.write_payload(payload)
            with self.assertRaises(ValueError), patch("torch.cuda.set_rng_state"):
                _restore_single_rank_rng_state(self.directory, torch.device("cuda", 1))

    def test_same_topology_still_requires_own_rank_and_restores_all_saved_devices(self):
        with self.assertRaises(FileNotFoundError):
            _restore_rng_state(self.directory, 1, torch.device("cpu"))
        payload = self.read_payload()
        payload["cuda"] = [torch.tensor([7], dtype=torch.uint8)]
        self.write_payload(payload)
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.set_rng_state_all") as all_devices, \
                patch("torch.cuda.set_rng_state") as one:
            _restore_rng_state(self.directory, 0, torch.device("cuda", 0))
        all_devices.assert_called_once()
        self.assertTrue(torch.equal(all_devices.call_args.args[0][0], payload["cuda"][0]))
        one.assert_not_called()

    def test_new_checkpoint_records_active_cuda_device(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_rng_state_all", return_value=[torch.zeros(2, dtype=torch.uint8)]), \
                patch("torch.cuda.current_device", return_value=0):
            _save_rng_state(self.directory, 0)
        self.assertEqual(self.read_payload()["cuda_device_index"], 0)


if __name__ == "__main__":
    unittest.main()
