"""CPU integration check for expanding a completed update from one to two ranks."""

from __future__ import annotations

import datetime
import multiprocessing as mp
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from prismatic.extern.hf.modeling_prismatic import PrismaticCausalLMOutputWithPast

from csgo_seen10.resume import remap_resume_position, validate_resume_settings
from csgo_seen10.sampling import GlobalUpdateSampler


DATASET_SIZE = 35
EFFECTIVE_BATCH = 8
EPOCH = 0
SEED = 73
SAVED_SETTINGS = {"batch_size": 8, "grad_accumulation_steps": 1, "world_size": 1,
                  "seed": SEED}
EXPANDED_SETTINGS = {"batch_size": 2, "grad_accumulation_steps": 2, "world_size": 2,
                     "seed": SEED}


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return DATASET_SIZE

    def __getitem__(self, key: tuple[int, int]) -> int:
        index, epoch = key
        assert epoch == EPOCH
        return index


def _components() -> tuple[nn.Module, torch.optim.Optimizer, object]:
    model = nn.Linear(2, 1, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3], gamma=0.5)
    return model, optimizer, scheduler


def _loader(rank: int, world_size: int, microbatch: int, accumulation: int,
            start_batch: int) -> DataLoader:
    sampler = GlobalUpdateSampler(
        TinyDataset(), effective_batch_size=EFFECTIVE_BATCH,
        microbatch_size=microbatch, accumulation_steps=accumulation,
        rank=rank, world_size=world_size, seed=SEED,
    )
    sampler.set_epoch(EPOCH)
    sampler.set_start_batch(start_batch)
    return DataLoader(TinyDataset(), batch_size=microbatch, sampler=sampler,
                      num_workers=0, generator=torch.Generator().manual_seed(123))


def _features(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack((ids.to(torch.float64) / DATASET_SIZE,
                     (ids % 7).to(torch.float64) / 7), dim=1)
    y = (0.3 + 0.7 * x[:, :1] - 0.2 * x[:, 1:]).square()
    return x, y


def _updates(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: object,
             loader: DataLoader, accumulation: int, limit: int | None = None) -> list[list[int]]:
    groups: list[list[int]] = []
    pending: list[int] = []
    for ids in loader:
        if not pending:
            optimizer.zero_grad(set_to_none=True)
        pending.extend(ids.tolist())
        x, y = _features(ids)
        ((model(x) - y).square().mean() / accumulation).backward()
        if len(pending) != accumulation * loader.batch_size:
            continue
        optimizer.step()
        scheduler.step()
        groups.append(pending)
        pending = []
        if limit is not None and len(groups) >= limit:
            break
    assert not pending
    return groups


def _rng_draws() -> tuple[float, float, float]:
    return (float(torch.rand(())), float(np.random.rand()), float(random.random()))


def _distributed_worker(rank: int, init_file: str, root: str, strict: bool) -> None:
    # Import here so the child uses the production helpers after all process setup.
    from csgo_seen10.runner import _restore_rng_state, _restore_single_rank_rng_state, _save_rng_state

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=30))
    try:
        directory = Path(root)
        checkpoint = directory / ("expanded" if strict else "single")
        payload = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
        model, optimizer, scheduler = _components()
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        ddp = DDP(model)
        expansion = validate_resume_settings(
            payload["settings"], EXPANDED_SETTINGS,
            sampler_policy="global_full_update_batches",
        )
        assert expansion is (not strict)
        if expansion:
            mapped = remap_resume_position(
                payload, payload["settings"], EXPANDED_SETTINGS,
                dataset_size=DATASET_SIZE,
            )
            position = mapped["new_position"]
            assert mapped["global_step"] == 1
            assert mapped["updates_per_epoch"] == 4
            assert mapped["effective_batch_size"] == EFFECTIVE_BATCH
            _restore_single_rank_rng_state(checkpoint, torch.device("cpu"))
        else:
            position = payload["data_position"]
            assert payload["step"] == 2
            _restore_rng_state(checkpoint, rank, torch.device("cpu"))
        draws = _rng_draws()
        loader = _loader(rank, 2, 2, 2, position["next_batch"])
        local_groups = _updates(ddp, optimizer, scheduler, loader, 2,
                                limit=None if strict else 1)
        # Both ranks' local slices together must recover each full global update.
        gathered: list[list[list[int]] | None] = [None, None]
        dist.all_gather_object(gathered, local_groups)
        all_draws: list[tuple[float, float, float] | None] = [None, None]
        dist.all_gather_object(all_draws, draws)
        if not strict:
            next_batch = position["next_batch"] + 2
            next_step = payload["step"] + 1
            expanded_dir = directory / "expanded"
            if rank == 0:
                expanded_dir.mkdir()
            dist.barrier()
            # Distinct saved streams detect accidental rank-file reuse on strict resume.
            for _ in range(rank):
                _rng_draws()
            _save_rng_state(expanded_dir, rank)
            expected_next_draws = _rng_draws()
            dist.barrier()
            if rank == 0:
                torch.save({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": next_step,
                    "data_position": {"epoch": EPOCH, "next_batch": next_batch},
                    "settings": EXPANDED_SETTINGS,
                }, expanded_dir / "state.pt")
            torch.save(expected_next_draws, directory / f"expected_rank_{rank}.pt")
        else:
            expected = torch.load(directory / f"expected_rank_{rank}.pt",
                                  weights_only=False)
            assert draws == expected, (rank, draws, expected)
        if rank == 0:
            torch.save({
                "groups": gathered, "draws": all_draws,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, directory / ("strict_result.pt" if strict else "expanded_result.pt"))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_ranks(root: Path, strict: bool) -> None:
    context = mp.get_context("spawn")
    init_file = root / ("strict_init" if strict else "expanded_init")
    workers = [context.Process(target=_distributed_worker,
                               args=(rank, str(init_file), str(root), strict))
               for rank in range(2)]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join(timeout=45)
        alive = [worker for worker in workers if worker.is_alive()]
        if alive:
            raise AssertionError(f"Gloo workers timed out: {[worker.pid for worker in alive]}")
        assert all(worker.exitcode == 0 for worker in workers), [
            worker.exitcode for worker in workers
        ]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def _global_groups(gathered: list[list[list[int]]]) -> list[list[int]]:
    assert len(gathered[0]) == len(gathered[1])
    return [gathered[0][i] + gathered[1][i] for i in range(len(gathered[0]))]


class _TinyNativeVLA(nn.Module):
    """Expose an unused trainable lm_head through the native ModelOutput type."""

    def __init__(self) -> None:
        super().__init__()
        self.llm_dim = 4
        self.embed = nn.Embedding(16, 4)
        self.encoder = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 16)
        self.vision_backbone = _TinyVisionInfo()

    def forward(self, *, input_ids: torch.Tensor, **kwargs) -> PrismaticCausalLMOutputWithPast:
        hidden = self.encoder(self.embed(input_ids))
        logits = self.lm_head(hidden)
        return PrismaticCausalLMOutputWithPast(
            logits=logits, loss=logits.square().mean(), hidden_states=(hidden,),
        )


class _TinyVisionInfo:
    def get_num_patches(self) -> int:
        return 0

    def get_num_images_in_input(self) -> int:
        return 1


def _native_worker(rank: int, init_file: str) -> None:
    from csgo_seen10.model import (DUMMY_ACTION_TOKEN_ID, STOP_INDEX, ModelBundle,
                                   native_run_forward_pass)
    from csgo_seen10.runner import DistributedContext, _wrap_ddp

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=30))
    try:
        torch.manual_seed(101)
        bundle = ModelBundle(
            vla=_TinyNativeVLA(),
            action_head=nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(20, 5)),
            processor=None, model_path="tiny-native-test",
        )
        ctx = DistributedContext(rank=rank, world_size=2, local_rank=rank,
                                 device=torch.device("cpu"))
        bundle = _wrap_ddp(bundle, ctx)
        optimizer = torch.optim.SGD(
            list(bundle.vla.parameters()) + list(bundle.action_head.parameters()), lr=0.01,
        )
        labels = torch.tensor([[-100] * 3 + [DUMMY_ACTION_TOKEN_ID] * 5
                               + [STOP_INDEX]] * 2)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 1, 1, 1, 1, 1, 1]] * 2),
            "labels": labels,
            "attention_mask": torch.ones((2, 9), dtype=torch.bool),
            "pixel_values": torch.zeros((2, 12, 2, 2)),
            "actions": torch.full((2, 1, 5), 0.2 + rank / 10),
        }
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = native_run_forward_pass(bundle, batch, device="cpu")
            loss.backward()
            optimizer.step()
        assert bundle.vla.module.lm_head.weight.grad is None
        parameters = torch.cat([
            parameter.detach().flatten() for parameter in bundle.action_head.parameters()
        ])
        peers = [torch.empty_like(parameters) for _ in range(2)]
        dist.all_gather(peers, parameters)
        torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_native_two_ranks(root: Path) -> None:
    context = mp.get_context("spawn")
    init_file = root / "native_init"
    workers = [context.Process(target=_native_worker, args=(rank, str(init_file)))
               for rank in range(2)]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join(timeout=45)
        alive = [worker for worker in workers if worker.is_alive()]
        if alive:
            raise AssertionError(f"Native Gloo workers timed out: {[w.pid for w in alive]}")
        assert all(worker.exitcode == 0 for worker in workers), [
            worker.exitcode for worker in workers
        ]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


class TopologyResumeGlooTests(unittest.TestCase):
    def test_native_model_output_unused_lm_head_survives_two_ddp_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _run_native_two_ranks(Path(temporary))

    def test_one_to_two_then_strict_two_to_two_matches_single_rank(self) -> None:
        from csgo_seen10.runner import _save_rng_state

        torch.set_num_threads(1)
        torch.manual_seed(29)
        np.random.seed(29)
        random.seed(29)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            single = root / "single"
            single.mkdir()
            model, optimizer, scheduler = _components()
            first = _updates(model, optimizer, scheduler, _loader(0, 1, 8, 1, 0), 1,
                             limit=1)
            self.assertEqual(len(first), 1)
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": 1,
                "data_position": {"epoch": EPOCH, "next_batch": 1},
                "settings": SAVED_SETTINGS,
            }, single / "state.pt")
            _save_rng_state(single, 0)
            expected_draws = _rng_draws()
            baseline_groups = _updates(
                model, optimizer, scheduler, _loader(0, 1, 8, 1, 1), 1,
            )
            self.assertEqual(len(baseline_groups), 3)
            baseline_model = {key: value.detach().clone()
                              for key, value in model.state_dict().items()}
            baseline_optimizer = optimizer.state_dict()
            baseline_scheduler = scheduler.state_dict()

            _run_two_ranks(root, strict=False)
            expanded = torch.load(root / "expanded_result.pt", weights_only=False)
            self.assertEqual(expanded["draws"], [expected_draws, expected_draws])
            self.assertEqual(_global_groups(expanded["groups"]), baseline_groups[:1])

            _run_two_ranks(root, strict=True)
            strict = torch.load(root / "strict_result.pt", weights_only=False)
            resumed_groups = (_global_groups(expanded["groups"])
                              + _global_groups(strict["groups"]))
            self.assertEqual(resumed_groups, baseline_groups)
            all_groups = first + resumed_groups
            self.assertEqual(len(all_groups), DATASET_SIZE // EFFECTIVE_BATCH)
            self.assertEqual(len(set(sum(all_groups, []))), 32)
            self.assertEqual(strict["scheduler"], baseline_scheduler)
            self.assertEqual(strict["optimizer"]["param_groups"],
                             baseline_optimizer["param_groups"])
            for key, expected in baseline_model.items():
                torch.testing.assert_close(strict["model"][key], expected,
                                           rtol=1e-11, atol=1e-12)
            for parameter, values in baseline_optimizer["state"].items():
                for key, expected in values.items():
                    actual = strict["optimizer"]["state"][parameter][key]
                    if isinstance(expected, torch.Tensor):
                        torch.testing.assert_close(actual, expected,
                                                   rtol=1e-11, atol=1e-12)
                    else:
                        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
