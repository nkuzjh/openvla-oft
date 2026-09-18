"""Training, inference and checkpoint orchestration for Seen-10 localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# Constants are selected when the native Prismatic modules are imported.
os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "CSGO")

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from .data import SEEN10_MAPS, CSGOSeen10Dataset
from .model import (
    ModelBundle,
    collate_samples,
    create_model,
    model_provenance,
    native_run_forward_pass,
    predict_normalized,
    save_component_checkpoint,
    sha256_file,
)

DEFAULT_DATA_ROOT = "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"
DEFAULT_SHARED_EVAL_DIR = "/home/jiahao/task/csgo_benchmark_v2_eval_general"
DEFAULT_MODEL_PATH = "checkpoints/openvla-7b"
DEFAULT_OUTPUT_ROOT = "outputs/csgo_benchmark_v2_seen10"
MODEL_NAME = "OpenVLA-OFT"


def _sha256_optional(path: str | os.PathLike[str]) -> str | None:
    candidate = Path(path)
    return sha256_file(candidate) if candidate.is_file() else None


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _jsonl_append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _set_seed(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    owns_process_group: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()


def init_distributed() -> DistributedContext:
    """Initialize torchrun DDP when launched under torchrun; support plain Python too."""

    requested_world = int(os.environ.get("WORLD_SIZE", "1"))
    requested_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    owns_group = False
    if requested_world > 1 or "RANK" in os.environ:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
            owns_group = True
        requested_world = dist.get_world_size()
        requested_rank = dist.get_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return DistributedContext(requested_rank, requested_world, local_rank, device, owns_group)


def finish_distributed(ctx: DistributedContext) -> None:
    if ctx.owns_process_group and dist.is_initialized():
        dist.destroy_process_group()


def _nested_get(config: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def _config_value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in config:
        return config[key]
    return _nested_get(config, ("train", key), _nested_get(config, ("model", key), default))


def _selected_maps(config: Mapping[str, Any]) -> tuple[str, ...]:
    maps = config.get("maps", config.get("data", {}).get("maps", SEEN10_MAPS))
    if maps is None:
        return SEEN10_MAPS
    selected = tuple(str(value) for value in maps)
    if not selected or any(name not in SEEN10_MAPS for name in selected) or len(set(selected)) != len(selected):
        raise ValueError(f"maps must be a non-empty subset of fixed Seen-10 order, got {selected!r}")
    # Keep the published map order even when a subset is requested.
    return tuple(name for name in SEEN10_MAPS if name in selected)


def _data_root(config: Mapping[str, Any]) -> str:
    return str(
        os.environ.get(
            "CSGO_DATA_ROOT",
            os.environ.get(
                "DATA_ROOT",
                config.get("data_root", config.get("data", {}).get("root", DEFAULT_DATA_ROOT)),
            ),
        )
    )


def _model_path(config: Mapping[str, Any]) -> str:
    return str(config.get("model_path", config.get("model", {}).get("path", DEFAULT_MODEL_PATH)))


def _configure_data_environment(config: Mapping[str, Any]) -> None:
    """Pass config-based protocol location to the lazy data adapter."""

    os.environ.setdefault(
        "SHARED_EVAL_DIR",
        str(config.get("shared_eval_dir", os.environ.get("SHARED_EVAL_DIR", DEFAULT_SHARED_EVAL_DIR))),
    )


def _output_root(config: Mapping[str, Any]) -> Path:
    return Path(config.get("output_root", DEFAULT_OUTPUT_ROOT)).expanduser().resolve()


def _seed_output_dir(config: Mapping[str, Any], seed: int) -> Path:
    return _output_root(config) / MODEL_NAME / f"seed_{int(seed)}"


def _smoke_output_dir(config: Mapping[str, Any], seed: int) -> Path:
    root = Path(config.get("smoke_output_root", "outputs/csgo_benchmark_v2_seen10_smoke")).expanduser().resolve()
    return root / MODEL_NAME / f"seed_{int(seed)}"


class RecordDataset(Dataset):
    """Map-style dataset over model-facing metadata records."""

    def __init__(self, records: Sequence[Mapping[str, Any]]):
        self.records = [dict(record) for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class ExactRankSampler(Sampler[int]):
    """Shard rows by rank without padding or duplicating evaluation samples."""

    def __init__(self, dataset: Dataset, *, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"Invalid rank/world_size: {self.rank}/{self.world_size}")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        # Kept for the same call site as DistributedSampler; evaluation order
        # is intentionally fixed across epochs and ranks.
        del epoch


def _loader_generator(seed: int, stream: int) -> torch.Generator:
    """Create a DataLoader-only RNG stream independent of model randomness."""

    generator = torch.Generator()
    generator.manual_seed((int(seed) * 1_000_003 + int(stream)) % (2**63 - 1))
    return generator


def _records_for_split(
    data_root: str,
    split: str,
    maps: Sequence[str],
    *,
    include_targets: bool,
    smoke: bool = False,
    samples_per_map: int = 10,
) -> RecordDataset:
    records: list[dict[str, Any]] = []
    for map_name in maps:
        dataset = CSGOSeen10Dataset(
            data_root,
            split=split,
            maps=[map_name],
            include_targets=include_targets,
            max_samples=samples_per_map if smoke else None,
        )
        records.extend(dataset[index] for index in range(len(dataset)))
    if not records:
        raise ValueError(f"No records selected for {split} maps={maps}")
    return RecordDataset(records)


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    sampler: Sampler[int],
    processor: Any,
    include_targets: bool,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        collate_fn=lambda instances: collate_samples(instances, processor, include_targets=include_targets),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
        # DataLoader iterator construction draws worker/base seeds from this
        # generator.  Keep that bookkeeping off the model RNG restored from a
        # native checkpoint so resume keeps the dropout stream intact.
        generator=generator,
    )


def _wrap_ddp(bundle: ModelBundle, ctx: DistributedContext) -> ModelBundle:
    if ctx.world_size <= 1:
        return bundle
    bundle.vla = DDP(
        bundle.vla,
        device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )
    bundle.action_head = DDP(
        bundle.action_head,
        device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    return bundle


def _trainable_parameters(bundle: ModelBundle) -> list[torch.nn.Parameter]:
    params = [param for param in bundle.vla.parameters() if param.requires_grad]
    params.extend(param for param in bundle.action_head.parameters() if param.requires_grad)
    if not params:
        raise RuntimeError("No trainable parameters found; LoRA/action head setup failed")
    return params


def _checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{int(step):08d}"


def _safe_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            raise FileExistsError(f"Refusing to replace non-symlink directory {link}")
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)


def _checkpoint_fingerprint(checkpoint_dir: Path) -> dict[str, Any]:
    files = []
    for candidate in sorted(checkpoint_dir.rglob("*")):
        if candidate.is_file() and candidate.name not in {"optimizer.pt"}:
            files.append({"path": str(candidate.relative_to(checkpoint_dir)), "sha256": sha256_file(candidate)})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {"sha256": digest, "files": files}


def _save_rng_state(checkpoint_dir: Path, rank: int) -> None:
    payload: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, checkpoint_dir / f"rng_state_rank_{rank}.pt")


def _restore_rng_state(checkpoint_dir: Path, rank: int, device: torch.device) -> bool:
    path = checkpoint_dir / f"rng_state_rank_{rank}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint is missing the exact rank RNG state for rank={rank}: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(payload["torch"])
    np.random.set_state(payload["numpy"])
    random.setstate(payload["python"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all(payload["cuda"])
    return True


def save_checkpoint(
    *,
    bundle: ModelBundle,
    checkpoint_dir: Path,
    step: int,
    optimizer: AdamW,
    scheduler: MultiStepLR,
    data_state: Mapping[str, Any],
    best_val_loss: float,
    ctx: DistributedContext,
    provenance: Mapping[str, Any],
) -> Path:
    """Save native adapter/head state plus optimizer, scheduler, RNG and data position."""

    if ctx.is_main:
        if checkpoint_dir.exists():
            raise FileExistsError(f"Checkpoint already exists; refusing to overwrite: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        save_component_checkpoint(bundle, checkpoint_dir, step)
    ctx.barrier()
    # Every rank has a distinct RNG stream; saving it is required for an exact
    # torchrun resume rather than only a model-weight resume.
    _save_rng_state(checkpoint_dir, ctx.rank)
    ctx.barrier()
    if ctx.is_main:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        state = {
            "step": int(step),
            "best_val_loss": float(best_val_loss),
            "data_position": dict(data_state),
            "optimizer": "optimizer.pt",
            "scheduler": "scheduler.pt",
            "rng_pattern": "rng_state_rank_<rank>.pt",
            "training_settings": dict(provenance.get("training_settings", {})),
            "provenance": dict(provenance),
        }
        _json_dump(checkpoint_dir / "training_state.json", state)
        _json_dump(checkpoint_dir / "provenance.json", dict(provenance))
    ctx.barrier()
    return checkpoint_dir


def _find_resume_checkpoint(path: str | os.PathLike[str], *, prefer: str = "late") -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir() and (candidate / "adapter_config.json").is_file():
        return candidate
    if candidate.is_dir() and (candidate / "lora_adapter" / "adapter_config.json").is_file():
        return candidate
    if candidate.is_dir() and (candidate / "checkpoints" / prefer).exists():
        return (candidate / "checkpoints" / prefer).resolve()
    if candidate.is_dir() and (candidate / prefer).exists():
        return (candidate / prefer).resolve()
    raise FileNotFoundError(f"Cannot resolve checkpoint from {path}")


def _load_training_state(checkpoint_dir: Path, *, device: torch.device) -> dict[str, Any]:
    path = checkpoint_dir / "training_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"Native resume checkpoint has no training_state.json: {checkpoint_dir}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"Invalid training_state.json: {path}")
    return state


def _restore_optimizer_scheduler(
    checkpoint_dir: Path,
    optimizer: AdamW,
    scheduler: MultiStepLR,
    *,
    device: torch.device,
) -> dict[str, Any]:
    state = _load_training_state(checkpoint_dir, device=device)
    optimizer_path = checkpoint_dir / str(state.get("optimizer", "optimizer.pt"))
    scheduler_path = checkpoint_dir / str(state.get("scheduler", "scheduler.pt"))
    if not optimizer_path.is_file() or not scheduler_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint missing optimizer/scheduler state: {checkpoint_dir}")
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))
    scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=False))
    return state


def _provenance_for_run(
    *,
    bundle: ModelBundle,
    data_root: str,
    config: Mapping[str, Any],
    seed: int,
    split: str,
    maps: Sequence[str],
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = Path(data_root) / "benchmark_manifest.json"
    report = Path(data_root) / "minimal_dataset_report.json"
    result = {
        "model": model_provenance(bundle.model_path),
        "model_name": MODEL_NAME,
        "repository": "openvla-oft",
        "repository_commit": _git_commit(),
        "adapter": "peft all-linear LoRA",
        "action_head": "native L1RegressionActionHead",
        "action_horizon": 1,
        "action_dim": 5,
        "image_channels": 12,
        "proprio": False,
        "data_root": str(Path(data_root).resolve()),
        "benchmark_manifest": str(manifest.resolve()),
        "benchmark_manifest_sha256": _sha256_optional(manifest),
        "minimal_dataset_report": str(report.resolve()),
        "minimal_dataset_report_sha256": _sha256_optional(report),
        "split": split,
        "maps": list(maps),
        "seed": int(seed),
        "config": str(config.get("_config_path", "")),
        "constants_selector": "CSGO",
    }
    if checkpoint_dir is not None:
        result["checkpoint_dir"] = str(checkpoint_dir.resolve())
        result["checkpoint_fingerprint"] = _checkpoint_fingerprint(checkpoint_dir)
    return result


def _resume_identity(
    *,
    config: Mapping[str, Any],
    data_root: str,
    seed: int,
    maps: Sequence[str],
    smoke: bool,
) -> dict[str, Any]:
    """Return the dataset/model recipe fields that must match on resume."""

    requested_model = _model_path(config)
    requested_path = Path(requested_model).expanduser()
    model_identity = str(requested_path.resolve()) if requested_path.is_dir() else requested_model
    manifest = Path(data_root) / "benchmark_manifest.json"
    report = Path(data_root) / "minimal_dataset_report.json"
    return {
        "model_request": model_identity,
        "data_root": str(Path(data_root).resolve()),
        "benchmark_manifest_sha256": _sha256_optional(manifest),
        "minimal_dataset_report_sha256": _sha256_optional(report),
        "split": "seen_train",
        "maps": list(maps),
        "seed": int(seed),
        "smoke": bool(smoke),
        "use_lora": bool(_config_value(config, "use_lora", True)),
        "lora_rank": int(_config_value(config, "lora_rank", 32)),
        "lora_dropout": float(_config_value(config, "lora_dropout", 0.0)),
        "gradient_checkpointing": bool(_config_value(config, "gradient_checkpointing", True)),
    }


def _git_commit() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _ensure_new_run_dir(run_dir: Path, *, resume: bool) -> None:
    if run_dir.exists() and not resume:
        if any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory already contains output; use --resume with an explicit checkpoint: {run_dir}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)


def _ensure_resume_target_is_safe(
    run_dir: Path,
    resume_step: int,
) -> None:
    """Refuse to append a resume run ahead of an already-saved checkpoint."""

    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return
    later_steps = []
    for candidate in checkpoint_root.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith("step_"):
            continue
        step_text = candidate.name[len("step_") :]
        if step_text.isdigit() and int(step_text) > int(resume_step):
            later_steps.append(candidate.name)
    if later_steps:
        later_steps.sort()
        raise FileExistsError(
            f"Cannot resume from step {int(resume_step)} into run directory {run_dir}: "
            f"later checkpoint directories already exist ({', '.join(later_steps)}). "
            "Resume from the run's `late` checkpoint or choose a new `output_root`."
        )


def _reduce_scalar(value: float, count: int, ctx: DistributedContext) -> tuple[float, int]:
    values = torch.tensor([float(value), float(count)], device=ctx.device, dtype=torch.float64)
    if ctx.world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), int(values[1].item())


def evaluate(
    bundle: ModelBundle,
    dataloader: DataLoader,
    *,
    ctx: DistributedContext,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate every selected validation row; no time limit or test data is used."""

    bundle.vla.eval()
    bundle.action_head.eval()
    total_loss = 0.0
    total_count = 0
    local_predictions: list[dict[str, Any]] = []
    evaluation_start = time.time()
    expected_count = len(dataloader.dataset)
    # ExactRankSampler shards validation rows without padding.  In a
    # multi-process run rank zero cannot report the global count until the
    # final reduction, so progress logs use its local sampler size explicitly.
    progress_count = expected_count if ctx.world_size <= 1 else len(dataloader.sampler)
    report_targets = {
        target
        for target in (
            1,
            2,
            math.ceil(progress_count * 0.2),
            math.ceil(progress_count * 0.4),
            math.ceil(progress_count * 0.6),
            math.ceil(progress_count * 0.8),
            progress_count,
        )
        if target > 0
    }
    reported_targets: set[int] = set()
    with torch.no_grad():
        for batch in dataloader:
            _, metrics, predictions = native_run_forward_pass(bundle, batch, device=ctx.device)
            batch_count = int(batch["actions"].shape[0])
            total_loss += float(metrics["loss_value"]) * batch_count
            total_count += batch_count
            if ctx.is_main:
                processed = total_count
                reached = sorted(
                    target for target in report_targets if target <= processed and target not in reported_targets
                )
                for target in reached:
                    scope = "" if ctx.world_size <= 1 else "rank0 "
                    print(
                        f"[val] {scope}processed={processed}/{progress_count} "
                        f"elapsed={time.time() - evaluation_start:.1f}s",
                        flush=True,
                    )
                    reported_targets.add(target)
            if collect_predictions:
                for sample_id, map_name, values in zip(
                    batch["sample_ids"], batch["map_names"], predictions[:, 0].float().cpu().tolist()
                ):
                    local_predictions.append(
                        {
                            "sample_id": str(sample_id),
                            "map_name": str(map_name),
                            "pred_x": float(values[0]),
                            "pred_y": float(values[1]),
                            "pred_z": float(values[2]),
                            "pred_pitch": float(values[3]),
                            "pred_yaw": float(values[4]),
                        }
                    )
    total_loss, total_count = _reduce_scalar(total_loss, total_count, ctx)
    bundle.vla.train()
    bundle.action_head.train()
    if total_count <= 0:
        raise ValueError("Validation selected no samples")
    result: dict[str, Any] = {"loss": total_loss / total_count, "count": float(total_count)}
    if collect_predictions:
        if ctx.world_size > 1:
            gathered: list[list[dict[str, Any]] | None] = [None for _ in range(ctx.world_size)]
            dist.all_gather_object(gathered, local_predictions)
            result["predictions"] = [row for shard in gathered if shard is not None for row in shard]
        else:
            result["predictions"] = local_predictions
    return result


def _render_localization_visuals(
    *,
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    output_dir: Path,
    maps: Sequence[str],
    seed: int,
    samples_per_map: int,
) -> dict[str, str]:
    """Render the companion visualizer's fixed ten samples per map."""

    target_dir = Path(output_dir).expanduser().resolve()
    missing_maps = tuple(
        map_name
        for map_name in maps
        if not (target_dir / f"vis_map_{map_name}.png").is_file()
    )
    if not missing_maps:
        return {}

    # This is a required smoke artifact, so import/call failures must surface
    # instead of being silently downgraded to a successful run.
    from .visualize import visualize_localization_predictions

    return visualize_localization_predictions(
        records,
        list(predictions),
        target_dir,
        seed=int(seed),
        samples_per_map=int(samples_per_map),
        maps=missing_maps,
        prediction_space="normalized",
        overwrite=False,
    )


def train(config: Mapping[str, Any], *, seed: int, resume_checkpoint: str | None = None, smoke: bool = False) -> Path:
    """Run native AdamW/MultiStepLR training with five validation/save events."""

    ctx = init_distributed()
    _set_seed(seed, ctx.rank)
    maps = _selected_maps(config)
    if not smoke and maps != SEEN10_MAPS:
        raise ValueError("Formal Seen-10 train requires all ten maps in the published order")
    data_root = _data_root(config)
    _configure_data_environment(config)
    run_dir = _smoke_output_dir(config, seed) if smoke else _seed_output_dir(config, seed)
    resume_dir = _find_resume_checkpoint(resume_checkpoint) if resume_checkpoint else None
    if resume_dir is not None:
        resume_state_hint = _load_training_state(resume_dir, device=torch.device("cpu"))
        _ensure_resume_target_is_safe(run_dir, int(resume_state_hint["step"]))
    _ensure_new_run_dir(run_dir, resume=resume_dir is not None)
    ctx.barrier()

    batch_size = int(_config_value(config, "batch_size", 1))
    grad_accum = int(_config_value(config, "grad_accumulation_steps", 32))
    learning_rate = float(_config_value(config, "learning_rate", 5e-4))
    max_steps = int(_config_value(config, "max_steps", 10_000))
    event_every = int(_config_value(config, "event_every", max_steps // 5))
    if max_steps <= 0 or max_steps % 5 != 0 or event_every != max_steps // 5 or event_every <= 0:
        raise ValueError("max_steps must be positive and divisible by 5; event_every must equal max_steps//5")
    if batch_size <= 0 or grad_accum <= 0:
        raise ValueError("batch_size and grad_accumulation_steps must be positive")
    if smoke:
        batch_size = 1
        grad_accum = 1
        max_steps = min(max_steps, int(_config_value(config, "smoke_max_steps", 5)))
        max_steps = max(5, max_steps)
        if max_steps % 5:
            max_steps += 5 - max_steps % 5
        event_every = max_steps // 5

    resume_identity = _resume_identity(
        config=config,
        data_root=data_root,
        seed=seed,
        maps=maps,
        smoke=smoke,
    )
    training_settings = {
        "batch_size": int(batch_size),
        "grad_accumulation_steps": int(grad_accum),
        "world_size": int(ctx.world_size),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "num_steps_before_decay": int(_config_value(config, "num_steps_before_decay", 100_000)),
        "event_every": int(event_every),
        "smoke": bool(smoke),
    }
    if resume_dir is not None:
        saved_settings = resume_state_hint.get("training_settings")
        saved_provenance = resume_state_hint.get("provenance")
        if not isinstance(saved_provenance, Mapping):
            raise ValueError("Resume checkpoint has no valid provenance record")
        if saved_settings is None:
            saved_settings = saved_provenance.get("training_settings")
        if not isinstance(saved_settings, Mapping):
            raise ValueError(
                "Resume checkpoint has no training_settings; refusing a potentially different sampler setup"
            )
        mismatches = {
            key: (saved_settings.get(key), value)
            for key, value in training_settings.items()
            if saved_settings.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Resume training settings differ from checkpoint: {mismatches}")
        saved_identity = saved_provenance.get("resume_identity")
        if not isinstance(saved_identity, Mapping):
            raise ValueError("Resume checkpoint has no resume_identity; refusing a different dataset/model recipe")
        identity_mismatches = {
            key: (saved_identity.get(key), value)
            for key, value in resume_identity.items()
            if saved_identity.get(key) != value
        }
        if identity_mismatches:
            raise ValueError(f"Resume dataset/model identity differs from checkpoint: {identity_mismatches}")

    model_path = _model_path(config)
    bundle = create_model(
        model_path,
        device=ctx.device,
        use_lora=bool(_config_value(config, "use_lora", True)),
        lora_rank=int(_config_value(config, "lora_rank", 32)),
        lora_dropout=float(_config_value(config, "lora_dropout", 0.0)),
        checkpoint_dir=resume_dir,
        gradient_checkpointing=bool(_config_value(config, "gradient_checkpointing", True)),
    )
    bundle = _wrap_ddp(bundle, ctx)

    train_dataset = _records_for_split(
        data_root,
        "seen_train",
        maps,
        include_targets=True,
        smoke=smoke,
        samples_per_map=int(_config_value(config, "smoke_samples_per_map", 10)),
    )
    val_dataset = _records_for_split(
        data_root,
        "seen_validation",
        maps,
        include_targets=True,
        smoke=smoke,
        samples_per_map=int(_config_value(config, "smoke_samples_per_map", 10)),
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
        seed=int(seed),
        drop_last=False,
    )
    # Validation is a metric over the exact selected rows.  DistributedSampler
    # pads an uneven dataset and would count duplicated rows, so use a strict
    # rank slice here.
    val_sampler = ExactRankSampler(val_dataset, rank=ctx.rank, world_size=ctx.world_size)
    num_workers = int(_config_value(config, "num_workers", 0))
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        processor=bundle.processor,
        include_targets=True,
        num_workers=num_workers,
        generator=_loader_generator(seed, 2 * ctx.rank),
    )
    val_loader = _loader(
        val_dataset,
        batch_size=int(_config_value(config, "val_batch_size", batch_size)),
        sampler=val_sampler,
        processor=bundle.processor,
        include_targets=True,
        num_workers=num_workers,
        generator=_loader_generator(seed, 1 + 2 * ctx.rank),
    )
    if len(train_loader) < grad_accum:
        raise ValueError(
            f"train loader has {len(train_loader)} batches but grad_accumulation_steps={grad_accum}; "
            "increase the selected dataset or lower accumulation"
        )

    trainable_parameters = _trainable_parameters(bundle)
    optimizer = AdamW(trainable_parameters, lr=learning_rate)
    decay_step = int(_config_value(config, "num_steps_before_decay", 100_000))
    scheduler = MultiStepLR(optimizer, milestones=[decay_step], gamma=0.1)
    start_step = 0
    epoch = 0
    next_batch = 0
    best_val_loss = math.inf
    if resume_dir is not None:
        state = _restore_optimizer_scheduler(resume_dir, optimizer, scheduler, device=ctx.device)
        start_step = int(state["step"])
        position = state.get("data_position", {})
        epoch = int(position.get("epoch", 0))
        next_batch = int(position.get("next_batch", 0))
        best_val_loss = float(state.get("best_val_loss", math.inf))
        _restore_rng_state(resume_dir, ctx.rank, ctx.device)

    provenance = _provenance_for_run(
        bundle=bundle,
        data_root=data_root,
        config=config,
        seed=seed,
        split="seen_train",
        maps=maps,
        checkpoint_dir=resume_dir,
    )
    provenance["resume_identity"] = resume_identity
    provenance["training_settings"] = training_settings
    if ctx.is_main:
        _json_dump(run_dir / "run_provenance.json", provenance)
        _json_dump(
            run_dir / "training_config.json",
            {key: value for key, value in config.items() if not key.startswith("_")}
            | {"seed": int(seed), "smoke_only": smoke},
        )
    ctx.barrier()

    loss_log = run_dir / "logs" / "main_loss.jsonl"
    validation_log = run_dir / "logs" / "validation.jsonl"
    if ctx.is_main and not loss_log.exists():
        loss_log.parent.mkdir(parents=True, exist_ok=True)
    ctx.barrier()
    global_step = start_step
    if global_step >= max_steps:
        if ctx.is_main:
            _plot_loss(loss_log, run_dir / "logs" / "main_loss.png")
        finish_distributed(ctx)
        return run_dir

    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    probe_parameter = trainable_parameters[0]
    while global_step < max_steps:
        train_sampler.set_epoch(epoch)
        progressed_any = False
        epoch_start_batch = next_batch
        if epoch_start_batch >= len(train_loader):
            epoch += 1
            next_batch = 0
            continue
        for batch_index, batch in enumerate(train_loader):
            if batch_index < epoch_start_batch:
                continue
            progressed_any = True
            bundle.vla.train()
            bundle.action_head.train()
            loss, metrics, _ = native_run_forward_pass(bundle, batch, device=ctx.device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step={global_step}: {loss.item()}")
            (loss / grad_accum).backward()
            local_micro = batch_index - epoch_start_batch
            should_step = (local_micro + 1) % grad_accum == 0
            if not should_step:
                continue
            probe_before = float(probe_parameter.detach().flatten()[0].float().item())
            probe_grad_norm = (
                float(probe_parameter.grad.detach().float().norm().item())
                if probe_parameter.grad is not None
                else 0.0
            )
            optimizer.step()
            probe_delta = abs(float(probe_parameter.detach().flatten()[0].float().item()) - probe_before)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            next_batch = batch_index + 1
            next_epoch = epoch
            if next_batch >= len(train_loader):
                next_epoch, next_batch = epoch + 1, 0
            if ctx.is_main:
                _jsonl_append(
                    loss_log,
                    {
                        "step": int(global_step),
                        "loss": float(metrics["loss_value"]),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "probe_grad_norm": probe_grad_norm,
                        "probe_parameter_delta": probe_delta,
                        "elapsed_seconds": float(time.time() - start_time),
                    },
                )
                if global_step <= 2 or global_step % 50 == 0:
                    print(
                        f"[train] optimizer_step={global_step}/{max_steps} "
                        f"loss={metrics['loss_value']:.6f} grad_norm={probe_grad_norm:.6g} "
                        f"parameter_delta={probe_delta:.6g} elapsed={time.time() - start_time:.1f}s",
                        flush=True,
                    )

            if global_step % event_every == 0:
                validation = evaluate(bundle, val_loader, ctx=ctx, collect_predictions=True)
                if ctx.is_main:
                    print(
                        f"[train] validation step={global_step}/{max_steps} "
                        f"processed={int(validation['count'])}/{len(val_dataset)} "
                        f"val_loss={validation['loss']:.6f}",
                        flush=True,
                    )
                    _jsonl_append(
                        validation_log,
                        {
                            "step": int(global_step),
                            "val_loss": float(validation["loss"]),
                            "count": int(validation["count"]),
                        },
                    )
                    validation_predictions = list(validation.get("predictions", []))
                    validation_prediction_path = run_dir / "logs" / "validation_predictions.jsonl"
                    validation_prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    with validation_prediction_path.open("a", encoding="utf-8") as stream:
                        for row in validation_predictions:
                            stream.write(
                                json.dumps(
                                    {"step": int(global_step), **row},
                                    ensure_ascii=False,
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                    _render_localization_visuals(
                        records=val_dataset.records,
                        predictions=validation_predictions,
                        output_dir=run_dir / "validation_visualizations" / f"step_{global_step:08d}",
                        maps=maps,
                        seed=seed,
                        samples_per_map=int(_config_value(config, "visual_samples_per_map", 10)),
                    )
                if validation["loss"] < best_val_loss:
                    best_val_loss = validation["loss"]
                    is_best = True
                else:
                    is_best = False
                checkpoint_dir = _checkpoint_path(run_dir, global_step)
                save_checkpoint(
                    bundle=bundle,
                    checkpoint_dir=checkpoint_dir,
                    step=global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    data_state={"epoch": int(next_epoch), "next_batch": int(next_batch)},
                    best_val_loss=best_val_loss,
                    ctx=ctx,
                    provenance=provenance,
                )
                if ctx.is_main:
                    _safe_link(run_dir / "checkpoints" / "late", checkpoint_dir)
                    _safe_link(run_dir / "late", checkpoint_dir)
                    if is_best or not (run_dir / "checkpoints" / "best").exists():
                        _safe_link(run_dir / "checkpoints" / "best", checkpoint_dir)
                        _safe_link(run_dir / "best", checkpoint_dir)
                ctx.barrier()
            if global_step >= max_steps:
                epoch, next_batch = next_epoch, next_batch
                break
            # Continue with the current epoch after a step.
        # Do not carry an incomplete accumulation tail into the next epoch.
        # Checkpoints are written immediately after full optimizer updates, so
        # clearing here preserves exact optimizer state on resume.
        optimizer.zero_grad(set_to_none=True)
        if global_step >= max_steps:
            break
        if not progressed_any:
            # The saved position can point beyond this loader only when a
            # malformed external state is supplied.
            raise RuntimeError(f"Training made no progress at epoch={epoch}, next_batch={next_batch}")
        epoch += 1
        next_batch = 0

    if ctx.is_main:
        _plot_loss(loss_log, run_dir / "logs" / "main_loss.png")
    ctx.barrier()
    finish_distributed(ctx)
    return run_dir


def _plot_loss(log_path: Path, output_path: Path) -> None:
    if not log_path.is_file():
        return
    rows = []
    with log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["step"] for row in rows], [row["loss"] for row in rows], linewidth=1.2)
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("main L1 loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _read_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return existing
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not isinstance(payload.get("sample_id"), str):
                raise ValueError(f"Invalid prediction row at {path}:{line_no}")
            sample_id = payload["sample_id"]
            if sample_id in existing:
                raise ValueError(f"Duplicate existing prediction sample_id={sample_id}")
            for field in ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw"):
                value = float(payload[field])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite existing prediction {sample_id}/{field}")
            existing[sample_id] = payload
    return existing


def _write_predictions_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Commit a prediction prefix without exposing a half-written JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp_path, path)


def _validate_existing_provenance(manifest_path: Path, provenance: Mapping[str, Any]) -> None:
    if not manifest_path.is_file():
        return
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(existing, Mapping):
        raise ValueError(f"Invalid inference manifest: {manifest_path}")
    for key in (
        "checkpoint_dir",
        "checkpoint_fingerprint",
        "benchmark_manifest_sha256",
        "split",
        "seed",
        "maps",
        "expected_sample_ids_sha256",
    ):
        if key in existing and existing.get(key) != provenance.get(key):
            raise ValueError(
                f"Inference resume provenance mismatch for {key}: "
                f"existing={existing.get(key)!r}, current={provenance.get(key)!r}"
            )


def inference(
    config: Mapping[str, Any],
    *,
    seed: int,
    checkpoint: str | None = None,
    smoke: bool = False,
    resume: bool = False,
) -> Path:
    """Generate normalized localization JSONL, resuming only missing sample IDs."""

    ctx = init_distributed()
    _set_seed(seed, ctx.rank)
    maps = _selected_maps(config)
    if not smoke and maps != SEEN10_MAPS:
        raise ValueError("Formal Seen-10 inference requires all ten maps in the published order")
    data_root = _data_root(config)
    _configure_data_environment(config)
    run_dir = _smoke_output_dir(config, seed) if smoke else _seed_output_dir(config, seed)
    output_dir = run_dir / "localization"
    prediction_path = output_dir / "predictions.jsonl"
    manifest_path = output_dir / "inference_manifest.json"
    checkpoint_dir = _find_resume_checkpoint(checkpoint or str(run_dir), prefer="best")
    bundle = create_model(
        _model_path(config),
        device=ctx.device,
        use_lora=bool(_config_value(config, "use_lora", True)),
        lora_rank=int(_config_value(config, "lora_rank", 32)),
        lora_dropout=float(_config_value(config, "lora_dropout", 0.0)),
        checkpoint_dir=checkpoint_dir,
        gradient_checkpointing=bool(_config_value(config, "gradient_checkpointing", True)),
    )
    bundle = _wrap_ddp(bundle, ctx)
    dataset = _records_for_split(
        data_root,
        "seen_discrete_test",
        maps,
        include_targets=False,
        smoke=smoke,
        samples_per_map=int(_config_value(config, "smoke_samples_per_map", 10)),
    )
    provenance = _provenance_for_run(
        bundle=bundle,
        data_root=data_root,
        config=config,
        seed=seed,
        split="seen_discrete_test",
        maps=maps,
        checkpoint_dir=checkpoint_dir,
    )
    provenance.update(
        {
            "prediction_pose_space": "normalized",
            "sample_count": len(dataset),
            "smoke_only": bool(smoke),
            "official_output_written": not smoke,
        }
    )
    expected_ids = {str(record["sample_id"]) for record in dataset.records}
    expected_ids_digest = hashlib.sha256(
        json.dumps([str(record["sample_id"]) for record in dataset.records], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    provenance["expected_sample_ids_sha256"] = expected_ids_digest
    work_provenance_path = output_dir / "inference_provenance.json"
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        _validate_existing_provenance(manifest_path, provenance)
        if work_provenance_path.is_file():
            _validate_existing_provenance(work_provenance_path, provenance)
        existing = _read_existing_predictions(prediction_path)
        prediction_file_ids = set(existing)
        # A previous process can have completed rank shards before being
        # interrupted while merging.  Consume those rows before generating
        # anything new; deleting them before this read loses resumable work.
        shard_paths = sorted(output_dir.glob(".predictions.rank_*.jsonl"))
        for shard_path in shard_paths:
            for sample_id, row in _read_existing_predictions(shard_path).items():
                if sample_id in existing:
                    if existing[sample_id] != row:
                        raise ValueError(f"Conflicting existing/shard prediction sample_id={sample_id}")
                    # A crash after the atomic prefix commit but before shard
                    # cleanup leaves an identical row in both places.  It is
                    # safe to retain one copy and continue the resume.
                    continue
                existing[sample_id] = row
        unknown = set(existing) - expected_ids
        if unknown:
            raise ValueError(f"Existing predictions contain sample IDs outside selected split: {sorted(unknown)[:5]}")
        if (prediction_file_ids or shard_paths) and not resume and set(existing) != expected_ids:
            raise FileExistsError(
                f"Partial predictions already exist at {prediction_path}; pass --resume to fill only missing sample IDs"
            )
        if (
            (prediction_file_ids or shard_paths)
            and not work_provenance_path.is_file()
            and not manifest_path.is_file()
        ):
            raise ValueError(
                "Cannot reuse an unprovenanced prediction output; remove it or restore its checkpoint provenance"
            )
        if not work_provenance_path.is_file():
            _json_dump(
                work_provenance_path,
                dict(provenance)
                | {
                    "status": "in_progress",
                    "expected_sample_ids_sha256": expected_ids_digest,
                },
            )
    else:
        existing = {}
    ctx.barrier()
    if ctx.world_size > 1:
        object_list = [existing if ctx.is_main else None]
        dist.broadcast_object_list(object_list, src=0)
        existing = object_list[0] or {}
    unknown = set(existing) - expected_ids
    if unknown:
        raise ValueError(f"Existing predictions contain sample IDs outside selected split: {sorted(unknown)[:5]}")
    missing_records = [record for record in dataset.records if str(record["sample_id"]) not in existing]
    # The rank-zero process has incorporated old shards into ``existing``;
    # remove those intermediates only after every rank has received that map.
    if ctx.is_main:
        if shard_paths:
            # Make the merged prefix durable before deleting the only copies
            # of rows produced by an interrupted distributed inference.
            merged_prefix = [
                existing[str(record["sample_id"])]
                for record in dataset.records
                if str(record["sample_id"]) in existing
            ]
            _write_predictions_atomic(prediction_path, merged_prefix)
        for shard_path in output_dir.glob(".predictions.rank_*.jsonl"):
            shard_path.unlink(missing_ok=True)
    ctx.barrier()
    if not missing_records:
        if ctx.is_main:
            ordered = [existing[str(record["sample_id"])] for record in dataset.records]
            if not prediction_path.is_file():
                _write_predictions_atomic(prediction_path, ordered)
            visual_dataset = _records_for_split(
                data_root,
                "seen_discrete_test",
                maps,
                include_targets=True,
                smoke=smoke,
                samples_per_map=int(_config_value(config, "smoke_samples_per_map", 10)),
            )
            _render_localization_visuals(
                records=visual_dataset.records,
                predictions=ordered,
                output_dir=output_dir / "visualizations",
                maps=maps,
                seed=seed,
                samples_per_map=int(_config_value(config, "visual_samples_per_map", 10)),
            )
            provenance.update(
                {
                    "status": "complete",
                    "completed_count": len(ordered),
                    "resumed_missing_count": 0,
                    "expected_sample_ids_sha256": expected_ids_digest,
                }
            )
            _json_dump(manifest_path, provenance)
            _json_dump(work_provenance_path, provenance)
        ctx.barrier()
        finish_distributed(ctx)
        return prediction_path
    work_dataset = RecordDataset(missing_records)
    sampler = ExactRankSampler(work_dataset, rank=ctx.rank, world_size=ctx.world_size)
    loader = _loader(
        work_dataset,
        batch_size=int(_config_value(config, "inference_batch_size", 1)),
        sampler=sampler,
        processor=bundle.processor,
        include_targets=False,
        num_workers=int(_config_value(config, "num_workers", 0)),
        generator=_loader_generator(seed, 3 + 2 * ctx.rank),
    )
    shard_path = output_dir / f".predictions.rank_{ctx.rank}.jsonl"
    bundle.vla.eval()
    bundle.action_head.eval()
    inference_start = time.time()
    processed_local = 0
    report_batches = {
        value
        for value in (
            1,
            2,
            *(math.ceil(len(loader) * fraction) for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)),
        )
        if value <= len(loader)
    }
    with shard_path.open("w", encoding="utf-8") as stream:
        for batch_index, batch in enumerate(loader, 1):
            values = predict_normalized(bundle, batch, device=ctx.device)
            processed_local += len(batch["sample_ids"])
            for sample_id, map_name, row_value in zip(batch["sample_ids"], batch["map_names"], values.tolist()):
                row = {
                    "sample_id": str(sample_id),
                    "map_name": str(map_name),
                    "pred_x": float(row_value[0]),
                    "pred_y": float(row_value[1]),
                    "pred_z": float(row_value[2]),
                    "pred_pitch": float(row_value[3]),
                    "pred_yaw": float(row_value[4]),
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if ctx.is_main and batch_index in report_batches:
                print(
                    f"[infer] processed={processed_local}/{len(missing_records)} "
                    f"elapsed={time.time() - inference_start:.1f}s",
                    flush=True,
                )
    progress = torch.tensor([float(processed_local)], device=ctx.device, dtype=torch.float64)
    if ctx.world_size > 1:
        dist.all_reduce(progress, op=dist.ReduceOp.SUM)
    if ctx.is_main:
        print(
            f"[infer] processed={int(progress.item())}/{len(missing_records)} "
            f"elapsed={time.time() - inference_start:.1f}s",
            flush=True,
        )
    ctx.barrier()
    if ctx.is_main:
        merged = dict(existing)
        for rank in range(ctx.world_size):
            shard = output_dir / f".predictions.rank_{rank}.jsonl"
            if shard.is_file():
                with shard.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            row = json.loads(line)
                            sample_id = row["sample_id"]
                            if sample_id in merged:
                                raise ValueError(f"Duplicate generated prediction sample_id={sample_id}")
                            merged[sample_id] = row
        missing = expected_ids - set(merged)
        if missing:
            raise RuntimeError(f"Inference did not produce all selected sample IDs; missing={sorted(missing)[:10]}")
        # Keep manifest split order and commit one complete JSONL.  Replacing a
        # partial file is intentional only under --resume; a complete file is
        # left untouched when every sample was already present.
        ordered = [merged[str(record["sample_id"])] for record in dataset.records]
        _write_predictions_atomic(prediction_path, ordered)
        # Load target metadata only after model predictions have been produced.
        # The model-facing dataset above was constructed with include_targets=False.
        visual_dataset = _records_for_split(
            data_root,
            "seen_discrete_test",
            maps,
            include_targets=True,
            smoke=smoke,
            samples_per_map=int(_config_value(config, "smoke_samples_per_map", 10)),
        )
        visual_dir = output_dir / "visualizations"
        _render_localization_visuals(
            records=visual_dataset.records,
            predictions=ordered,
            output_dir=visual_dir,
            maps=maps,
            seed=seed,
            samples_per_map=int(_config_value(config, "visual_samples_per_map", 10)),
        )
        provenance.update(
            {
                "status": "complete",
                "completed_count": len(ordered),
                "resumed_missing_count": len(expected_ids - set(prediction_file_ids)) if resume else 0,
                "expected_sample_ids_sha256": expected_ids_digest,
            }
        )
        _json_dump(manifest_path, provenance)
        _json_dump(work_provenance_path, provenance)
        # A completed output can be reused on future invocations without
        # changing bytes, while shard files are always disposable intermediates.
        for rank in range(ctx.world_size):
            (output_dir / f".predictions.rank_{rank}.jsonl").unlink(missing_ok=True)
    ctx.barrier()
    finish_distributed(ctx)
    return prediction_path


def eval_command(config: Mapping[str, Any], *, seed: int, smoke: bool = False) -> int:
    """Invoke the independent shared evaluator for the requested output tree."""

    import subprocess

    if not smoke and _selected_maps(config) != SEEN10_MAPS:
        raise ValueError("Formal Seen-10 evaluation requires all ten maps in the published order")
    data_root = _data_root(config)
    run_dir = _smoke_output_dir(config, seed) if smoke else _seed_output_dir(config, seed)
    pred_root = run_dir / "localization"
    output = run_dir / "evaluation" / ("smoke_localization.json" if smoke else "localization")
    evaluator = (
        Path(os.environ.get("SHARED_EVAL_DIR", config.get("shared_eval_dir", DEFAULT_SHARED_EVAL_DIR)))
        / "run_eval.py"
    )
    evaluator_python = os.environ.get(
        "UNILIP_PYTHON",
        str(config.get("unilip_python", "/home/jiahao/miniconda3/envs/UniLIP/bin/python")),
    )
    if smoke:
        command = [
            evaluator_python,
            str(evaluator),
            "smoke",
            "localization",
            "--pred-root",
            str(pred_root),
            "--data-root",
            str(data_root),
            "--limit",
            "1",
        ]
    else:
        command = [
            evaluator_python,
            str(evaluator),
            "localization",
            "--pred-root",
            str(pred_root),
            "--data-root",
            str(data_root),
            "--output",
            str(output),
        ]
    return subprocess.call(command)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/csgo_seen10.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser


__all__ = [
    "DistributedContext",
    "eval_command",
    "inference",
    "init_distributed",
    "load_config",
    "train",
]
