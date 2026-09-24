"""Bounded acceptance checks for the approved Seen-10 aligned recipe.

This diagnostic never invokes the training, inference, or evaluation entry
points.  By default it checks configuration, real train-only Q99 statistics,
the exact update sampler, and checkpoint event/link rules on CPU.  The
explicit ``--gpu-smoke`` flag additionally performs at most two optimizer
updates on one real Seen-10 training sample, then removes its temporary
checkpoint.  It never writes benchmark predictions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "CSGO")

import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from csgo_seen10.action_normalization import ActionNormalization, fit_seen_train_stats
from csgo_seen10.data import SEEN10_MAPS
from csgo_seen10.model import collate_samples, create_model, forward_action, save_component_checkpoint
from csgo_seen10.runner import (
    _checkpoint_path,
    _config_value,
    _data_root,
    _model_kwargs,
    _model_path,
    _records_for_split,
    _safe_link,
    _validate_aligned_config,
    load_config,
    parameter_audit,
)
from csgo_seen10.sampling import GlobalUpdateSampler, event_steps


EXPECTED_EVENTS = [4_000, 8_000, 12_000, 16_000, 19_500]
EXPECTED_TOTAL_PARAMETERS = 7_739_887_045
EXPECTED_TRAINABLE_PARAMETERS = 198_649_861


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sampler_and_events() -> dict[str, Any]:
    if event_steps(19_500, 4_000) != EXPECTED_EVENTS:
        raise AssertionError("Checkpoint events do not include exact final step")
    dataset = range(50_000)
    world_results = {}
    baseline = None
    for world_size in (1, 2, 4):
        samplers = [
            GlobalUpdateSampler(
                dataset, effective_batch_size=128, microbatch_size=1,
                accumulation_steps=128 // world_size, rank=rank,
                world_size=world_size, seed=42,
            )
            for rank in range(world_size)
        ]
        rank_data = []
        for sampler in samplers:
            sampler.set_epoch(0)
            rank_data.append(list(sampler))
            if len(rank_data[-1]) != 49_920 // world_size:
                raise AssertionError("Sampler local rank length changed")
        local_size = 128 // world_size
        reconstructed = []
        for update in range(390):
            batch = []
            for rank_data_one in rank_data:
                batch.extend(index for index, epoch in rank_data_one[update * local_size:(update + 1) * local_size])
            if len(batch) != 128 or len(set(batch)) != 128:
                raise AssertionError("Global update contains padding or duplicates")
            reconstructed.extend(batch)
        if len(set(reconstructed)) != 49_920:
            raise AssertionError("Epoch has repeated sample IDs")
        if baseline is None:
            baseline = reconstructed
        elif reconstructed != baseline:
            raise AssertionError("GPU count changes global sample grouping")
        world_results[str(world_size)] = {
            "updates_per_epoch": 390,
            "used_per_epoch": 49_920,
            "dropped_per_epoch": 80,
            "local_length": len(rank_data[0]),
            "global_order_sha256": hashlib.sha256(
                torch.tensor(reconstructed, dtype=torch.int64).numpy().astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        }
    if 390 * 50 != 19_500 or 49_920 * 50 != 2_496_000:
        raise AssertionError("Planned update and exposure totals changed")
    # A resumed run at an update boundary must see the same suffix of the
    # logical epoch.  No persistent iterator state is required.
    sampler = GlobalUpdateSampler(
        dataset, effective_batch_size=128, microbatch_size=1,
        accumulation_steps=128, rank=0, world_size=1, seed=42,
    )
    sampler.set_epoch(7)
    full = list(sampler)
    remaining_from_step_17 = list(sampler)[17 * 128:]
    if remaining_from_step_17 != full[17 * 128:]:
        raise AssertionError("Sampler resume suffix differs")
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        checkpoint_root = run_dir / "checkpoints"
        for step in EXPECTED_EVENTS:
            _checkpoint_path(run_dir, step).mkdir(parents=True)
        best_step = 12_000
        _safe_link(checkpoint_root / "best", _checkpoint_path(run_dir, best_step))
        _safe_link(checkpoint_root / "late", _checkpoint_path(run_dir, 19_500))
        if (checkpoint_root / "best").resolve() != _checkpoint_path(run_dir, best_step):
            raise AssertionError("best checkpoint link resolves incorrectly")
        if (checkpoint_root / "late").resolve() != _checkpoint_path(run_dir, 19_500):
            raise AssertionError("late checkpoint link resolves incorrectly")
    return {"events": EXPECTED_EVENTS, "world_sizes": world_results,
            "planned_updates": 19_500, "planned_localization_exposures": 2_496_000,
            "best_late_links": "pass", "resume_suffix": "pass"}


def _check_real_stats(config: dict[str, Any]) -> tuple[ActionNormalization, Any, dict[str, Any]]:
    data_root = Path(_data_root(config))
    manifest_hash = _sha256(data_root / "benchmark_manifest.json")
    dataset = _records_for_split(str(data_root), "seen_train", SEEN10_MAPS, include_targets=True)
    stats = fit_seen_train_stats(dataset.records, manifest_hash)
    transform = ActionNormalization("bounds_q99", stats)
    if len(dataset) != 50_000 or stats["source_split"] != "seen_train":
        raise AssertionError("Q99 did not fit the complete Seen-10 training split")
    sample = torch.tensor([dataset[0]["target_pose"]], dtype=torch.float32)
    internal = transform.normalize(sample)
    recovered = transform.inverse(internal)
    if not torch.isfinite(internal).all() or not torch.isfinite(recovered).all():
        raise AssertionError("Real Q99 transform produced a nonfinite value")
    all_poses = np.asarray([record["target_pose"] for record in dataset.records], dtype=np.float64)
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    lower_clip_fraction = np.mean(all_poses < q01, axis=0).tolist()
    upper_clip_fraction = np.mean(all_poses > q99, axis=0).tolist()
    return transform, dataset, {
        "sample_count": stats["sample_count"],
        "manifest_sha256": manifest_hash,
        "ordered_sample_ids_sha256": stats["ordered_sample_ids_sha256"],
        "stats_sha256": stats["stats_sha256"],
        "q01": stats["q01"], "q99": stats["q99"],
        "lower_clip_fraction": lower_clip_fraction,
        "upper_clip_fraction": upper_clip_fraction,
        "example_external": sample[0].tolist(),
        "example_target": internal[0].tolist(),
        "example_inverse_of_clipped_target": recovered[0].tolist(),
    }


def _first_named_parameter(model: torch.nn.Module, predicate) -> tuple[str, torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        if predicate(name, parameter):
            return name, parameter
    raise AssertionError("Expected model parameter not found")


def _check_gpu_smoke(config: dict[str, Any], transform: ActionNormalization, dataset: Any,
                     out_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no GPU smoke was run")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < 45 * (1024 ** 3):
        raise RuntimeError(f"GPU 0 has only {free_bytes / 2**30:.1f} GiB free; 45 GiB is required")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0")
    bundle = create_model(_model_path(config), device=device,
                          **_model_kwargs(config, transform))
    bundle.vla.train()
    bundle.action_head.train()
    base = bundle.base_vla
    if base is None:
        raise AssertionError("Missing base model for freeze audit")
    vision_name, vision = _first_named_parameter(base.vision_backbone, lambda n, p: True)
    projector_name, projector = _first_named_parameter(base.projector, lambda n, p: True)
    if any(p.requires_grad for p in base.vision_backbone.parameters()) or any(
        p.requires_grad for p in base.projector.parameters()
    ):
        raise AssertionError("Vision or VL projector still has trainable parameters")
    frozen_before = (vision.detach().cpu().clone(), projector.detach().cpu().clone())
    lo_name, lo_parameter = _first_named_parameter(
        bundle.vla, lambda n, p: p.requires_grad and "lora_B" in n and "lm_head" not in n
    )
    head_name, head_parameter = _first_named_parameter(
        bundle.action_head, lambda n, p: p.requires_grad and n.endswith("fc2.bias")
    )
    lo_before = lo_parameter.detach().cpu().clone()
    head_before = head_parameter.detach().cpu().clone()
    proprio_calls: list[dict[str, int]] = []
    native_proprio = base._process_proprio_features

    def checked_proprio(projected, proprio, proprio_projector):
        if proprio is not None or proprio_projector is not None:
            raise AssertionError("A proprio state or projector entered the model forward")
        result = native_proprio(projected, proprio, proprio_projector)
        if result.shape[1] != projected.shape[1]:
            raise AssertionError("The model added a state token to visual tokens")
        proprio_calls.append({"before": projected.shape[1], "after": result.shape[1]})
        return result

    base._process_proprio_features = checked_proprio
    trainable = [p for p in list(bundle.vla.parameters()) + list(bundle.action_head.parameters()) if p.requires_grad]
    optimizer = AdamW(trainable, lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler = MultiStepLR(optimizer, milestones=[100_000], gamma=0.1)
    audit = parameter_audit(bundle, optimizer)
    (out_dir / "parameter_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if audit["total_parameters"] != EXPECTED_TOTAL_PARAMETERS or audit["declared_trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError(f"Unexpected model parameter counts: {audit['total_parameters']}, {audit['declared_trainable_parameters']}")
    if len(audit["optimizer_groups"]) != 1 or audit["optimizer_groups"][0]["lr"] != 5e-4:
        raise AssertionError("Official single optimizer group not preserved")
    if any(key in dataset[0] for key in ("proprio", "state", "robot_state")):
        raise AssertionError("Training record contains a state input")
    batch = collate_samples(
        [dict(dataset[0], _augmentation_epoch=0)], bundle.processor,
        include_targets=True, training=True,
        fpv_augmentation="oft_photometric_only", radar_augmentation="oft_photometric_only",
        augmentation_seed=42,
    )
    if tuple(batch["actions"].shape) != (1, 1, 5):
        raise AssertionError("Collated action target is not [1,1,5]")
    result = {"gpu_free_gib_before": round(free_bytes / 2**30, 2),
              "gpu_total_gib": round(total_bytes / 2**30, 2),
              "total_parameters": audit["total_parameters"],
              "declared_trainable_parameters": audit["declared_trainable_parameters"],
              "optimizer_group_count": len(audit["optimizer_groups"]),
              "vision_probe": vision_name, "projector_probe": projector_name,
              "lora_probe": lo_name, "head_probe": head_name,
              "target_shape": list(batch["actions"].shape), "updates": []}
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, output = forward_action(bundle, batch, device=device, train=True)
        if not math.isfinite(float(loss.detach())) or not torch.isfinite(output).all():
            raise AssertionError("Nonfinite loss or externally normalized prediction")
        if tuple(output.shape) != (1, 1, 5):
            raise AssertionError("Prediction is not [1,1,5]")
        loss.backward()
        if any(p.grad is not None for p in base.vision_backbone.parameters()) or any(
            p.grad is not None for p in base.projector.parameters()
        ):
            raise AssertionError("Frozen vision/projector received gradients")
        if lo_parameter.grad is None or head_parameter.grad is None:
            raise AssertionError("LLM LoRA or action head has no gradient")
        lora_grad_norm = float(lo_parameter.grad.detach().float().norm().item())
        head_grad_norm = float(head_parameter.grad.detach().float().norm().item())
        lm_head_grads = [p.grad for n, p in bundle.vla.named_parameters() if "lm_head" in n and p.requires_grad]
        if not lm_head_grads or any(g is not None for g in lm_head_grads):
            raise AssertionError("Declared lm_head LoRA dependency differs from continuous L1 path")
        optimizer.step()
        scheduler.step()
        if optimizer.param_groups[0]["lr"] != 5e-4:
            raise AssertionError("LR decayed before official 100k milestone")
        result["updates"].append({"step": step, "loss": float(loss.detach()),
                                  "external_l1": metrics["external_loss_value"],
                                  "output_shape": list(output.shape), "lr": optimizer.param_groups[0]["lr"],
                                  "lora_probe_grad_norm": lora_grad_norm,
                                  "head_probe_grad_norm": head_grad_norm})
    lo_delta = float((lo_parameter.detach().float().cpu() - lo_before.float()).abs().max().item())
    head_delta = float((head_parameter.detach().float().cpu() - head_before.float()).abs().max().item())
    result["probe_max_abs_delta"] = {"lora": lo_delta, "head": head_delta}
    if lo_delta == 0 or head_delta == 0:
        raise AssertionError(f"LLM LoRA/head update probe delta: {lo_delta}/{head_delta}")
    if not torch.equal(frozen_before[0], vision.detach().cpu()) or not torch.equal(
        frozen_before[1], projector.detach().cpu()
    ):
        raise AssertionError("Frozen vision or projector parameter changed")
    if len(proprio_calls) != 2:
        raise AssertionError(f"Proprio path was checked {len(proprio_calls)} times for two forwards")
    result["proprio_token_checks"] = proprio_calls

    # Exercise the production component saver and loader without creating a
    # formal run.  Only one 7B model is kept alive at a time.
    with tempfile.TemporaryDirectory(prefix="csgo-aligned-smoke-") as directory:
        checkpoint_dir = Path(directory) / "checkpoint"
        bundle.vla.eval()
        bundle.action_head.eval()
        with torch.no_grad():
            _, _, prediction_before = forward_action(bundle, batch, device=device, train=False)
        prediction_before_cpu = prediction_before.cpu().clone()
        save_component_checkpoint(bundle, checkpoint_dir, 2)
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        if not (checkpoint_dir / "lora_adapter" / "adapter_model.safetensors").is_file():
            raise AssertionError("LoRA adapter checkpoint missing")
        if not (checkpoint_dir / "action_normalization.json").is_file():
            raise AssertionError("Q99 statistics checkpoint missing")
        loaded_transform = ActionNormalization.load(checkpoint_dir / "action_normalization.json")
        if loaded_transform.to_dict() != transform.to_dict():
            raise AssertionError("Checkpoint Q99 statistics changed")
        head_saved = torch.load(checkpoint_dir / "action_head--2_checkpoint.pt", map_location="cpu", weights_only=True)
        head_now = bundle.action_head.state_dict()
        if set(head_saved) != set(head_now) or any(not torch.equal(v, head_now[k].cpu()) for k, v in head_saved.items()):
            raise AssertionError("Saved action head differs from live trained head")
        from safetensors.torch import load_file
        adapter_saved = load_file(str(checkpoint_dir / "lora_adapter" / "adapter_model.safetensors"))
        if not adapter_saved:
            raise AssertionError("Saved LoRA adapter contains no tensors")
        saved_optimizer = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=False)
        saved_scheduler = torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu", weights_only=False)
        # Remove every large reference before reloading the checkpoint.
        base._process_proprio_features = native_proprio
        del head_now, head_saved, adapter_saved, loss, output, prediction_before
        del lo_parameter, head_parameter, vision, projector, trainable
        del optimizer, scheduler, bundle, base, native_proprio, checked_proprio
        gc.collect()
        torch.cuda.empty_cache()

        reloaded = create_model(
            _model_path(config), device=device, checkpoint_dir=checkpoint_dir,
            **_model_kwargs(config, transform)
        )
        reloaded.vla.eval()
        reloaded.action_head.eval()
        with torch.no_grad():
            _, _, prediction_after = forward_action(reloaded, batch, device=device, train=False)
        torch.testing.assert_close(prediction_after.cpu(), prediction_before_cpu, rtol=0, atol=1e-5)

        restored_trainable = [
            p for p in list(reloaded.vla.parameters()) + list(reloaded.action_head.parameters()) if p.requires_grad
        ]
        restored_optimizer = AdamW(
            restored_trainable, lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        restored_scheduler = MultiStepLR(restored_optimizer, milestones=[100_000], gamma=0.1)
        restored_optimizer.load_state_dict(saved_optimizer)
        restored_scheduler.load_state_dict(saved_scheduler)
        if restored_optimizer.param_groups[0]["lr"] != 5e-4 or restored_scheduler.last_epoch != 2:
            raise AssertionError("Optimizer/scheduler restore mismatch")
        if len(restored_optimizer.state) != len(saved_optimizer["state"]):
            raise AssertionError("Optimizer state count changed after reload")
        result["checkpoint_components"] = {
            "lora_tensors": len(load_file(str(checkpoint_dir / "lora_adapter" / "adapter_model.safetensors"))),
            "head_tensors": len(reloaded.action_head.state_dict()),
            "normalization_hash": transform.stats["stats_sha256"],
            "optimizer_state_entries": len(saved_optimizer["state"]),
            "scheduler_last_epoch": restored_scheduler.last_epoch,
            "prediction_max_abs_reload_diff": float(
                (prediction_after.cpu() - prediction_before_cpu).abs().max().item()
            ),
            "temporary_checkpoint_removed": True,
        }
        del restored_trainable, restored_optimizer, restored_scheduler, reloaded
        gc.collect()
        torch.cuda.empty_cache()
    result["gpu_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 2)
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/csgo_seen10_aligned_v2.yaml")
    parser.add_argument("--gpu-smoke", action="store_true", help="Run at most two tiny training updates on GPU 0")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = PROJECT_ROOT / "outputs" / "csgo_aligned_validation" / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"status": "running", "mode": "gpu_smoke" if args.gpu_smoke else "cpu_only",
                              "timestamp_utc": timestamp, "checks": {}}
    try:
        config = load_config(args.config)
        _validate_aligned_config(config)
        if _config_value(config, "batch_size", 0) != 1 or _config_value(config, "grad_accumulation_steps", 0) != 128:
            raise AssertionError("Configured single-GPU microbatch/accumulation is not 1×128")
        report["checks"]["config"] = "pass"
        report["checks"]["schedule_sampler"] = _check_sampler_and_events()
        transform, dataset, stats_report = _check_real_stats(config)
        report["checks"]["train_only_q99"] = stats_report
        if args.gpu_smoke:
            report["checks"]["gpu_smoke"] = _check_gpu_smoke(config, transform, dataset, out_dir)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_dir / "report.json")
    print(report["status"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
