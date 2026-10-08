"""Pure validation and position mapping for a single-rank topology expansion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_TOPOLOGY_KEYS = frozenset({"batch_size", "grad_accumulation_steps", "world_size"})


def _positive_int(settings: Mapping[str, Any], key: str) -> int:
    value = settings.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Resume training setting {key} must be a positive integer: {value!r}")
    return value


def validate_resume_settings(
    saved: Mapping[str, Any], current: Mapping[str, Any], *, sampler_policy: str
) -> bool:
    """Validate settings and return whether a supported 1-to-N expansion is needed."""

    if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
        raise ValueError("Resume training settings must be mappings")
    saved_world = _positive_int(saved, "world_size")
    current_world = _positive_int(current, "world_size")
    expanding = saved_world == 1 and current_world > 1
    allowed_changes = _TOPOLOGY_KEYS if expanding else frozenset()
    mismatches = {
        key: (saved.get(key), value)
        for key, value in current.items()
        if key not in allowed_changes and saved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume training settings differ from checkpoint: {mismatches}")
    if not expanding:
        return False
    if sampler_policy != "global_full_update_batches":
        raise ValueError(
            "Single-rank checkpoint expansion requires sampler_policy="
            "global_full_update_batches"
        )
    old_effective = (
        _positive_int(saved, "batch_size")
        * _positive_int(saved, "grad_accumulation_steps")
        * saved_world
    )
    new_effective = (
        _positive_int(current, "batch_size")
        * _positive_int(current, "grad_accumulation_steps")
        * current_world
    )
    if old_effective != new_effective:
        raise ValueError(
            "Resume effective batch differs across topology expansion: "
            f"saved={old_effective}, current={new_effective}"
        )
    return True


def remap_resume_position(
    state: Mapping[str, Any],
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    dataset_size: int,
) -> dict[str, Any]:
    """Map a completed global update boundary to the expanded topology."""

    if type(dataset_size) is not int or dataset_size < 0:
        raise ValueError(f"Resume dataset_size must be a nonnegative integer: {dataset_size!r}")
    old_accum = _positive_int(saved, "grad_accumulation_steps")
    new_accum = _positive_int(current, "grad_accumulation_steps")
    effective = _positive_int(saved, "batch_size") * old_accum * _positive_int(saved, "world_size")
    new_effective = _positive_int(current, "batch_size") * new_accum * _positive_int(current, "world_size")
    if effective != new_effective:
        raise ValueError("Resume effective batch differs across topology expansion")
    updates_per_epoch = dataset_size // effective
    if updates_per_epoch <= 0:
        raise ValueError("Resume dataset contains no complete global update")
    position = state.get("data_position")
    if not isinstance(position, Mapping):
        raise ValueError("Resume checkpoint has no valid data_position")
    epoch, next_batch, step = position.get("epoch"), position.get("next_batch"), state.get("step")
    for name, value in (("epoch", epoch), ("next_batch", next_batch), ("step", step)):
        if type(value) is not int or value < 0:
            raise ValueError(f"Resume {name} must be a nonnegative integer: {value!r}")
    max_old_batches = updates_per_epoch * old_accum
    if next_batch > max_old_batches:
        raise ValueError(
            f"Resume next_batch={next_batch} exceeds epoch boundary {max_old_batches}"
        )
    if next_batch % old_accum:
        raise ValueError(
            f"Resume next_batch={next_batch} is not a complete update "
            f"(accumulation={old_accum})"
        )
    updates_within_epoch = next_batch // old_accum
    expected_step = epoch * updates_per_epoch + updates_within_epoch
    if step != expected_step:
        raise ValueError(
            f"Resume step={step} disagrees with epoch={epoch}, next_batch={next_batch}; "
            f"expected step={expected_step}"
        )
    new_epoch = epoch + 1 if updates_within_epoch == updates_per_epoch else epoch
    new_next_batch = 0 if new_epoch != epoch else updates_within_epoch * new_accum
    return {
        "old_position": {"epoch": epoch, "next_batch": next_batch},
        "new_position": {"epoch": new_epoch, "next_batch": new_next_batch},
        "updates_per_epoch": updates_per_epoch,
        "updates_within_epoch": updates_within_epoch,
        "effective_batch_size": effective,
        "global_step": step,
    }
