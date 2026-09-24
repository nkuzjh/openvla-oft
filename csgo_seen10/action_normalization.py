"""OFT Q99 action transform for Benchmark v2's *external* normalized pose.

The benchmark pose is already normalized by the published map calibration.
This module adds an optional, model-internal OFT transform; it must never be
used to recompute the benchmark's calibration or to fit validation/test data.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .data import SEEN10_MAPS


ACTION_DIM = 5
EPSILON = 1e-8
EXPECTED_TRAIN_COUNT = 50_000
_STATS_KEYS = ("q01", "q99", "min", "max", "mask")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _validate_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")


def validate_stats(stats: Mapping[str, Any], *, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Validate provenance and statistics, including the stored content hash."""

    required = {
        "schema_version", "source_split", "sample_count", "manifest_sha256",
        "ordered_sample_ids_sha256", "q01", "q99", "min", "max", "mask", "stats_sha256",
    }
    if not isinstance(stats, Mapping) or set(stats) != required:
        raise ValueError(f"Q99 statistics need exactly {sorted(required)}")
    result = dict(stats)
    if result["schema_version"] != 1 or result["source_split"] != "seen_train":
        raise ValueError("Q99 statistics must be fitted from the published seen_train split")
    if not isinstance(result["sample_count"], int) or result["sample_count"] <= 0:
        raise ValueError("Q99 sample_count must be a positive integer")
    _validate_hash(result["manifest_sha256"], "manifest_sha256")
    _validate_hash(result["ordered_sample_ids_sha256"], "ordered_sample_ids_sha256")
    _validate_hash(result["stats_sha256"], "stats_sha256")
    if expected_manifest_sha256 is not None and result["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("Q99 statistics were fitted against a different manifest")
    for key in _STATS_KEYS:
        value = result[key]
        if not isinstance(value, list) or len(value) != ACTION_DIM:
            raise ValueError(f"Q99 {key} must have {ACTION_DIM} values")
    if result["mask"] != [True] * ACTION_DIM:
        raise ValueError("Every pose dimension, including yaw, must use Q99 normalization")
    for key in ("q01", "q99", "min", "max"):
        try:
            result[key] = [float(v) for v in result[key]]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Q99 {key} contains a non-numeric value") from exc
        if not all(math.isfinite(v) for v in result[key]):
            raise ValueError(f"Q99 {key} contains a non-finite value")
    for j in range(ACTION_DIM):
        lo, hi = result["min"][j], result["max"][j]
        qlo, qhi = result["q01"][j], result["q99"][j]
        if not lo <= qlo <= qhi <= hi:
            raise ValueError(f"Invalid Q99 order in dimension {j}")
        if lo != hi and qlo == qhi:
            raise ValueError(f"Nonconstant pose dimension {j} has q01 == q99")
    if result["stats_sha256"] != _sha256_json({k: v for k, v in result.items() if k != "stats_sha256"}):
        raise ValueError("Q99 statistics content hash does not match")
    return result


def fit_seen_train_stats(
    records: Sequence[Mapping[str, Any]] | Any,
    manifest_sha256: str,
    *,
    expected_count: int = EXPECTED_TRAIN_COUNT,
) -> dict[str, Any]:
    """Fit all five Q99 dimensions on the complete published training split.

    ``records`` can be a Seen10Dataset or a sequence whose every row has an
    explicit ``split='seen_train'`` marker.  The full-count check prevents a
    sampler's dropped tail or a small debugging subset from being used as the
    deployed normalization statistics.
    """

    _validate_hash(manifest_sha256, "manifest_sha256")
    dataset_split = getattr(records, "split", None)
    if dataset_split is not None and dataset_split != "seen_train":
        raise ValueError("Q99 statistics may only be fitted from seen_train")
    rows = getattr(records, "records", records)
    if len(rows) != expected_count:
        raise ValueError(f"Expected all {expected_count} seen_train records; got {len(rows)}")
    if not rows:
        raise ValueError("Cannot fit Q99 statistics on an empty split")
    ids: list[str] = []
    poses: list[list[float]] = []
    for row in rows:
        row_split = row.get("split", dataset_split)
        if row_split != "seen_train":
            raise ValueError("Every Q99 source row must be marked seen_train")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Every Q99 source row needs a sample_id")
        pose = row.get("target_pose", row.get("pose"))
        if pose is None or len(pose) != ACTION_DIM:
            raise ValueError(f"Q99 source row {sample_id} needs a normalized 5D pose")
        values = [float(v) for v in pose]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Q99 source row {sample_id} contains a non-finite pose")
        ids.append(sample_id)
        poses.append(values)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs in Q99 source split")
    if expected_count == EXPECTED_TRAIN_COUNT:
        map_counts = {name: 0 for name in SEEN10_MAPS}
        for row in rows:
            name = row.get("map_name")
            if name not in map_counts or not str(row["sample_id"]).startswith(f"{name}/"):
                raise ValueError("Q99 source row has an invalid Seen-10 map identity")
            map_counts[name] += 1
        if any(count != 5_000 for count in map_counts.values()):
            raise ValueError(f"Q99 fit requires 5,000 samples per Seen-10 map: {map_counts}")

    values = np.asarray(poses, dtype=np.float64)
    q01 = np.quantile(values, 0.01, axis=0).tolist()
    q99 = np.quantile(values, 0.99, axis=0).tolist()
    stats: dict[str, Any] = {
        "schema_version": 1,
        "source_split": "seen_train",
        "sample_count": len(rows),
        "manifest_sha256": manifest_sha256,
        "ordered_sample_ids_sha256": hashlib.sha256("".join(f"{i}\n" for i in ids).encode("utf-8")).hexdigest(),
        "q01": q01,
        "q99": q99,
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mask": [True] * ACTION_DIM,
    }
    stats["stats_sha256"] = _sha256_json(stats)
    return validate_stats(stats)


class ActionNormalization:
    """Internal continuous-action transform; identity preserves old runs."""

    def __init__(self, mode: str = "none", stats: Mapping[str, Any] | None = None) -> None:
        if mode not in ("none", "bounds_q99"):
            raise ValueError(f"Unsupported action normalization mode {mode!r}")
        if mode == "bounds_q99" and stats is None:
            raise ValueError("bounds_q99 requires frozen seen_train statistics")
        if mode == "none" and stats is not None:
            raise ValueError("Identity action normalization must not carry Q99 statistics")
        self.mode = mode
        self.stats = validate_stats(stats) if stats is not None else None

    def _check(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != ACTION_DIM:
            raise ValueError("Action normalization expects a torch tensor ending in five pose dimensions")
        if not torch.is_floating_point(value):
            raise ValueError("Action normalization expects floating-point pose values")

    def _vector(self, key: str, like: torch.Tensor) -> torch.Tensor:
        assert self.stats is not None
        return torch.as_tensor(self.stats[key], dtype=like.dtype, device=like.device)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """Convert external normalized pose to clipped OFT training target."""

        self._check(value)
        if self.mode == "none":
            return value
        q01 = self._vector("q01", value)
        q99 = self._vector("q99", value)
        result = torch.clamp(2.0 * (value - q01) / (q99 - q01 + EPSILON) - 1.0, -1.0, 1.0)
        constant = self._vector("min", value) == self._vector("max", value)
        return torch.where(constant, torch.zeros_like(result), result)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Convert a model prediction to benchmark pose without prediction clip."""

        self._check(value)
        if self.mode == "none":
            return value
        q01 = self._vector("q01", value)
        q99 = self._vector("q99", value)
        # Match the native OFT inverse exactly: constant dimensions also use
        # the epsilon in the denominator's inverse, with no extra override.
        return 0.5 * (value + 1.0) * (q99 - q01 + EPSILON) + q01

    denormalize = inverse

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "stats": dict(self.stats) if self.stats is not None else None}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionNormalization":
        if set(value) != {"mode", "stats"}:
            raise ValueError("Action normalization state needs mode and stats")
        return cls(mode=value["mode"], stats=value["stats"])

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "ActionNormalization":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["ACTION_DIM", "EPSILON", "ActionNormalization", "fit_seen_train_stats", "validate_stats"]
