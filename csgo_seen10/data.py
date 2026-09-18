"""Manifest driven data access for the CSGO Benchmark v2 Seen-10 split.

The benchmark bundle is deliberately kept outside this repository.  This
module reads the published split files through the read-only shared evaluator
protocol and exposes the resulting records to the model integration layer.
Image decoding and model/processor collation stay out of this module: each
sample contains absolute FPV and radar paths, plus an optional target pose.

The normalized pose order is ``[x, y, z, pitch, yaw]``.  X and Y are divided
by 1024, Z uses the frozen per-map calibration, and both angles are divided by
360 degrees (the split stores angles in radians, so the protocol converts
them consistently).  Values are never clipped.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover - allows metadata-only inspection
    class _TorchDataset:  # type: ignore[no-redef]
        """Small fallback so the metadata reader remains importable without torch."""

        pass


SEEN10_MAPS = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)

SEEN10_SPLITS = (
    "seen_train",
    "seen_validation",
    "seen_discrete_test",
    "seen_continuous",
)

SEEN10_COUNTS = {
    "seen_train": {map_name: 5_000 for map_name in SEEN10_MAPS},
    "seen_validation": {map_name: 500 for map_name in SEEN10_MAPS},
    "seen_discrete_test": {map_name: 2_000 for map_name in SEEN10_MAPS},
    "seen_continuous": {map_name: 20 * 64 for map_name in SEEN10_MAPS},
}

_SPLIT_ALIASES = {
    "train": "seen_train",
    "validation": "seen_validation",
    "discrete_test": "seen_discrete_test",
    "continuous": "seen_continuous",
}

_DEFAULT_DATA_ROOT = "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"
_DEFAULT_SHARED_EVAL_DIR = "/home/jiahao/task/csgo_benchmark_v2_eval_general"


def _shared_protocol_class() -> type:
    """Load ``BenchmarkData`` from the shared evaluator without importing it.

    The evaluator directory is intentionally not installed as a package.  A
    file based import keeps this project independent of evaluator metrics and
    makes the data contract explicit while reusing its strict manifest,
    calibration, path and count validation.
    """

    shared_dir = Path(os.environ.get("SHARED_EVAL_DIR", _DEFAULT_SHARED_EVAL_DIR)).expanduser()
    protocol_path = (shared_dir / "protocol.py").resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(
            "Shared Benchmark v2 protocol not found at "
            f"{protocol_path}; set SHARED_EVAL_DIR to the evaluator directory"
        )

    module_name = "_csgo_benchmark_v2_shared_protocol"
    loaded = sys.modules.get(module_name)
    if loaded is None or Path(getattr(loaded, "__file__", "")).resolve() != protocol_path:
        spec = importlib.util.spec_from_file_location(module_name, protocol_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load shared protocol from {protocol_path}")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        spec.loader.exec_module(loaded)
    benchmark_data = getattr(loaded, "BenchmarkData", None)
    if benchmark_data is None:
        raise ImportError(f"Shared protocol {protocol_path} has no BenchmarkData class")
    return benchmark_data


def _canonical_split(split: str) -> str:
    canonical = _SPLIT_ALIASES.get(split, split)
    if canonical not in SEEN10_SPLITS:
        raise ValueError(f"Unsupported Seen-10 split {split!r}; choose from {SEEN10_SPLITS}")
    return canonical


def _selected_maps(maps: Sequence[str] | None) -> tuple[str, ...]:
    """Validate map overrides while preserving the published map order."""

    if maps is None:
        return SEEN10_MAPS
    if isinstance(maps, (str, bytes)):
        raise ValueError("maps must be a sequence of map names, not a string")
    requested = tuple(maps)
    unknown = [map_name for map_name in requested if map_name not in SEEN10_MAPS]
    if unknown:
        raise ValueError(f"Unknown Seen-10 maps: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("maps contains duplicates")
    return tuple(map_name for map_name in SEEN10_MAPS if map_name in requested)


def _z_bounds(
    z_calibration: Mapping[str, Any] | Sequence[float] | None = None,
    *,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[float, float]:
    """Resolve and validate a published per-map Z range."""

    if z_calibration is not None:
        if isinstance(z_calibration, Mapping):
            if z_min is None:
                z_min = z_calibration.get("z_min", z_calibration.get("min_z"))
            if z_max is None:
                z_max = z_calibration.get("z_max", z_calibration.get("max_z"))
        elif len(z_calibration) == 2:
            if z_min is None:
                z_min = z_calibration[0]
            if z_max is None:
                z_max = z_calibration[1]
        else:
            raise ValueError("z_calibration must be a {z_min, z_max} mapping or a two-value sequence")
    if z_min is None or z_max is None:
        raise ValueError("A published z_min and z_max are required for pose conversion")
    z_min, z_max = float(z_min), float(z_max)
    if not math.isfinite(z_min) or not math.isfinite(z_max) or z_max <= z_min:
        raise ValueError(f"Invalid z calibration: z_min={z_min}, z_max={z_max}")
    return z_min, z_max


def _pose_values(
    pose: Mapping[str, Any] | Sequence[float], *, angles_in_radians: bool = False
) -> tuple[float, float, float, float, float]:
    """Read physical ``[x, y, z, pitch, yaw]`` values.

    A split row uses ``angle_v`` and ``angle_h`` in radians; mappings with
    those names are converted to degrees.  Mappings using ``pitch``/``yaw``
    and ordinary sequences are interpreted as physical degrees unless
    ``angles_in_radians`` is explicitly requested.
    """

    if isinstance(pose, Mapping):
        try:
            x, y, z = (float(pose[key]) for key in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Pose mapping needs numeric x, y and z fields") from exc
        if "angle_v" in pose or "angle_h" in pose:
            try:
                pitch, yaw = float(pose["angle_v"]), float(pose["angle_h"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Pose mapping needs numeric angle_v and angle_h fields") from exc
            angles_in_radians = True
        else:
            try:
                pitch, yaw = float(pose["pitch"]), float(pose["yaw"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Pose mapping needs numeric pitch and yaw fields") from exc
    else:
        if len(pose) != 5:
            raise ValueError(f"Pose sequence must contain five values, got {len(pose)}")
        try:
            x, y, z, pitch, yaw = (float(value) for value in pose)
        except (TypeError, ValueError) as exc:
            raise ValueError("Pose sequence must contain numeric values") from exc

    if angles_in_radians:
        pitch, yaw = math.degrees(pitch), math.degrees(yaw)
    values = (x, y, z, pitch, yaw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Pose contains a non-finite value: {values!r}")
    return values


def normalize_pose(
    pose: Mapping[str, Any] | Sequence[float],
    z_calibration: Mapping[str, Any] | Sequence[float] | None = None,
    *,
    z_min: float | None = None,
    z_max: float | None = None,
    angles_in_radians: bool = False,
) -> list[float]:
    """Normalize one physical pose to Benchmark v2's five-value order.

    Sequence and ``pitch``/``yaw`` mappings use degrees by default.  A split
    row mapping with ``angle_v``/``angle_h`` is recognized as radians.  The
    returned values are not clipped, matching the evaluator's prediction
    contract.
    """

    x, y, z, pitch, yaw = _pose_values(pose, angles_in_radians=angles_in_radians)
    low, high = _z_bounds(z_calibration, z_min=z_min, z_max=z_max)
    return [x / 1024.0, y / 1024.0, (z - low) / (high - low), pitch / 360.0, yaw / 360.0]


def denormalize_pose(
    pose: Sequence[float],
    z_calibration: Mapping[str, Any] | Sequence[float] | None = None,
    *,
    z_min: float | None = None,
    z_max: float | None = None,
) -> list[float]:
    """Convert a normalized Benchmark v2 pose to physical ``xyzhw`` values.

    Physical output uses x/y/z coordinate units and pitch/yaw in degrees.
    No clipping or angle wrapping is applied, so out-of-range predictions are
    retained for metrics and are only edge-clipped by the visualizer markers.
    """

    if len(pose) != 5:
        raise ValueError(f"Normalized pose must contain five values, got {len(pose)}")
    try:
        x, y, z, pitch, yaw = (float(value) for value in pose)
    except (TypeError, ValueError) as exc:
        raise ValueError("Normalized pose must contain numeric values") from exc
    if not all(math.isfinite(value) for value in (x, y, z, pitch, yaw)):
        raise ValueError("Normalized pose contains a non-finite value")
    low, high = _z_bounds(z_calibration, z_min=z_min, z_max=z_max)
    return [x * 1024.0, y * 1024.0, z * (high - low) + low, pitch * 360.0, yaw * 360.0]


def localization_instruction(map_name: str) -> str:
    """Return the fixed localization task instruction for one map."""

    if map_name not in SEEN10_MAPS:
        raise ValueError(f"Unknown Seen-10 map: {map_name!r}")
    return (
        f"Localize the camera in Counter-Strike 2 map '{map_name}' from the "
        "first-person view and radar map. Predict the absolute normalized "
        "5DoF pose [x, y, z, pitch, yaw]."
    )


def _metadata_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep model-facing metadata while excluding all pose targets."""

    map_name = str(record["map_name"])
    result: dict[str, Any] = {
        "sample_id": str(record["sample_id"]),
        "map_name": map_name,
        "file_frame": str(record["file_frame"]),
        "image_path": str(record["image_path"]),
        "radar_path": str(record["radar_path"]),
        # Explicit aliases make the contract easy for a collator to consume.
        "fpv_path": str(record["image_path"]),
        "map_path": str(record["radar_path"]),
        "instruction": localization_instruction(map_name),
        "z_calibration": dict(record["z_calibration"]),
        "clip_id": record.get("clip_id"),
        "frame_index": record.get("frame_index"),
        "file_num": record.get("file_num"),
        "frame_id": record.get("frame_id"),
    }
    return result


class Seen10Dataset(_TorchDataset):
    """PyTorch-compatible manifest-driven Seen-10 dataset.

    ``records``/``rows`` retain the strict shared-protocol records for output
    joining and visualization.  ``__getitem__`` returns paths and metadata;
    the caller owns image loading and processor collation.  Set
    ``include_targets=False`` for inference so no normalized or raw GT pose is
    present in any sample handed to the model.
    """

    def __init__(
        self,
        data_root: str | os.PathLike[str] | None = None,
        *,
        split: str = "seen_train",
        maps: Sequence[str] | None = None,
        include_targets: bool = True,
        max_samples: int | None = None,
    ) -> None:
        if data_root is None:
            data_root = os.environ.get("CSGO_DATA_ROOT", os.environ.get("DATA_ROOT", _DEFAULT_DATA_ROOT))
        canonical_split = _canonical_split(split)
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        self.data_root = Path(data_root).expanduser().resolve()
        benchmark_data = _shared_protocol_class()(self.data_root)
        if tuple(benchmark_data.maps) != SEEN10_MAPS:
            raise ValueError(f"Published map order differs from Seen-10: {benchmark_data.maps!r}")
        self._benchmark_data = benchmark_data
        self.split = canonical_split
        self.maps = _selected_maps(maps)
        self.include_targets = bool(include_targets)
        # BenchmarkData performs manifest, report, calibration, split count,
        # path, identity and continuous clip-order validation.
        self.records = list(benchmark_data.rows(canonical_split, maps=self.maps, max_samples=max_samples))
        self.rows = self.records

    @property
    def z_ranges(self) -> Mapping[str, Mapping[str, float]]:
        """Frozen per-map calibration used by the shared protocol."""

        return self._benchmark_data.z_ranges

    def clips(self, *, max_clips: int | None = None) -> list[dict[str, Any]]:
        """Return ordered continuous clips for this dataset's selected maps."""

        if self.split != "seen_continuous":
            raise ValueError("clips() is only available for split='seen_continuous'")
        return self._benchmark_data.clips("seen_continuous", maps=self.maps, max_clips=max_clips)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self.records)
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        record = self.records[index]
        item = _metadata_record(record)
        if self.include_targets:
            normalized = [float(value) for value in record["pose"]]
            item.update(
                {
                    "target_pose": normalized,
                    # ``target`` is a short collator-friendly alias; all
                    # labels stay separate from the image/instruction fields.
                    "target": normalized,
                    "pose": normalized,
                    "pose_raw": dict(record["pose_raw"]),
                    "target_pose_physical": denormalize_pose(
                        normalized, record["z_calibration"]
                    ),
                }
            )
        return item

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Iterate full shared-protocol records for joining predictions/GT."""

        yield from self.records


# Name aliases used by small integration wrappers.
ManifestSeen10Dataset = Seen10Dataset
CSGOSeen10Dataset = Seen10Dataset


__all__ = [
    "SEEN10_COUNTS",
    "SEEN10_MAPS",
    "SEEN10_SPLITS",
    "CSGOSeen10Dataset",
    "ManifestSeen10Dataset",
    "Seen10Dataset",
    "denormalize_pose",
    "localization_instruction",
    "normalize_pose",
]
