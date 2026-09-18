"""Deterministic Seen-10 localization visualizations.

Every invocation chooses the same ten rows per map for a given ``seed`` and
draws their GT/prediction pairs on the published radar image.  The matching
FPV images are placed in a vertical strip on the right.  Predictions retain
their original physical values in the text labels, while only their radar
markers are clipped to the map edge for readability.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import SEEN10_MAPS, denormalize_pose

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - dependency is part of project env
    raise ImportError("Seen-10 visualization requires Pillow") from exc


# Ten stable, high-contrast colors.  GT and prediction for one sample always
# share a color; marker fill/outline distinguishes the two.
SAMPLE_COLORS = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
)


def _records(value: Any) -> list[Mapping[str, Any]]:
    if hasattr(value, "records"):
        value = value.records
    if isinstance(value, Mapping):
        value = list(value.values())
    result = list(value)
    if not all(isinstance(row, Mapping) for row in result):
        raise TypeError("records must be mappings or a dataset exposing .records")
    return result


def _canonical_sample_id(sample_id: object, map_name: str | None = None) -> str:
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("A prediction needs a non-empty sample_id")
    parts = sample_id.replace("\\", "/").split("/")
    if len(parts) > 1:
        if map_name is not None and parts[-2] != map_name:
            raise ValueError(f"sample_id map disagrees with map_name: {sample_id!r} vs {map_name!r}")
        map_name = parts[-2]
    frame = Path(parts[-1]).stem
    if not frame:
        raise ValueError(f"Invalid sample_id: {sample_id!r}")
    return f"{map_name}/{frame}" if map_name else frame


def _prediction_index(predictions: Any) -> dict[str, Any]:
    """Normalize sequence or sample-id mapping predictions to lookup keys."""

    if isinstance(predictions, Mapping):
        entries = list(predictions.items())
        result: dict[str, Any] = {}
        for key, value in entries:
            if isinstance(value, Mapping):
                map_name = value.get("map_name", value.get("map"))
                canonical = _canonical_sample_id(key, str(map_name) if map_name else None)
            else:
                canonical = _canonical_sample_id(key)
            result[canonical] = value
        return result

    result = {}
    for index, value in enumerate(predictions):
        if not isinstance(value, Mapping):
            raise TypeError(f"Prediction {index} must be a mapping")
        map_name = value.get("map_name", value.get("map"))
        sample_id = value.get("sample_id")
        if sample_id is None and map_name is not None and value.get("file_frame") is not None:
            sample_id = f"{map_name}/{value['file_frame']}"
        canonical = _canonical_sample_id(sample_id, str(map_name) if map_name else None)
        if canonical in result:
            raise ValueError(f"Duplicate prediction for {canonical}")
        result[canonical] = value
    return result


def _pose_from_prediction(value: Any, *, prediction_space: str, calibration: Mapping[str, Any]) -> list[float]:
    if isinstance(value, Mapping):
        fields = ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw")
        if all(field in value for field in fields):
            pose = [float(value[field]) for field in fields]
        elif "pred_pose" in value:
            pose = [float(number) for number in value["pred_pose"]]
        elif "prediction" in value:
            pose = [float(number) for number in value["prediction"]]
        elif "pose" in value:
            pose = [float(number) for number in value["pose"]]
        elif all(field in value for field in ("x", "y", "z", "pitch", "yaw")):
            pose = [float(value[field]) for field in ("x", "y", "z", "pitch", "yaw")]
            prediction_space = "physical"
        else:
            raise ValueError("Prediction needs pred_x/pred_y/pred_z/pred_pitch/pred_yaw or a five-value pose")
    else:
        pose = [float(number) for number in value]
    if len(pose) != 5 or not all(math.isfinite(number) for number in pose):
        raise ValueError(f"Prediction pose must contain five finite values: {pose!r}")
    if prediction_space == "normalized":
        return denormalize_pose(pose, calibration)
    if prediction_space == "physical":
        return pose
    raise ValueError("prediction_space must be 'normalized' or 'physical'")


def _gt_pose(record: Mapping[str, Any]) -> list[float]:
    if "pose" in record and "z_calibration" in record:
        return denormalize_pose(record["pose"], record["z_calibration"])
    raw = record.get("pose_raw")
    if isinstance(raw, Mapping):
        try:
            x, y, z = float(raw["x"]), float(raw["y"]), float(raw["z"])
            if "angle_v_rad" in raw:
                pitch = math.degrees(float(raw["angle_v_rad"]))
            elif "angle_v" in raw:
                pitch = math.degrees(float(raw["angle_v"]))
            else:
                pitch = float(raw["pitch"])
            if "angle_h_rad" in raw:
                yaw = math.degrees(float(raw["angle_h_rad"]))
            elif "angle_h" in raw:
                yaw = math.degrees(float(raw["angle_h"]))
            else:
                yaw = float(raw["yaw"])
            return [x, y, z, pitch, yaw]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid pose_raw in visualization record") from exc
    raise ValueError(f"Record {record.get('sample_id')!r} has no GT pose")


def _stable_map_seed(seed: int, map_name: str) -> int:
    digest = hashlib.sha256(f"csgo_seen10_visualize:{int(seed)}:{map_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def select_visualization_records(
    records: Any,
    *,
    seed: int = 0,
    samples_per_map: int = 10,
    maps: Sequence[str] | None = None,
) -> dict[str, list[Mapping[str, Any]]]:
    """Select a deterministic prefix-preserving sample set for each map."""

    if samples_per_map <= 0:
        raise ValueError("samples_per_map must be positive")
    rows = _records(records)
    selected_maps = tuple(SEEN10_MAPS if maps is None else maps)
    unknown = [map_name for map_name in selected_maps if map_name not in SEEN10_MAPS]
    if unknown:
        raise ValueError(f"Unknown Seen-10 maps: {unknown}")
    if len(set(selected_maps)) != len(selected_maps):
        raise ValueError("maps contains duplicates")
    grouped = {
        map_name: [row for row in rows if row.get("map_name", row.get("map")) == map_name]
        for map_name in selected_maps
    }
    result: dict[str, list[Mapping[str, Any]]] = {}
    for map_name in selected_maps:
        map_rows = grouped[map_name]
        if not map_rows:
            continue
        count = min(samples_per_map, len(map_rows))
        chooser = random.Random(_stable_map_seed(seed, map_name))
        indices = sorted(chooser.sample(range(len(map_rows)), count))
        result[map_name] = [map_rows[index] for index in indices]
    return result


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    ratio = min(width / image.width, height / image.height)
    new_size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def _fpv_panel(
    image_path: str | os.PathLike[str],
    gt_pose: Sequence[float],
    pred_pose: Sequence[float],
    color: tuple[int, int, int],
    *,
    width: int = 448,
    image_height: int = 448,
) -> Image.Image:
    with Image.open(image_path) as source:
        fpv = source.convert("RGB")
    fpv = _fit_image(fpv, width, image_height)
    header_height = 52
    panel = Image.new("RGB", (width, header_height + image_height), "white")
    draw = ImageDraw.Draw(panel)
    gt_text = "gt_xyzhw=" + "[" + ", ".join(f"{float(value):.1f}" for value in gt_pose) + "]"
    pred_text = "pred_xyzhw=" + "[" + ", ".join(f"{float(value):.1f}" for value in pred_pose) + "]"
    # Keep both long physical tuples inside the panel while retaining the
    # largest readable font for ordinary values.
    font_size = max(10, min(16, width // 20))
    while font_size > 8:
        candidate = _font(font_size)
        if all(draw.textbbox((0, 0), text, font=candidate)[2] <= width - 8 for text in (gt_text, pred_text)):
            break
        font_size -= 1
    font = _font(font_size)
    draw.ellipse((8, 12, 22, 26), fill=color, outline="black", width=1)
    # Pillow's ``anchor='ma'`` centers text at the requested x coordinate.
    draw.text((width / 2, 12), gt_text, fill="black", font=font, anchor="ma")
    draw.text((width / 2, 32), pred_text, fill="black", font=font, anchor="ma")
    panel.paste(fpv, (0, header_height))
    return panel


def _render_map(
    map_name: str,
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
    output_dir: Path,
    *,
    prediction_space: str,
    fpv_width: int,
    fpv_height: int,
    output_extension: str,
    overwrite: bool,
) -> tuple[Path, list[str]]:
    if not records:
        raise ValueError(f"No records selected for map {map_name}")
    radar_path = records[0].get("radar_path", records[0].get("map_path"))
    if not radar_path:
        raise ValueError(f"Record for {map_name} has no radar_path")
    with Image.open(radar_path) as source:
        radar = source.convert("RGBA")

    selected_ids: list[str] = []
    resolved: list[tuple[Mapping[str, Any], list[float], list[float]]] = []
    for record in records:
        sample_id = _canonical_sample_id(record.get("sample_id"), map_name)
        if sample_id not in predictions:
            raise ValueError(f"Missing prediction for selected sample {sample_id}")
        calibration = record.get("z_calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError(f"Record {sample_id} has no z_calibration")
        resolved.append(
            (
                record,
                _gt_pose(record),
                _pose_from_prediction(
                    predictions[sample_id],
                    prediction_space=prediction_space,
                    calibration=calibration,
                ),
            )
        )
        selected_ids.append(sample_id)

    panels = [
        _fpv_panel(
            record.get("image_path", record.get("fpv_path")),
            gt_pose,
            pred_pose,
            SAMPLE_COLORS[index % len(SAMPLE_COLORS)],
            width=fpv_width,
            image_height=fpv_height,
        )
        for index, (record, gt_pose, pred_pose) in enumerate(resolved)
    ]
    strip_width = max(panel.width for panel in panels)
    strip_height = sum(panel.height for panel in panels)
    strip = Image.new("RGB", (strip_width, strip_height), "white")
    y_offset = 0
    for panel in panels:
        strip.paste(panel, (0, y_offset))
        y_offset += panel.height

    # Match the radar height to the ten-panel strip, preserving its aspect
    # ratio.  Coordinates are still interpreted in the benchmark's 1024x1024
    # physical XY frame and scaled independently to the displayed radar size.
    radar = radar.resize(
        (max(1, round(radar.width * strip_height / radar.height)), strip_height),
        Image.Resampling.LANCZOS,
    )
    overlay = Image.new("RGBA", radar.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale_x, scale_y = radar.width / 1024.0, radar.height / 1024.0
    marker_radius = max(5, round(min(radar.width, radar.height) / 150.0))
    line_width = max(2, round(marker_radius / 2.5))

    for index, (_record, gt_pose, pred_pose) in enumerate(resolved):
        color = SAMPLE_COLORS[index % len(SAMPLE_COLORS)]

        def point(pose: Sequence[float]) -> tuple[float, float]:
            # Edge clipping is visual-only; labels retain the raw prediction.
            x = min(1024.0, max(0.0, float(pose[0]))) * scale_x
            y = min(1024.0, max(0.0, float(pose[1]))) * scale_y
            return x, y

        gt_x, gt_y = point(gt_pose)
        pred_x, pred_y = point(pred_pose)
        draw.line((gt_x, gt_y, pred_x, pred_y), fill=(*color, 220), width=line_width)
        draw.ellipse(
            (gt_x - marker_radius, gt_y - marker_radius, gt_x + marker_radius, gt_y + marker_radius),
            fill=(*color, 235),
            outline="black",
            width=1,
        )
        pred_radius = marker_radius + 3
        draw.ellipse(
            (pred_x - pred_radius, pred_y - pred_radius, pred_x + pred_radius, pred_y + pred_radius),
            fill=None,
            outline=(*color, 255),
            width=max(2, line_width),
        )

    composed = Image.new("RGB", (radar.width + strip.width, max(radar.height, strip.height)), "white")
    composed.paste(Image.alpha_composite(radar, overlay).convert("RGB"), (0, 0))
    composed.paste(strip, (radar.width, 0))
    suffix = output_extension.lstrip(".")
    path = output_dir / f"vis_map_{map_name}.{suffix}"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing visualization: {path}")
    save_kwargs = {"quality": 95} if suffix.lower() in {"jpg", "jpeg"} else {}
    composed.save(path, **save_kwargs)
    return path, selected_ids


def visualize_localization_predictions(
    records: Any,
    predictions: Any,
    output_dir: str | os.PathLike[str],
    *,
    seed: int = 0,
    samples_per_map: int = 10,
    maps: Sequence[str] | None = None,
    prediction_space: str = "normalized",
    fpv_width: int = 448,
    fpv_height: int = 448,
    output_extension: str = "png",
    overwrite: bool = False,
) -> dict[str, str]:
    """Render one deterministic radar+FPV sheet per selected Seen-10 map.

    ``predictions`` accepts the standard JSONL row sequence or a mapping from
    sample ID to either a standard row or a five-value pose.  Standard rows
    are interpreted in normalized Benchmark v2 space by default.  The return
    value maps each rendered map name to its output path.
    """

    if fpv_width <= 0 or fpv_height <= 0:
        raise ValueError("fpv_width and fpv_height must be positive")
    prediction_index = _prediction_index(predictions)
    selected = select_visualization_records(
        records,
        seed=seed,
        samples_per_map=samples_per_map,
        maps=maps,
    )
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for map_name in (SEEN10_MAPS if maps is None else tuple(maps)):
        if map_name not in selected:
            continue
        path, _ = _render_map(
            map_name,
            selected[map_name],
            prediction_index,
            target_dir,
            prediction_space=prediction_space,
            fpv_width=fpv_width,
            fpv_height=fpv_height,
            output_extension=output_extension,
            overwrite=overwrite,
        )
        output_paths[map_name] = str(path)
    return output_paths


# Readable aliases for integration scripts.
render_seen10_visualizations = visualize_localization_predictions
visualize_seen10 = visualize_localization_predictions


__all__ = [
    "SAMPLE_COLORS",
    "render_seen10_visualizations",
    "select_visualization_records",
    "visualize_localization_predictions",
    "visualize_seen10",
]
