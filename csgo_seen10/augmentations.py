"""Label-preserving color augmentation for Seen-10 FPV and radar images.

The four OFT color parameter ranges are retained.  Pixel values are bounded
to RGB [0, 1] after each operation; this pixel clamp is unrelated to the
OFT action-target Q99 clip.  No crop, flip, rotation or erasing is performed.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
from PIL import Image


PHOTOMETRIC_POLICY = "oft_photometric_only"


def _rng(*, seed: int, epoch: int, sample_id: str, view: str) -> random.Random:
    if not isinstance(seed, int) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Augmentation seed and nonnegative epoch must be integers")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Augmentation requires a nonempty sample_id")
    if view not in ("fpv", "radar"):
        raise ValueError("Augmentation view must be 'fpv' or 'radar'")
    key = f"csgo-seen10-photo-v1\0{seed}\0{epoch}\0{sample_id}\0{view}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _adjust_hsv(rgb: np.ndarray, *, saturation: float, hue_shift: float) -> np.ndarray:
    """Adjust saturation and hue in floating-point HSV, preserving geometry."""

    high = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    delta = high - low
    sat = np.divide(delta, high, out=np.zeros_like(high), where=high > 0)
    hue = np.zeros_like(high)
    colored = delta > 0
    red = colored & (rgb[..., 0] == high)
    green = colored & ~red & (rgb[..., 1] == high)
    blue = colored & ~red & ~green
    hue[red] = ((rgb[..., 1] - rgb[..., 2])[red] / delta[red]) % 6
    hue[green] = (rgb[..., 2] - rgb[..., 0])[green] / delta[green] + 2
    hue[blue] = (rgb[..., 0] - rgb[..., 1])[blue] / delta[blue] + 4
    hue = (hue / 6 + hue_shift) % 1.0
    sat = np.clip(sat * saturation, 0.0, 1.0)

    scaled = hue * 6
    sector = np.floor(scaled).astype(np.uint8) % 6
    fraction = scaled - np.floor(scaled)
    p = high * (1 - sat)
    q = high * (1 - fraction * sat)
    t = high * (1 - (1 - fraction) * sat)
    out = np.empty_like(rgb)
    choices = (
        (high, t, p),
        (q, high, p),
        (p, high, t),
        (p, q, high),
        (t, p, high),
        (high, p, q),
    )
    for index, channels in enumerate(choices):
        mask = sector == index
        for channel in range(3):
            out[..., channel][mask] = channels[channel][mask]
    return out


def augment_image(
    image: Image.Image,
    *,
    policy: str = "none",
    seed: int = 42,
    epoch: int = 0,
    sample_id: str,
    view: str,
) -> Image.Image:
    """Return one RGB image for both visual branches of the OFT processor.

    The random stream is a hash of seed, epoch, sample ID and image role.  FPV
    and radar are deliberately sampled independently.  This is stable across
    data-loader workers and GPU counts and does not mutate global RNG state.

    Brightness is an additive delta in [−0.2, 0.2], as in TF image
    ``random_brightness``.  Contrast and saturation use factors in [0.8,
    1.2]; hue uses a turn fraction in [−0.05, 0.05].  The operation order is
    brightness, contrast, saturation, hue.  Validation and inference use
    ``policy='none'``.
    """

    if policy not in ("none", PHOTOMETRIC_POLICY):
        raise ValueError(f"Unsupported image augmentation policy {policy!r}")
    if not isinstance(image, Image.Image):
        raise TypeError("augment_image expects a PIL image")
    rgb_image = image.convert("RGB")
    if policy == "none":
        return rgb_image
    rng = _rng(seed=seed, epoch=epoch, sample_id=sample_id, view=view)
    brightness = rng.uniform(-0.2, 0.2)
    contrast = rng.uniform(0.8, 1.2)
    saturation = rng.uniform(0.8, 1.2)
    hue = rng.uniform(-0.05, 0.05)

    pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
    pixels = np.clip(pixels + brightness, 0.0, 1.0)
    mean = pixels.mean(axis=(0, 1), keepdims=True)
    pixels = np.clip((pixels - mean) * contrast + mean, 0.0, 1.0)
    pixels = _adjust_hsv(pixels, saturation=saturation, hue_shift=hue)
    pixels = np.clip(pixels, 0.0, 1.0)
    return Image.fromarray(np.rint(pixels * 255.0).astype(np.uint8), mode="RGB")


__all__ = ["PHOTOMETRIC_POLICY", "augment_image"]
