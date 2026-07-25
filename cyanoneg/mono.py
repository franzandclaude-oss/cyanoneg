"""RGB → monochrome conversion and per-channel noise reporting.

35mm colour negative scans carry very different grain per channel — blue is usually worst.
The mixer therefore takes arbitrary weights so a noisy channel can be down-weighted or
dropped entirely, and :func:`channel_noise` measures which channel that should be rather
than assuming.
"""

from __future__ import annotations

import numpy as np

from .imageio import Image, from_linear, to_linear

#: Rec.601 luma weights — the familiar Photoshop-style 30/59/11 default from PLAN.md.
DEFAULT_WEIGHTS: tuple[float, float, float] = (0.30, 0.59, 0.11)


def to_mono(
    image: Image,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    *,
    mix_in: str = "linear",
) -> Image:
    """Mix RGB down to a single channel, returning a mono image in the same space.

    ``mix_in='linear'`` (default) decodes to linear light, mixes, and re-encodes — the
    physically meaningful weighted sum. ``mix_in='encoded'`` mixes the encoded values
    directly, reproducing what Photoshop's channel mixer does; kept for comparison against
    an existing Photoshop-based workflow.

    Weights are normalised to sum to 1, so (1, 1, 0) means "average red and green, drop
    blue" without changing overall brightness.
    """
    if image.is_mono:
        return image
    w = np.asarray(weights, dtype=np.float32)
    if w.sum() <= 0:
        raise ValueError(f"channel weights must sum to a positive value, got {weights}")
    w = w / w.sum()

    if mix_in == "linear":
        lin = to_linear(image.data, image.space)
        mono = from_linear(lin @ w, image.space)
    elif mix_in == "encoded":
        mono = (image.data @ w).astype(np.float32)
    else:
        raise ValueError(f"mix_in must be 'linear' or 'encoded', got {mix_in!r}")
    return image.replace(np.clip(mono, 0.0, 1.0))


def channel_noise(image: Image) -> dict[str, float]:
    """Estimate per-channel noise so the grainiest channel can be identified.

    High-pass residual (pixel minus 3×3 box mean) summarised by MAD, scaled to sigma
    (×1.4826) — robust to image content and outliers. Values are comparable *between
    channels of the same image*, not across images.
    """
    if image.is_mono:
        raise ValueError("channel_noise needs an RGB image")
    out: dict[str, float] = {}
    for i, name in enumerate(("red", "green", "blue")):
        ch = image.data[..., i]
        # 3x3 box mean via cumulative sums would be overkill; shifted-add is clear and fast.
        acc = np.zeros_like(ch)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += np.roll(np.roll(ch, dy, axis=0), dx, axis=1)
        residual = ch - acc / 9.0
        # Trim the wrapped border introduced by roll.
        residual = residual[1:-1, 1:-1]
        mad = float(np.median(np.abs(residual - np.median(residual))))
        out[name] = mad * 1.4826
    return out


def suggest_weights(noise: dict[str, float], threshold: float = 2.0) -> tuple[float, float, float]:
    """Suggest mixer weights from a noise report.

    Starts from :data:`DEFAULT_WEIGHTS` and zeroes any channel whose noise exceeds
    ``threshold`` × the quietest channel — the "drop the grainy blue channel" move from
    PLAN.md, made quantitative. Never drops all channels: if every channel is noisy the
    default weights come back unchanged.
    """
    base = dict(zip(("red", "green", "blue"), DEFAULT_WEIGHTS))
    quiet = min(noise.values())
    if quiet <= 0:
        return DEFAULT_WEIGHTS
    kept = {ch: w for ch, w in base.items() if noise[ch] <= threshold * quiet}
    if not kept:
        return DEFAULT_WEIGHTS
    return tuple(base[ch] if ch in kept else 0.0 for ch in ("red", "green", "blue"))  # type: ignore[return-value]
