"""Orchestration: positive scan → print-ready negative.

The order below is mandated by PLAN.md (per Reeder/Anderson and Ware) and asserted by
test — the correction curve applies to the **positive, before inversion**, and the mirror
flip comes last. Getting either wrong silently produces wrong tones or an unprintable
negative, which is exactly the failure mode this tool exists to remove.

    1. load positive          (16-bit TIFF preferred, float32 internally)
    2. mono conversion        (channel mixer)
    3. apply paper LUT        (in the profile's working_space, on the positive)
    4. resize                 (to print size at output ppi, resampled in linear light)
    5. invert
    6. colour block
    7. flip horizontal        (ink side against emulsion)
    8. export                 (16-bit TIFF)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from . import imageio as cio
from .blocker import apply_blocker
from .imageio import Image
from .mono import DEFAULT_WEIGHTS, to_mono
from .profiles import Profile

DEFAULT_OUTPUT_PPI = 360  # Epson's native input resolution


@dataclass(frozen=True)
class PrintSize:
    """Target print dimensions in millimetres. The image is fitted within, never cropped."""

    width_mm: float
    height_mm: float

    def pixels(self, ppi: float) -> tuple[int, int]:
        return (
            int(round(self.width_mm / 25.4 * ppi)),
            int(round(self.height_mm / 25.4 * ppi)),
        )


# --------------------------------------------------------------------------- steps
# Each step is a named function so the ordering test can patch and record them.


def step_mono(image: Image, weights: tuple[float, float, float]) -> Image:
    return to_mono(image, weights)


def step_apply_lut(image: Image, profile: Profile) -> Image:
    """Apply the paper curve to the *positive*, in the profile's working space."""
    data = image.data
    if image.space != profile.working_space:
        data = cio.convert_space(data, image.space, profile.working_space)
    return Image(
        data=profile.lut.apply(data),
        space=profile.working_space,
        ppi=image.ppi,
        bit_depth=image.bit_depth,
    )


def step_resize(image: Image, size: PrintSize, ppi: float) -> Image:
    """Fit within the print size, resampling in linear light to avoid halo artefacts."""
    box_w, box_h = size.pixels(ppi)
    src_w, src_h = image.size
    scale = min(box_w / src_w, box_h / src_h)
    dst = (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))))

    linear = cio.to_linear(image.data, image.space)
    pil = PILImage.fromarray(linear.astype(np.float32), mode="F")
    resized = np.asarray(pil.resize(dst, PILImage.Resampling.LANCZOS), dtype=np.float32)
    encoded = cio.from_linear(np.clip(resized, 0.0, 1.0), image.space)
    return Image(data=encoded, space=image.space, ppi=ppi, bit_depth=image.bit_depth)


def step_invert(image: Image) -> Image:
    return image.replace(1.0 - image.data)


def step_blocker(image: Image, profile: Profile) -> Image:
    rgb = profile.blocker.get("rgb")
    sat = profile.blocker.get("saturation")
    if rgb is None or sat is None:
        raise ValueError(
            f"profile {profile.name!r} has no blocker colour yet — "
            "print and measure the HSB blocker grid first"
        )
    return image.replace(apply_blocker(image.data, tuple(rgb), float(sat)))


def step_flip(image: Image) -> Image:
    return image.replace(np.ascontiguousarray(image.data[:, ::-1]))


# --------------------------------------------------------------------------- pipeline


def make_negative(
    source: str | Path | Image,
    profile: Profile,
    print_size: PrintSize,
    *,
    output_path: str | Path | None = None,
    output_ppi: float = DEFAULT_OUTPUT_PPI,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    space: cio.Space | None = None,
    raw_scan: bool = False,
) -> Image:
    """Run the full positive → negative pipeline.

    ``raw_scan=True`` is the flagged path for un-inverted lab scans: the source is a
    negative, so it is inverted to a positive immediately after loading, and the rest of
    the pipeline is unchanged.

    Returns the final negative; writes it to ``output_path`` when given.
    """
    image = source if isinstance(source, Image) else cio.load_image(source, space=space)

    if raw_scan:
        image = step_invert(image)  # raw scan → positive; the tonal pipeline needs a positive

    image = step_mono(image, weights)  # 2
    image = step_apply_lut(image, profile)  # 3 — on the positive, before inversion
    image = step_resize(image, print_size, output_ppi)  # 4
    image = step_invert(image)  # 5
    image = step_blocker(image, profile)  # 6
    image = step_flip(image)  # 7

    if output_path is not None:
        cio.save_tiff(output_path, image)  # 8
    return image
