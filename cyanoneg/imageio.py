"""Image loading, saving, and transfer-curve conversion.

The whole tool hinges on one rule: **the colour space is always explicit**. The supplied
calibration chart is sRGB, the LUT is measured in whatever space the printed step tablet was
measured in, and applying a curve in the wrong space is a silent tonal error that looks
plausible on screen. So nothing here ever guesses — an untagged file without an explicit
space raises rather than assuming.

sRGB is the default working space throughout, matching ``EDN_RGB_256.tif``. Note that sRGB is
*not* a pure 2.2 power curve: it has a linear segment below 0.04045, which matters in exactly
the shadow region where cyanotype's long toe already causes trouble.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
from PIL import Image as PILImage
from PIL import ImageCms

Space = Literal["srgb", "gamma22", "linear"]
SPACES: tuple[Space, ...] = ("srgb", "gamma22", "linear")

DEFAULT_SPACE: Space = "srgb"

# sRGB IEC61966-2.1 breakpoints.
_SRGB_LINEAR_CUT = 0.04045
_SRGB_ENCODED_CUT = 0.0031308
_SRGB_SLOPE = 12.92
_SRGB_A = 0.055
_SRGB_GAMMA = 2.4

_ICC_TAG = 34675  # TIFF InterColorProfile


class ColourSpaceError(ValueError):
    """Raised when a file's colour space cannot be established and none was supplied."""


@dataclass(frozen=True)
class Image:
    """An image in a declared colour space.

    ``data`` is float32 in [0, 1] — either (H, W) monochrome or (H, W, 3) RGB.
    ``bit_depth`` records what the file held, purely so saving can round-trip faithfully.
    """

    data: np.ndarray
    space: Space
    ppi: float | None = None
    bit_depth: int = 16

    @property
    def is_mono(self) -> bool:
        return self.data.ndim == 2

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return self.data.shape[1], self.data.shape[0]

    def replace(self, data: np.ndarray, space: Space | None = None) -> Image:
        return Image(
            data=data,
            space=self.space if space is None else space,
            ppi=self.ppi,
            bit_depth=self.bit_depth,
        )


# --------------------------------------------------------------------------- transfer curves


def _check_space(space: str) -> Space:
    if space not in SPACES:
        raise ValueError(f"unknown colour space {space!r}; expected one of {SPACES}")
    return space  # type: ignore[return-value]


def to_linear(a: np.ndarray, space: Space) -> np.ndarray:
    """Decode from ``space`` to linear light."""
    _check_space(space)
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    if space == "linear":
        return a
    if space == "gamma22":
        return np.power(a, 2.2, dtype=np.float32)
    # Evaluate both branches on clipped input so the power never sees a negative base.
    lo = a / _SRGB_SLOPE
    hi = np.power((a + _SRGB_A) / (1.0 + _SRGB_A), _SRGB_GAMMA, dtype=np.float32)
    return np.where(a <= _SRGB_LINEAR_CUT, lo, hi).astype(np.float32)


def from_linear(a: np.ndarray, space: Space) -> np.ndarray:
    """Encode linear light into ``space``."""
    _check_space(space)
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    if space == "linear":
        return a
    if space == "gamma22":
        return np.power(a, 1.0 / 2.2, dtype=np.float32)
    lo = a * _SRGB_SLOPE
    hi = (1.0 + _SRGB_A) * np.power(a, 1.0 / _SRGB_GAMMA, dtype=np.float32) - _SRGB_A
    return np.where(a <= _SRGB_ENCODED_CUT, lo, hi).astype(np.float32)


def convert_space(a: np.ndarray, src: Space, dst: Space) -> np.ndarray:
    """Re-encode ``a`` from ``src`` to ``dst`` via linear light."""
    if _check_space(src) == _check_space(dst):
        return np.asarray(a, dtype=np.float32)
    return from_linear(to_linear(a, src), dst)


# --------------------------------------------------------------------------- ICC handling


def srgb_icc() -> bytes:
    """Bytes of an sRGB ICC profile, for embedding on save."""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def space_from_icc(icc: bytes | None) -> Space | None:
    """Best-effort identification of a colour space from an embedded ICC profile.

    Returns ``None`` when the profile is absent or unrecognised — callers must then be told
    the space explicitly rather than falling back to a guess.
    """
    if not icc:
        return None
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = (ImageCms.getProfileDescription(profile) or "").strip().lower()
    except Exception:  # noqa: BLE001 - a malformed profile is simply unidentifiable
        return None
    if "srgb" in desc:
        return "srgb"
    if "gamma 2.2" in desc or "gamma2.2" in desc:
        return "gamma22"
    if "linear" in desc:
        return "linear"
    return None


# --------------------------------------------------------------------------- load / save


def _normalise(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Scale integer pixel data to float32 [0, 1], reporting the original bit depth."""
    if raw.dtype == np.uint8:
        return (raw.astype(np.float32) / 255.0, 8)
    if raw.dtype == np.uint16:
        return (raw.astype(np.float32) / 65535.0, 16)
    if raw.dtype in (np.float32, np.float64):
        return (np.clip(raw.astype(np.float32), 0.0, 1.0), 32)
    raise ValueError(f"unsupported pixel type {raw.dtype}")


def _load_tiff(path: Path) -> tuple[np.ndarray, bytes | None, float | None]:
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        raw = page.asarray()
        icc = page.tags[_ICC_TAG].value if _ICC_TAG in page.tags else None
        ppi = None
        if "XResolution" in page.tags:
            num, den = page.tags["XResolution"].value
            unit = page.tags["ResolutionUnit"].value if "ResolutionUnit" in page.tags else 2
            if den and int(unit) == 2:  # 2 == inches
                ppi = num / den
            elif den and int(unit) == 3:  # 3 == centimetres
                ppi = (num / den) * 2.54
    return raw, icc, ppi


def _load_other(path: Path) -> tuple[np.ndarray, bytes | None, float | None]:
    with PILImage.open(path) as im:
        icc = im.info.get("icc_profile")
        dpi = im.info.get("dpi")
        ppi = float(dpi[0]) if dpi else None
        raw = np.array(im)
    return raw, icc, ppi


def load_image(path: str | Path, space: Space | None = None) -> Image:
    """Load an image as float32 [0, 1] in a known colour space.

    ``space`` overrides whatever the file claims. If the file carries no identifiable ICC
    profile and no ``space`` is given, this raises :class:`ColourSpaceError` — guessing here
    is what produces negatives that are subtly, unfixably wrong.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        raw, icc, ppi = _load_tiff(path)
    else:
        raw, icc, ppi = _load_other(path)

    resolved = _check_space(space) if space is not None else space_from_icc(icc)
    if resolved is None:
        raise ColourSpaceError(
            f"{path.name} has no identifiable ICC profile and no colour space was given. "
            f"Pass space= explicitly (one of {SPACES}) — do not let the tool assume."
        )

    data, bit_depth = _normalise(raw)
    if data.ndim == 3 and data.shape[2] == 4:  # drop alpha; negatives have no use for it
        data = data[..., :3]
    return Image(data=data, space=resolved, ppi=ppi, bit_depth=bit_depth)


def save_tiff(
    path: str | Path,
    image: Image | np.ndarray,
    *,
    ppi: float | None = None,
    space: Space | None = None,
) -> Path:
    """Write 16-bit TIFF with resolution tagged and, for sRGB, an ICC profile embedded.

    ``gamma22`` and ``linear`` are written untagged: no standard profile is generated for
    them here, so the space lives in the paper profile rather than in the file. sRGB is the
    default working space precisely so this stays an edge case.
    """
    path = Path(path)
    if isinstance(image, Image):
        data, out_ppi, out_space = image.data, ppi or image.ppi, space or image.space
    else:
        data, out_ppi, out_space = np.asarray(image), ppi, space
        if out_space is None:
            raise ColourSpaceError("space= is required when saving a bare array")
    _check_space(out_space)

    quantised = np.rint(np.clip(data, 0.0, 1.0) * 65535.0).astype(np.uint16)

    kwargs: dict = {}
    if out_ppi:
        kwargs["resolution"] = (float(out_ppi), float(out_ppi))
        kwargs["resolutionunit"] = "INCH"
    if out_space == "srgb":
        icc = srgb_icc()
        kwargs["extratags"] = [(_ICC_TAG, 7, len(icc), icc, True)]

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, quantised, photometric="rgb" if data.ndim == 3 else "minisblack", **kwargs)
    return path
