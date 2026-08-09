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

import functools
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

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
    #: Name of the embedded profile the pixels were converted *from*, if any. Recorded so a
    #: conversion is always something the tool can say it did, never something it did quietly.
    converted_from: str | None = None

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
            converted_from=self.converted_from,
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


def icc_description(icc: bytes | None) -> str | None:
    """The human-readable name inside an ICC profile, or None if it cannot be read."""
    if not icc:
        return None
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(bytes(icc)))
        return (ImageCms.getProfileDescription(profile) or "").strip() or None
    except Exception:  # noqa: BLE001 - a malformed profile simply has no name
        return None


def space_from_icc(icc: bytes | None) -> Space | None:
    """Identify one of *our* working spaces from an embedded ICC profile, by name.

    This is the fast path for files already in a space the pipeline speaks: it means no
    conversion happens and the pixels pass through untouched. Anything else returns ``None``
    and goes to :func:`icc_to_srgb`, which converts rather than guesses.
    """
    desc = (icc_description(icc) or "").lower()
    if not desc:
        return None
    if "srgb" in desc:
        return "srgb"
    if "gamma 2.2" in desc or "gamma2.2" in desc:
        return "gamma22"
    if "linear" in desc:
        return "linear"
    return None


# --- matrix-shaper conversion ------------------------------------------------------------
#
# Old scans arrive tagged Adobe RGB, ProPhoto, Gray Gamma, whatever the scanner software of
# the day wrote. Refusing them all and making Steven re-export an archive would be the wrong
# trade: the profile is a *declaration*, not a guess, so reading it breaks no rule of this
# module. What it must never do is fail quietly, hence :attr:`Image.converted_from`.
#
# Pillow's ImageCms cannot help here — PIL has no 16-bit RGB mode, so it cannot transform a
# 16-bit scan at all. The parsing below is ~60 lines and covers every matrix-shaper profile,
# which is every RGB display/working profile in practice. LUT-based profiles (scanner input
# profiles with A2B0 tables) are not covered and fall through to the explicit-space error.

_S15F16 = 65536.0  # s15Fixed16Number scale


def _icc_tags(icc: bytes) -> dict[str, bytes]:
    count = struct.unpack_from(">I", icc, 128)[0]
    out: dict[str, bytes] = {}
    for i in range(count):
        sig, offset, size = struct.unpack_from(">4sII", icc, 132 + 12 * i)
        out[sig.decode("ascii", "replace")] = icc[offset : offset + size]
    return out


def _parse_xyz(tag: bytes) -> np.ndarray:
    return np.array(struct.unpack_from(">3i", tag, 8), dtype=np.float64) / _S15F16


def _parse_curve(tag: bytes) -> Callable[[np.ndarray], np.ndarray]:
    """Build the forward transfer function (encoded value → linear light) for one channel."""
    kind = tag[:4].decode("ascii", "replace")
    if kind == "curv":
        n = struct.unpack_from(">I", tag, 8)[0]
        if n == 0:  # identity
            return lambda a: a
        if n == 1:  # single u8Fixed8 gamma
            g = struct.unpack_from(">H", tag, 12)[0] / 256.0
            return lambda a, g=g: np.power(a, g)
        table = np.frombuffer(tag, dtype=">u2", count=n, offset=12).astype(np.float64) / 65535.0
        grid = np.linspace(0.0, 1.0, n)
        return lambda a, table=table, grid=grid: np.interp(a, grid, table)
    if kind == "para":
        ftype = struct.unpack_from(">H", tag, 8)[0]
        counts = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}
        if ftype not in counts:
            raise ValueError(f"unsupported parametricCurveType {ftype}")
        p = np.array(struct.unpack_from(f">{counts[ftype]}i", tag, 12), dtype=np.float64) / _S15F16

        def apply(a: np.ndarray, ftype: int = ftype, p: np.ndarray = p) -> np.ndarray:
            if ftype == 0:
                return np.power(np.maximum(a, 0.0), p[0])
            g, aa, b = p[0], p[1], p[2]
            c = p[3] if ftype >= 2 else 0.0
            # Types 0–2 split at -b/a; types 3–4 carry an explicit breakpoint d.
            d = p[4] if ftype >= 3 else (-b / aa if aa else 0.0)
            hi = np.power(np.maximum(aa * a + b, 0.0), g)
            if ftype == 2:
                return np.where(a >= d, hi + c, c)
            if ftype == 3:
                return np.where(a >= d, hi, c * a)
            if ftype == 4:
                return np.where(a >= d, hi + p[5], c * a + p[6])
            return np.where(a >= d, hi, 0.0)

        return apply
    raise ValueError(f"unsupported curve tag type {kind!r}")


@dataclass(frozen=True)
class _MatrixShaper:
    """A profile reduced to what a conversion needs: per-channel TRCs and RGB→XYZ."""

    curves: tuple[Callable[[np.ndarray], np.ndarray], ...]
    matrix: np.ndarray  # 3x3, linear RGB → XYZ relative to the D50 PCS

    def to_xyz(self, data: np.ndarray) -> np.ndarray:
        linear = np.stack([c(data[..., i]) for i, c in enumerate(self.curves)], axis=-1)
        return linear @ self.matrix.T


def _matrix_shaper(icc: bytes) -> _MatrixShaper | None:
    """Reduce an ICC profile to a matrix shaper, or None if it is not one.

    Grey profiles are included: a single kTRC with no colorants is treated as a neutral
    axis, which is what "Gray Gamma 2.2" and friends actually mean.
    """
    try:
        tags = _icc_tags(icc)
        if {"rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC"} <= tags.keys():
            curves = tuple(_parse_curve(tags[t]) for t in ("rTRC", "gTRC", "bTRC"))
            matrix = np.stack([_parse_xyz(tags[t]) for t in ("rXYZ", "gXYZ", "bXYZ")], axis=1)
            return _MatrixShaper(curves, matrix)
        if "kTRC" in tags and "wtpt" in tags:
            grey = _parse_curve(tags["kTRC"])
            white = _parse_xyz(tags["wtpt"])
            # Grey is fed in on all three channels, so averaging them back gives
            # XYZ = grey(v) x white — the neutral axis, which is all a grey profile has.
            return _MatrixShaper((grey, grey, grey), np.outer(white, np.full(3, 1 / 3)))
    except Exception:  # noqa: BLE001 - an unparseable profile is simply not usable here
        return None
    return None


@functools.lru_cache(maxsize=1)
def _srgb_shaper() -> _MatrixShaper:
    """sRGB as a matrix shaper, read from the same ICC data we embed on save."""
    shaper = _matrix_shaper(srgb_icc())
    if shaper is None:  # pragma: no cover - would mean Pillow's own sRGB profile is broken
        raise RuntimeError("could not parse Pillow's sRGB profile")
    return shaper


def icc_to_srgb(data: np.ndarray, icc: bytes | None) -> np.ndarray | None:
    """Convert float [0,1] pixel data from an embedded profile into sRGB.

    Returns ``None`` when the profile cannot be reduced to a matrix shaper, so the caller
    can fall back to demanding an explicit space rather than inventing one.

    Colours outside sRGB are clipped — relative colorimetric with no gamut mapping. For a
    negative destined for a process with a density range near 1.1 that is immaterial, and
    it is what every "Convert to Profile" with clipping does anyway.
    """
    src = _matrix_shaper(bytes(icc)) if icc else None
    if src is None:
        return None
    mono = data.ndim == 2
    rgb = np.stack([data] * 3, axis=-1) if mono else data
    if rgb.shape[-1] != 3:
        return None

    xyz = src.to_xyz(rgb.astype(np.float64))
    linear = xyz @ np.linalg.inv(_srgb_shaper().matrix).T
    out = from_linear(np.clip(linear, 0.0, 1.0).astype(np.float32), "srgb")
    return out.mean(axis=-1).astype(np.float32) if mono else out


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

    Three outcomes, in order:

    1. ``space`` given — taken as an override and applied to the numbers as they are, with
       no conversion. This is the escape hatch for files whose tag is absent or wrong.
    2. The embedded profile names one of our working spaces — used directly, pixels
       untouched.
    3. The embedded profile is something else (Adobe RGB, ProPhoto, Gray Gamma…) — the
       pixels are *converted* into sRGB and :attr:`Image.converted_from` records the
       profile they came from.

    A file with no profile and no ``space`` raises :class:`ColourSpaceError`. Guessing is
    what produces negatives that are subtly, unfixably wrong; reading a profile the file
    itself declares is not guessing.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        raw, icc, ppi = _load_tiff(path)
    else:
        raw, icc, ppi = _load_other(path)

    data, bit_depth = _normalise(raw)
    if data.ndim == 3 and data.shape[2] == 4:  # drop alpha; negatives have no use for it
        data = data[..., :3]

    if space is not None:
        return Image(data=data, space=_check_space(space), ppi=ppi, bit_depth=bit_depth)

    resolved = space_from_icc(icc)
    if resolved is not None:
        return Image(data=data, space=resolved, ppi=ppi, bit_depth=bit_depth)

    converted = icc_to_srgb(data, icc)
    if converted is not None:
        return Image(
            data=converted,
            space="srgb",
            ppi=ppi,
            bit_depth=bit_depth,
            converted_from=icc_description(icc) or "an unnamed ICC profile",
        )

    what = f"has an ICC profile ({icc_description(icc)}) this tool cannot convert" if icc \
        else "has no ICC profile"
    raise ColourSpaceError(
        f"{path.name} {what}, and no colour space was given. "
        f"Pass space= explicitly (one of {SPACES}) — do not let the tool assume."
    )


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
