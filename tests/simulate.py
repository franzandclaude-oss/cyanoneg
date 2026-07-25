"""Simulate a flatbed scan of a finished cyanotype print of a generated target.

The simulator is deliberately unkind: it scales (scan ppi ≠ target ppi), rotates or
perspective-warps, blurs, adds noise, and surrounds the print with scanner-bed border —
because that is what real scans look like. If analyze.py survives this, fiducial
detection and the homography are earning their keep.
"""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

from cyanoneg.analyze import homography, lightness
from cyanoneg.imageio import Image, from_linear
from cyanoneg.targets import Target

PAPER_LSTAR = 93.0
BLACK_LSTAR = 22.0


def _lstar_to_y(lstar: np.ndarray) -> np.ndarray:
    """Inverse CIE L* — exact inverse of analyze.lightness."""
    f = (np.asarray(lstar, dtype=np.float64) + 16.0) / 116.0
    return np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29))


def render_print(target: Target, response=None, *, cell_response=None) -> np.ndarray:
    """Render the cyanotype print of a target in print orientation, as linear Y.

    ``response`` maps a wedge cell's normalised value in [0, 1] → normalised print
    **lightness** in [0, 1] (0 = max black, 1 = paper white), matching what
    ``analyze_wedge`` measures. For targets without a ``value`` key (the blocker grids),
    pass ``cell_response`` taking the whole cell dict instead.

    Everything that is not a labelled cell — border, margins, fiducials, printed labels —
    is rendered from the film's own ink coverage, so the polarity can never drift out of
    step with the generator: heavy ink blocks UV and prints light, clear film prints dark.
    """
    sidecar = target.sidecar
    film_print_view = target.film[:, ::-1]  # contact printing un-mirrors the film
    ink = 1.0 - film_print_view.min(axis=-1).astype(np.float64)
    lightness_norm = ink.copy()  # clear film (ink 0) → black; full ink → paper white

    if cell_response is None:
        cell_response = lambda cell: response(cell["value"] / (sidecar["levels"] - 1))  # noqa: E731
    for cell in sidecar["cells"] + sidecar.get("references", []):
        lightness_norm[
            cell["y_px"] : cell["y_px"] + cell["h_px"],
            cell["x_px"] : cell["x_px"] + cell["w_px"],
        ] = float(cell_response(cell))

    lstar = BLACK_LSTAR + lightness_norm * (PAPER_LSTAR - BLACK_LSTAR)
    return _lstar_to_y(lstar)


def _gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur via shifted sums (PIL's blur rejects float mode)."""
    radius = max(1, int(round(3 * sigma)))
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    taps /= taps.sum()
    for axis in (0, 1):
        out = np.zeros_like(a)
        for offset, weight in zip(range(-radius, radius + 1), taps):
            out += weight * np.roll(a, offset, axis=axis)
        a = out
    return a


def scan_of(
    print_y: np.ndarray,
    *,
    scale: float = 0.8,
    rotate_deg: float = 1.5,
    perspective: float = 0.004,
    noise: float = 0.004,
    blur_px: float = 1.0,
    margin_px: int = 120,
    orientation: str = "as-is",
    seed: int = 5,
) -> Image:
    """Turn a rendered print into a plausible scanner output image (sRGB, mono)."""
    rng = np.random.default_rng(seed)
    h, w = print_y.shape
    sw, sh = int(w * scale), int(h * scale)

    # Content corners in the output scan, jittered for perspective + rotated.
    theta = np.deg2rad(rotate_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    base = np.array([[0, 0], [sw, 0], [0, sh], [sw, sh]], dtype=np.float64)
    centre = np.array([sw / 2, sh / 2])
    jitter = rng.uniform(-perspective, perspective, (4, 2)) * np.array([sw, sh])
    dst_corners = (base - centre) @ rot.T + centre + jitter + margin_px

    out_w, out_h = sw + 2 * margin_px, sh + 2 * margin_px
    src_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)
    coeffs = homography(dst_corners, src_corners).flatten()[:8]

    pil = PILImage.fromarray(print_y.astype(np.float32), mode="F")
    warped = pil.transform(
        (out_w, out_h),
        PILImage.Transform.PERSPECTIVE,
        tuple(coeffs),
        resample=PILImage.Resampling.BILINEAR,
        fillcolor=float(_lstar_to_y(np.array([PAPER_LSTAR - 3]))[0]),  # scanner lid, off-white
    )
    y = np.asarray(warped, dtype=np.float64)
    if blur_px:
        y = _gaussian_blur(y, blur_px)
    y = np.clip(y + rng.normal(0, noise, y.shape), 0.0, 1.0)

    encoded = from_linear(y.astype(np.float32), "srgb")
    if orientation == "rot90":
        encoded = np.rot90(encoded, 1)
    elif orientation == "rot180":
        encoded = np.rot90(encoded, 2)
    elif orientation == "rot270":
        encoded = np.rot90(encoded, 3)
    elif orientation == "mirror":
        encoded = encoded[:, ::-1]
    elif orientation != "as-is":
        raise ValueError(orientation)
    return Image(np.ascontiguousarray(encoded), "srgb", ppi=300)
