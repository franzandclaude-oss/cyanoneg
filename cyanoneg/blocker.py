"""Greyscale → colour-blocked RGB.

The ET-1810's driver offers no ink-density control, so UV density is controlled entirely by
what colour is printed. A greyscale negative on a 4-ink dye machine wastes the three best
UV blockers; the negative is therefore always rendered in a blocker hue.

**Fixed-hue model** (Phase 1, from PLAN.md):

    blocker_eff = white + s * (blocker - white)      # saturation scales toward white
    RGB         = white - n * (white - blocker_eff)  # n = negative value in [0, 1]

``n = 0`` → pure white (clear film, full exposure); ``n = 1`` → the full blocker colour
(maximum UV density). ``s`` is the Harmon-style saturation scalar: one curve, saturation
sets the density range.

**Zone-varying model** (Phase 3, EDN-style): one hue rarely blocks best across the whole
scale — the ink mix that holds back UV most effectively at full density is not necessarily
the one that does so in the highlights, where far less ink is laid down. This model stores
control points (density n → RGB) and interpolates between them, giving a 1D → 3D transform
that is exported as a 3D LUT for Photoshop QA.

Both models are *measured*, never assumed: the fixed hue comes from the HSB grid, the zone
control points from the zone grid. A profile carrying zone data it did not measure is a
profile lying about its provenance.
"""

from __future__ import annotations

import colorsys

import numpy as np

#: A neutral placeholder until the blocker grid is measured: pure red, the hue with dye
#: precedent. Deliberately conspicuous — a profile still using it is uncalibrated.
PLACEHOLDER_RGB: tuple[int, int, int] = (255, 0, 0)


def effective_blocker(rgb: tuple[int, int, int], saturation: float = 1.0) -> np.ndarray:
    """The blocker colour after saturation scaling, as float [0, 1] RGB."""
    if not 0.0 <= saturation <= 1.0:
        raise ValueError(f"saturation must be within [0, 1], got {saturation}")
    b = np.asarray(rgb, dtype=np.float32) / 255.0
    if b.shape != (3,) or b.min() < 0 or b.max() > 1:
        raise ValueError(f"blocker rgb must be three 0-255 values, got {rgb}")
    return 1.0 + saturation * (b - 1.0)


def apply_blocker(
    negative: np.ndarray,
    rgb: tuple[int, int, int],
    saturation: float = 1.0,
) -> np.ndarray:
    """Map a mono *negative* (0 = clear, 1 = densest) to colour-blocked RGB.

    Input must already be inverted — this is pipeline step 6, after step 5's inversion.
    """
    n = np.asarray(negative, dtype=np.float32)
    if n.ndim != 2:
        raise ValueError("apply_blocker expects a 2D mono negative")
    b_eff = effective_blocker(rgb, saturation)
    out = 1.0 - n[..., None] * (1.0 - b_eff[None, None, :])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def hue_to_rgb(hue_degrees: float, sat: float = 1.0, brightness: float = 1.0) -> tuple[int, int, int]:
    """HSB → 0-255 RGB, for building the blocker-grid target's hue sweep."""
    r, g, b = colorsys.hsv_to_rgb((hue_degrees % 360.0) / 360.0, sat, brightness)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


# --------------------------------------------------------------------------- zone model


def zone_curves(zones: list[dict], size: int = 256) -> np.ndarray:
    """Expand zone control points into a (size, 3) float table indexed by density.

    ``zones`` is a list of ``{"n": density in [0, 1], "rgb": [r, g, b]}`` control points.
    A point at ``n = 0`` is implied to be white (clear film) unless given. Interpolation
    is linear per channel — the control points come from measurements, and inventing
    curvature between them would be fabricating data.
    """
    if not zones:
        raise ValueError("zone blocker needs at least one control point")
    points = sorted(zones, key=lambda z: float(z["n"]))
    ns = [float(z["n"]) for z in points]
    if any(not 0.0 <= n <= 1.0 for n in ns):
        raise ValueError("zone control points must have n within [0, 1]")
    if len(set(ns)) != len(ns):
        raise ValueError("zone control points must have distinct n values")

    rgbs = [np.asarray(z["rgb"], dtype=np.float64) / 255.0 for z in points]
    for rgb in rgbs:
        if rgb.shape != (3,) or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("zone rgb values must be three 0-255 numbers")

    if ns[0] > 0.0:  # clear film at zero density
        ns.insert(0, 0.0)
        rgbs.insert(0, np.ones(3))
    if ns[-1] < 1.0:  # hold the densest measured colour to the top
        ns.append(1.0)
        rgbs.append(rgbs[-1])

    grid = np.linspace(0.0, 1.0, size)
    stack = np.stack(rgbs)  # (points, 3)
    return np.stack([np.interp(grid, ns, stack[:, c]) for c in range(3)], axis=-1).astype(np.float32)


def apply_zone_blocker(negative: np.ndarray, zones: list[dict], size: int = 256) -> np.ndarray:
    """Map a mono negative through zone control points to colour-blocked RGB."""
    n = np.asarray(negative, dtype=np.float32)
    if n.ndim != 2:
        raise ValueError("apply_zone_blocker expects a 2D mono negative")
    table = zone_curves(zones, size)
    idx = np.clip(n, 0.0, 1.0) * (size - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.minimum(lo + 1, size - 1)
    frac = (idx - lo)[..., None]
    return (table[lo] * (1.0 - frac) + table[hi] * frac).astype(np.float32)


def fixed_hue_as_zones(rgb: tuple[int, int, int], saturation: float = 1.0) -> list[dict]:
    """Express a fixed-hue blocker as zone control points.

    Useful for exporting any profile as a 3D LUT, and for tests: the zone model with
    these two points must reproduce the fixed-hue model exactly.
    """
    b_eff = effective_blocker(rgb, saturation)
    return [
        {"n": 0.0, "rgb": [255, 255, 255]},
        {"n": 1.0, "rgb": [float(v * 255.0) for v in b_eff]},
    ]


def export_blocker_cube(path, zones: list[dict], size: int = 33, title: str = "cyanoneg blocker"):
    """Write the blocker transform as a 3D .cube (Photoshop: Color Lookup).

    The blocker is a 1D → 3D map, so the cube applies it to each input's luminance-free
    grey level: entries are indexed by RGB but only the diagonal is meaningful, which is
    exactly how a greyscale negative enters it. This is a QA/inspection artefact — the
    pipeline always applies the table directly at full precision.
    """
    from pathlib import Path

    path = Path(path)
    table = zone_curves(zones, 256)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0", ""]
    grid = np.linspace(0.0, 1.0, size)
    # .cube ordering: red varies fastest, then green, then blue.
    for b in grid:
        for g in grid:
            for r in grid:
                level = (r + g + b) / 3.0
                out = table[int(round(level * 255))]
                lines.append(f"{out[0]:.6f} {out[1]:.6f} {out[2]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path
