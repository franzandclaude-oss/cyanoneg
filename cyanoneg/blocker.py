"""Greyscale → colour-blocked RGB.

The ET-1810's driver offers no ink-density control, so UV density is controlled entirely by
what colour is printed. A greyscale negative on a 4-ink dye machine wastes the three best
UV blockers; the negative is therefore always rendered in a blocker hue.

**The polarity convention — get this wrong and prints come out inverted.**

``v`` is the *negative's greyscale pixel value*, exactly as ``step_invert`` produces it
(``v = 1 - positive``):

    v = 0  black in the negative  → maximum blocker ink → UV blocked → paper stays WHITE
    v = 1  white in the negative  → no ink, clear film  → UV passes  → paper goes DARK

Since ``v = 1 - p``, a **bright** positive gives ``v = 0`` and therefore heavy ink and a
white print — which is what a real digital negative looks like: dense where the scene was
bright. An inkjet lays more ink for darker pixels, and this mapping is the explicit
equivalent of that.

**Fixed-hue model** (Phase 1, from PLAN.md):

    blocker_eff = white + s * (blocker - white)   # saturation scales toward white
    RGB         = blocker_eff + v * (white - blocker_eff)

``s`` is the Harmon-style saturation scalar: one curve, saturation sets the density range.

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
    """Recolour a mono negative into the blocker hue.

    ``negative`` is the greyscale negative as produced by ``step_invert``: **0 is black
    (maximum ink), 1 is white (clear film)** — see the module docstring. Blacks become the
    blocker colour; whites stay clear.
    """
    v = np.asarray(negative, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError("apply_blocker expects a 2D mono negative")
    b_eff = effective_blocker(rgb, saturation)
    out = b_eff[None, None, :] + v[..., None] * (1.0 - b_eff[None, None, :])
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
    """Recolour a mono negative through zone control points.

    ``negative`` uses the same convention as :func:`apply_blocker` (0 = black = maximum
    ink). Zone control points are keyed by *ink density* n, so the lookup is by ``1 - v``.
    """
    v = np.asarray(negative, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError("apply_zone_blocker expects a 2D mono negative")
    table = zone_curves(zones, size)
    idx = np.clip(1.0 - v, 0.0, 1.0) * (size - 1)
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


def recover_coverage(view: np.ndarray, blocker: dict) -> np.ndarray:
    """Recover per-pixel ink coverage (0 = clear film, 1 = full measured ink) from a
    colour-blocked negative, by inverting the profile's own blocker model.

    ``view`` is an ``(..., 3)`` array of blocked RGB in ``[0, 1]`` (the same convention
    :func:`apply_blocker` and :func:`apply_zone_blocker` produce).

    This exists because ``1 - min(R, G, B)`` — the obvious shortcut — is only correct for a
    blocker that is fully saturated *and* reaches zero in some channel. Below saturation
    1.0 the minimum channel never reaches zero, so that shortcut silently understates
    coverage: it degrades smoothly rather than failing loudly, which is exactly the kind of
    error this codebase otherwise refuses to let through. Inverting the model itself keeps
    the answer correct at any saturation, and for the zone-varying model too.
    """
    model = blocker.get("model")
    if model == "zone_hue":
        zones = blocker.get("zones")
        if not zones:
            raise ValueError("a zone_hue blocker needs 'zones' to recover coverage")
        table = zone_curves(zones)
    else:
        rgb, saturation = blocker.get("rgb"), blocker.get("saturation")
        if rgb is None or saturation is None:
            raise ValueError("a fixed_hue blocker needs measured 'rgb' and 'saturation' to recover coverage")
        table = zone_curves(fixed_hue_as_zones(rgb, saturation))

    # Invert whichever channel the blocker actually moves the most. That is both the most
    # informative single channel to read and the only one guaranteed away from a near-zero
    # range — the failure mode `min(R, G, B)` does not defend against.
    channel_range = table.max(axis=0) - table.min(axis=0)
    channel = int(np.argmax(channel_range))
    if channel_range[channel] < 1e-6:
        # An unsaturated (or placeholder-white) blocker moves nothing measurable in any
        # channel; there is no ink to recover because the model itself lays down none.
        return np.zeros(view.shape[:-1], dtype=np.float32)

    density = np.linspace(0.0, 1.0, table.shape[0])
    order = np.argsort(table[:, channel])
    observed = np.clip(view[..., channel], 0.0, 1.0)
    return np.interp(observed, table[order, channel], density[order]).astype(np.float32)


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
