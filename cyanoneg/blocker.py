"""Greyscale → colour-blocked RGB.

The ET-1810's driver offers no ink-density control, so UV density is controlled entirely by
what colour is printed. A greyscale negative on a 4-ink dye machine wastes the three best
UV blockers; the negative is therefore always rendered in a blocker hue.

Phase 1 model (fixed hue, from PLAN.md):

    blocker_eff = white + s * (blocker - white)      # saturation scales toward white
    RGB         = white - n * (white - blocker_eff)  # n = negative value in [0, 1]

``n = 0`` → pure white (clear film, full exposure); ``n = 1`` → the full blocker colour
(maximum UV density). ``s`` is the Harmon-style saturation scalar: one curve, saturation
sets the density range. The optimal hue and ``s`` come from measuring the HSB grid target —
they are never assumed.
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
