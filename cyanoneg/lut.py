"""1D LUT construction, application, smoothing, and export.

This is the mathematically critical module: :func:`derive_correction` is the whole point of
the tool. Given patch measurements of how the print process responds to input values, it
builds the curve that, applied to the positive *before* inversion, makes the printed result
linear. The synthetic round-trip test in ``tests/`` proves this inversion works before any
film, paper or chemistry is spent.

PCHIP is implemented directly (Fritsch–Carlson) rather than pulling in scipy: it is ~40
lines, and a monotone interpolant that silently changed behaviour with a scipy upgrade
would be exactly the kind of quiet calibration drift this project exists to eliminate.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_SIZE = 256


# --------------------------------------------------------------------------- PCHIP core


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson monotone slopes at each knot."""
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    d = np.zeros(n)

    # Interior knots: weighted harmonic mean where deltas share a sign, else 0.
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    # Endpoints: one-sided three-point estimate, clamped to preserve monotonicity.
    def _end(h0: float, h1: float, d0: float, d1: float) -> float:
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if s * d0 <= 0:
            return 0.0
        if abs(s) > 3 * abs(d0):
            return 3 * d0
        return s

    if n == 2:
        d[0] = d[-1] = delta[0]
    else:
        d[0] = _end(h[0], h[1], delta[0], delta[1])
        d[-1] = _end(h[-1], h[-2], delta[-1], delta[-2])
    return d


def pchip_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate the monotone cubic through (x, y) at query points ``xq``.

    ``x`` must be strictly increasing. Queries outside [x0, xn] are clamped — a calibration
    curve has no business extrapolating.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        raise ValueError("pchip needs at least two knots")
    if np.any(np.diff(x) <= 0):
        raise ValueError("pchip knots must be strictly increasing")
    d = _pchip_slopes(x, y)
    xq = np.clip(np.asarray(xq, dtype=np.float64), x[0], x[-1])
    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)

    h = x[idx + 1] - x[idx]
    t = (xq - x[idx]) / h
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)
    return h00 * y[idx] + h10 * h * d[idx] + h01 * y[idx + 1] + h11 * h * d[idx + 1]


# --------------------------------------------------------------------------- Lut


@dataclass
class Lut:
    """A 1D lookup table over [0, 1].

    ``values[i]`` is the output for input ``i / (len(values) - 1)``. Applied by linear
    interpolation, so a 256-entry table is smooth on 16-bit data.
    """

    values: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.0, DEFAULT_SIZE))

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim != 1 or len(self.values) < 2:
            raise ValueError("a Lut is a 1D table of at least two values")

    @classmethod
    def identity(cls, size: int = DEFAULT_SIZE) -> Lut:
        return cls(np.linspace(0.0, 1.0, size))

    @property
    def size(self) -> int:
        return len(self.values)

    def is_identity(self, tol: float = 1e-9) -> bool:
        return bool(np.allclose(self.values, np.linspace(0.0, 1.0, self.size), atol=tol))

    def apply(self, a: np.ndarray) -> np.ndarray:
        """Apply to an array in [0, 1] by linear interpolation."""
        grid = np.linspace(0.0, 1.0, self.size)
        out = np.interp(np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0), grid, self.values)
        return out.astype(np.float32)

    def enforce_monotonic(self) -> Lut:
        """Return a non-decreasing copy (running maximum, clipped to [0, 1])."""
        return Lut(np.clip(np.maximum.accumulate(self.values), 0.0, 1.0))

    def smooth_pchip(self, knots: int = 21) -> Lut:
        """Re-fit through ``knots`` evenly spaced samples with a monotone cubic.

        Averages a small neighbourhood around each knot first, so isolated measurement
        spikes are suppressed rather than interpolated through.
        """
        grid = np.linspace(0.0, 1.0, self.size)
        kx = np.linspace(0.0, 1.0, knots)
        half = max(1, self.size // (knots * 2))
        ky = np.empty(knots)
        for i, k in enumerate(kx):
            centre = int(round(k * (self.size - 1)))
            lo, hi = max(0, centre - half), min(self.size, centre + half + 1)
            ky[i] = self.values[lo:hi].mean()
        # Pin the endpoints: paper white and max black are measured references, not noise.
        ky[0], ky[-1] = self.values[0], self.values[-1]
        return Lut(pchip_eval(kx, ky, grid)).enforce_monotonic()

    # ----------------------------------------------------------------- export

    def export_cube(self, path: str | Path, title: str = "cyanoneg") -> Path:
        """Write a 1D .cube file (loads in Photoshop via Color Lookup)."""
        path = Path(path)
        lines = [f'TITLE "{title}"', f"LUT_1D_SIZE {self.size}", ""]
        lines += [f"{v:.6f} {v:.6f} {v:.6f}" for v in np.clip(self.values, 0.0, 1.0)]
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path

    def export_acv(self, path: str | Path, points: int = 16) -> Path:
        """Write a Photoshop .acv curves file.

        The format (Photoshop File Formats spec): big-endian, version word 4, curve count,
        then per curve a point count followed by (output, input) pairs on a 0–255 scale.
        A curve holds at most 19 points, so the table is subsampled; Photoshop interpolates
        between them, which is why this is a QA/inspection format — the pipeline itself
        always applies the full-resolution table.

        One curve is written (the composite RGB curve), matching how a correction curve is
        used in the Photoshop workflow this tool replaces.
        """
        if not 2 <= points <= 19:
            raise ValueError("a Photoshop curve holds between 2 and 19 points")
        path = Path(path)
        grid = np.linspace(0.0, 1.0, points)
        outputs = self.apply(grid)

        payload = struct.pack(">hh", 4, 1)  # version 4, one curve
        payload += struct.pack(">h", points)
        for x, y in zip(grid, outputs):
            payload += struct.pack(">hh", int(round(float(y) * 255)), int(round(float(x) * 255)))
        path.write_bytes(payload)
        return path


# --------------------------------------------------------------------------- calibration


def derive_correction(
    measured_in: np.ndarray,
    measured_out: np.ndarray,
    size: int = DEFAULT_SIZE,
    knots: int = 21,
) -> Lut:
    """Build the correction curve from patch measurements.

    ``measured_in`` are the values sent to the printer (the patch values of the step
    tablet, in the working space). ``measured_out`` are the corresponding normalised print
    responses — 0 for the response at input 0, 1 for the response at input 1, as produced
    by normalising scan readings against the max-black and paper-white references.

    The process maps in → out; the correction is the *inverse*: for each desired output
    level we look up which input produces it. Duplicate readings (flat shadow/highlight
    regions where the process has clipped) are averaged, the axis swap is interpolated
    with PCHIP, and the result is monotone-smoothed.

    Applying this LUT before printing makes (correction ∘ process) ≈ identity — which is
    precisely what the synthetic round-trip test asserts.
    """
    mi = np.asarray(measured_in, dtype=np.float64).ravel()
    mo = np.asarray(measured_out, dtype=np.float64).ravel()
    if mi.shape != mo.shape or len(mi) < 3:
        raise ValueError("need matching in/out measurement arrays of at least 3 patches")

    # Sort by *output* — after the swap, output is the independent axis.
    order = np.argsort(mo)
    mo, mi = mo[order], mi[order]

    # Collapse duplicate outputs (process clipping) by averaging their inputs.
    ux, inverse = np.unique(np.round(mo, 9), return_inverse=True)
    uy = np.zeros(len(ux))
    for i in range(len(ux)):
        uy[i] = mi[inverse == i].mean()

    if len(ux) < 2:
        raise ValueError("measurements are constant — the process response carries no information")

    # Ensure full [0, 1] coverage so the correction is defined everywhere.
    if ux[0] > 0.0:
        ux = np.concatenate(([0.0], ux))
        uy = np.concatenate(([uy[0]], uy))
    if ux[-1] < 1.0:
        ux = np.concatenate((ux, [1.0]))
        uy = np.concatenate((uy, [uy[-1]]))

    grid = np.linspace(0.0, 1.0, size)
    raw = Lut(pchip_eval(ux, np.clip(uy, 0.0, 1.0), grid)).enforce_monotonic()
    return raw.smooth_pchip(knots=knots)
