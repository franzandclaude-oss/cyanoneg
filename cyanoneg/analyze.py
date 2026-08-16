"""Read scanned calibration prints → measurements → profile data.

Chain (PLAN.md): detect fiducials → perspective-correct → locate patches geometrically →
sample each with ≥11×11 averaging → normalise against paper-white and max-black references
→ average redundant patches → derive the correction (``lut.derive_correction``) → report
density range and residual spikes.

Everything works on a flatbed scan of the **cyanotype print**, which shows the target in
print orientation (the contact print un-mirrors the film), matching sidecar coordinates
directly. A scan of the film itself, or a rotated scan, is recovered automatically: the
hollow fiducial plus the frame's aspect ratio pin down the orientation uniquely among all
eight rotation/mirror combinations.

Implementation is numpy + PIL only — no scipy/OpenCV. The homography is a direct 8×8 DLT
solve; connected components are a BFS flood fill over the (small) set of dark pixels.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .imageio import Image, load_image, to_linear
from .lut import Lut, derive_correction

#: Reflection density range worth having for classic cyanotype; outside this the process
#: (exposure, coating, chemistry) needs attention before the curve can help.
DR_WINDOW = (1.2, 1.4)

#: Copies of the same level differing by more than this (normalised response) are spikes.
#: How far one copy of a level may sit from its siblings' median before it is called an
#: outlier, in normalised print lightness.
#:
#: Deliberately *not* a max-minus-min test on the whole group. Range grows with sample
#: count for perfectly clean data, so a spread threshold tuned at 2 copies fires on almost
#: every level at 16 — which is exactly what happened when the wedge moved to k=16 and 31
#: of 32 levels flagged on a good print.
SPIKE_TOLERANCE = 0.04

#: Accepted blob size, as a fraction of the expected fiducial size.
#:
#: The upper bound has to sit below the patch-to-fiducial size ratio or the filter does not
#: separate them — in the default wedge that ratio is 78/56 = 1.39, so 1.8 (the first value
#: tried) admitted exactly the blobs it was added to exclude. 1.3 leaves room for a
#: fiducial's blob to grow under blur and thresholding while still rejecting a patch.
_FID_SIZE_BAND = (0.6, 1.3)

#: Robust scale multiplier for outlier rejection. 1.4826 * MAD estimates sigma for normal
#: data; rejecting beyond 4 sigma keeps coating variation (which is the signal's own noise)
#: while catching an ink spike or a coating defect on one patch.
SPIKE_SIGMAS = 4.0
_MAD_TO_SIGMA = 1.4826

_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


class AnalysisError(RuntimeError):
    """Raised when a scan cannot be read well enough to measure."""


# --------------------------------------------------------------------------- colour maths


def luminance(image: Image) -> np.ndarray:
    """Linear-light luminance Y in [0, 1]."""
    lin = to_linear(image.data, image.space)
    if lin.ndim == 2:
        return lin
    return (lin @ _REC709).astype(np.float32)


def lightness(y: np.ndarray) -> np.ndarray:
    """CIE L* from linear luminance (D65 white, Yn = 1) — perceptually uniform."""
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)
    f = np.where(y > (6 / 29) ** 3, np.cbrt(y), y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116.0 * f - 16.0


#: What a patch is measured *with*. ``lstar`` is neutral lightness, correct for a
#: monochrome cyanotype where the print has one colorant and luminance carries all of it.
#:
#: The per-channel entries exist for tricolour. A madder-toned magenta layer absorbs
#: green; measured in luminance its response is diluted by the other two channels, and
#: after the full M -> Y -> C cycle it would be read through colorants that had nothing to
#: do with it. ``lstar_g`` reads the layer through the channel it actually modulates.
#: ``tricolour.RESPONSE_QUANTITY`` maps each layer to its own.
RESPONSE_CHANNEL = {"lstar_r": 0, "lstar_g": 1, "lstar_b": 2}
QUANTITIES = ("lstar", *RESPONSE_CHANNEL)


def _response_plane(scan: Image, quantity: str) -> np.ndarray:
    """The linear-light plane a given quantity is measured from.

    Per-channel quantities take the channel's own linear value in place of luminance and
    push it through the same L* curve, so the two are on one scale and the normalisation
    and density-range maths downstream need no special case.
    """
    if quantity == "lstar":
        return luminance(scan)
    if quantity not in RESPONSE_CHANNEL:
        raise AnalysisError(
            f"unknown quantity {quantity!r}; expected one of {', '.join(QUANTITIES)}"
        )
    if scan.data.ndim != 3:
        raise AnalysisError(
            f"{quantity} needs a colour scan, but this one is single-channel — a mono "
            "scan cannot separate a layer from the colorants printed over it"
        )
    return to_linear(scan.data, scan.space)[..., RESPONSE_CHANNEL[quantity]].astype(np.float32)


# --------------------------------------------------------------------------- homography


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3×3 homography H with dst ~ H·src, from exactly four point pairs (DLT)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("homography needs exactly four (x, y) pairs on each side")
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    h = np.linalg.solve(np.asarray(rows), np.asarray(rhs))
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((len(points), 1))
    mapped = np.hstack([points, ones]) @ h.T
    return mapped[:, :2] / mapped[:, 2:3]


# --------------------------------------------------------------------------- blob finding


@dataclass
class Blob:
    ys: np.ndarray
    xs: np.ndarray

    @property
    def area(self) -> int:
        return len(self.ys)

    @property
    def centre(self) -> tuple[float, float]:
        """(x, y)"""
        return float(self.xs.mean()), float(self.ys.mean())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return int(self.ys.min()), int(self.xs.min()), int(self.ys.max()), int(self.xs.max())

    @property
    def squareness(self) -> float:
        y0, x0, y1, x1 = self.bbox
        h, w = y1 - y0 + 1, x1 - x0 + 1
        return min(h, w) / max(h, w)

    @property
    def fill(self) -> float:
        y0, x0, y1, x1 = self.bbox
        return self.area / ((y1 - y0 + 1) * (x1 - x0 + 1))


def _blob_size(blob: Blob) -> int:
    """Longer side of the blob's bounding box, in detection pixels."""
    y0, x0, y1, x1 = blob.bbox
    return max(y1 - y0, x1 - x0) + 1


def _expected_fiducial_size(scan, sidecar: dict, fid_px: int, scale: int, candidates) -> float | None:
    """How large a fiducial should appear, in detection pixels.

    Preferred route is the scan's own ppi against the sidecar's: the print is a known
    physical size, so a fiducial's size on any scan of it follows directly.

    Falling back when the scan carries no ppi, the fiducials are all identical by
    construction, so the largest cluster of similarly-sized candidates is almost certainly
    them — dark wedge patches vary in how much of each one clears the threshold, while the
    fiducials are solid clear film and clear it completely every time.
    """
    scan_ppi = getattr(scan, "ppi", None)
    sidecar_ppi = sidecar.get("ppi")
    if scan_ppi and sidecar_ppi:
        return fid_px * (scan_ppi / sidecar_ppi) / scale

    sizes = sorted(_blob_size(b) for b in candidates)
    if len(sizes) < 4:
        return None
    best_count, best_size = 0, None
    for s in sizes:  # widest window holding sizes within 25% of each other
        group = [t for t in sizes if s <= t <= s * 1.25]
        if len(group) > best_count:
            best_count, best_size = len(group), float(np.median(group))
    return best_size if best_count >= 4 else None


def _connected_components(mask: np.ndarray, min_area: int) -> list[Blob]:
    """4-connected components of a boolean mask via BFS — fine at detection resolution."""
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    blobs = []
    seeds = np.argwhere(mask)
    for sy, sx in seeds:
        if visited[sy, sx]:
            continue
        queue = deque([(int(sy), int(sx))])
        visited[sy, sx] = True
        ys, xs = [], []
        while queue:
            y, x = queue.popleft()
            ys.append(y)
            xs.append(x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if len(ys) >= min_area:
            blobs.append(Blob(np.asarray(ys), np.asarray(xs)))
    return blobs


# --------------------------------------------------------------------------- fiducials


@dataclass
class Frame:
    """Detected target frame: scan-space fiducial centres keyed by print-space corner."""

    corners: dict[str, tuple[float, float]]  # name -> (x, y) in scan pixels
    h_print_to_scan: np.ndarray = field(default_factory=lambda: np.eye(3))


_DETECT_WIDTH = 1200


def detect_fiducials(scan: Image, sidecar: dict) -> Frame:
    """Find the four fiducials in a scan and identify which print corner each one is.

    Fiducials print as the darkest square marks on the sheet (clear film → maximum
    exposure). Detection: downsample, threshold on the dark tail, take square-ish blobs,
    pick the one nearest each corner of the detected content, then use the hollow one and
    the frame's aspect ratio to resolve orientation and mirroring.
    """
    y_full = luminance(scan)
    scale = max(1, int(round(y_full.shape[1] / _DETECT_WIDTH)))
    h, w = y_full.shape[0] // scale, y_full.shape[1] // scale
    small = y_full[: h * scale, : w * scale].reshape(h, scale, w, scale).mean(axis=(1, 3))

    # Threshold between the dark tail and the midtone bulk.
    dark_ref = np.percentile(small, 2)
    mid_ref = np.percentile(small, 60)
    if mid_ref - dark_ref < 0.05:
        raise AnalysisError("scan has almost no tonal separation — is this the right file?")
    mask = small < dark_ref + 0.35 * (mid_ref - dark_ref)

    fid_px = sidecar["fiducials"]["top_left"]["size_px"]
    blobs = _connected_components(mask, min_area=9)
    candidates = [b for b in blobs if b.squareness > 0.55 and b.fill > 0.45]
    if len(candidates) < 4:
        raise AnalysisError(
            f"found only {len(candidates)} fiducial candidates — "
            "check the scan includes the full border and try a higher-resolution scan"
        )

    # Keep only blobs the size a fiducial should be.
    #
    # Shape alone does not distinguish a fiducial from a dark wedge patch — both are dark
    # squares. Which blobs pass the threshold depends on how deep the print's shadows came
    # out, so the same sheet scanned two ways yielded 60 candidates one time and 82 the
    # other, and in the second the corner picks landed on patches instead of fiducials.
    # Detection then failed with "no hollow fiducial found" while the fiducial sat plainly
    # in the scan.
    #
    # The expected size is known: the sidecar records it in print pixels, and the scan's
    # own ppi gives the conversion. Without a ppi tag, fall back to the largest cluster of
    # similarly-sized candidates, since the fiducials are all identical by construction.
    expected = _expected_fiducial_size(scan, sidecar, fid_px, scale, candidates)
    if expected is not None:
        sized = [b for b in candidates if _FID_SIZE_BAND[0] * expected <= _blob_size(b) <= _FID_SIZE_BAND[1] * expected]
        if len(sized) >= 4:
            candidates = sized

    xs = [b.centre[0] for b in candidates]
    ys = [b.centre[1] for b in candidates]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    corner_points = {
        "a": (x0, y0),  # scan's own top-left, before orientation is known
        "b": (x1, y0),
        "c": (x0, y1),
        "d": (x1, y1),
    }
    chosen: dict[str, Blob] = {}
    for key, (cx, cy) in corner_points.items():
        best = min(candidates, key=lambda b: (b.centre[0] - cx) ** 2 + (b.centre[1] - cy) ** 2)
        chosen[key] = best
    if len({id(b) for b in chosen.values()}) < 4:
        raise AnalysisError("fiducial candidates do not form four distinct corners")

    # Hollow test at full resolution: centre of the blob vs its own dark pixels.
    def is_hollow(blob: Blob) -> bool:
        cx, cy = blob.centre
        y0b, x0b, y1b, x1b = blob.bbox
        size = max(y1b - y0b, x1b - x0b) + 1
        r = max(1, int(size * scale) // 6)
        yc, xc = int(cy * scale + scale // 2), int(cx * scale + scale // 2)
        centre_patch = y_full[yc - r : yc + r + 1, xc - r : xc + r + 1]
        dark_level = float(np.median(y_full[blob.ys * scale, blob.xs * scale]))
        return float(np.median(centre_patch)) > dark_level + 0.1

    hollow_key = None
    for key, blob in chosen.items():
        if is_hollow(blob):
            if hollow_key is not None:
                raise AnalysisError("more than one hollow fiducial found — scan is ambiguous")
            hollow_key = key
    if hollow_key is None:
        raise AnalysisError("no hollow fiducial found — cannot establish orientation")

    # Orientation: the hollow fiducial is top-left in print space; the print frame is
    # landscape (width > height). Exactly one of the eight rotation/mirror assignments
    # satisfies both.
    scan_landscape = (x1 - x0) >= (y1 - y0)
    print_landscape = _sidecar_landscape(sidecar)
    assignments = {
        # hollow at scan corner -> mapping of print corners to scan corners, for the
        # same-aspect case (rotation by 0/180 or pure mirrors)...
        "a": {"top_left": "a", "top_right": "b", "bottom_left": "c", "bottom_right": "d"},
        "b": {"top_left": "b", "top_right": "a", "bottom_left": "d", "bottom_right": "c"},
        "c": {"top_left": "c", "top_right": "d", "bottom_left": "a", "bottom_right": "b"},
        "d": {"top_left": "d", "top_right": "c", "bottom_left": "b", "bottom_right": "a"},
    }
    rotated = {
        # ...and for the 90/270 case, where scan aspect is transposed.
        "a": {"top_left": "a", "top_right": "c", "bottom_left": "b", "bottom_right": "d"},
        "b": {"top_left": "b", "top_right": "d", "bottom_left": "a", "bottom_right": "c"},
        "c": {"top_left": "c", "top_right": "a", "bottom_left": "d", "bottom_right": "b"},
        "d": {"top_left": "d", "top_right": "b", "bottom_left": "c", "bottom_right": "a"},
    }
    mapping = (assignments if scan_landscape == print_landscape else rotated)[hollow_key]

    corners = {}
    for print_corner, key in mapping.items():
        cx, cy = chosen[key].centre
        # Refine at full resolution: centroid of dark pixels in a window around the blob.
        corners[print_corner] = _refine_centre(y_full, cx * scale, cy * scale, fid_px)

    src = np.array(
        [_fiducial_centre_print(sidecar, name) for name in ("top_left", "top_right", "bottom_left", "bottom_right")]
    )
    dst = np.array([corners[name] for name in ("top_left", "top_right", "bottom_left", "bottom_right")])
    return Frame(corners=corners, h_print_to_scan=homography(src, dst))


def _sidecar_landscape(sidecar: dict) -> bool:
    tl = sidecar["fiducials"]["top_left"]
    br = sidecar["fiducials"]["bottom_right"]
    return (br["x_px"] - tl["x_px"]) >= (br["y_px"] - tl["y_px"])


def _fiducial_centre_print(sidecar: dict, name: str) -> tuple[float, float]:
    f = sidecar["fiducials"][name]
    return (f["x_px"] + f["size_px"] / 2.0, f["y_px"] + f["size_px"] / 2.0)


def _refine_centre(y_full: np.ndarray, cx: float, cy: float, fid_px: int) -> tuple[float, float]:
    """Centroid of dark pixels in a window around a coarse fiducial centre."""
    r = max(8, fid_px)
    y0, y1 = max(0, int(cy) - r), min(y_full.shape[0], int(cy) + r)
    x0, x1 = max(0, int(cx) - r), min(y_full.shape[1], int(cx) + r)
    window = y_full[y0:y1, x0:x1]
    threshold = window.min() + 0.3 * (np.median(window) - window.min() + 1e-6)
    ys, xs = np.nonzero(window < threshold)
    if len(ys) == 0:
        return (cx, cy)
    return (x0 + float(xs.mean()), y0 + float(ys.mean()))


# --------------------------------------------------------------------------- sampling


def sample_cells(scan: Image, sidecar: dict, frame: Frame, quantity: str = "lstar") -> list[dict]:
    """Median-sample every sidecar cell through the detected homography.

    Returns one record per cell: the sidecar cell plus ``lstar``, ``y_linear``, ``rgb``
    and ``spread`` (inter-quartile range of the sample window, for quality flags).

    ``rgb`` is the patch's own linear-light colour. The tonal analysis has no use for it —
    it works in luminance throughout — but the soft proof does: predicting what a print
    looks like means predicting its colour, and Prussian blue desaturates towards paper
    along a curve no two-endpoint blend reproduces. Measuring it here costs nothing and
    means the proof never has to model it.

    ``quantity`` selects what ``response`` and ``response_linear`` hold. ``lstar`` and
    ``y_linear`` are always the neutral pair regardless, so a per-channel reading never
    costs the luminance one — useful when comparing a toned layer against the stain floor
    the control region gives. At the default the two pairs are the same numbers, which is
    what keeps the mono path byte-identical.
    """
    y_full = luminance(scan)
    response_full = _response_plane(scan, quantity)
    rgb_full = to_linear(scan.data, scan.space) if scan.data.ndim == 3 else None
    cells = sidecar["cells"]
    centres_print = np.array(
        [(c["x_px"] + c["w_px"] / 2.0, c["y_px"] + c["h_px"] / 2.0) for c in cells]
    )
    centres_scan = apply_homography(frame.h_print_to_scan, centres_print)

    # Window size: 40% of the patch pitch as mapped into scan pixels, at least 11.
    tl = np.array([_fiducial_centre_print(sidecar, "top_left")])
    tr = np.array([_fiducial_centre_print(sidecar, "top_right")])
    span_print = np.linalg.norm(tr - tl)
    span_scan = np.linalg.norm(
        apply_homography(frame.h_print_to_scan, tr) - apply_homography(frame.h_print_to_scan, tl)
    )
    scan_per_print = span_scan / span_print
    results = []
    for cell, (sx, sy) in zip(cells, centres_scan):
        half = max(5, int(cell["w_px"] * scan_per_print * 0.4 / 2))
        yc, xc = int(round(sy)), int(round(sx))
        if not (half <= yc < y_full.shape[0] - half and half <= xc < y_full.shape[1] - half):
            raise AnalysisError(
                f"cell at print ({cell['x_px']}, {cell['y_px']}) maps outside the scan — "
                "fiducial detection likely failed; rescan with the whole border visible"
            )
        window = y_full[yc - half : yc + half + 1, xc - half : xc + half + 1]
        y_med = float(np.median(window))
        q1, q3 = np.percentile(window, (25, 75))
        record = dict(cell)
        record["y_linear"] = y_med
        record["lstar"] = float(lightness(np.array([y_med]))[0])
        record["spread"] = float(q3 - q1)
        if quantity == "lstar":
            record["response_linear"], record["response"] = y_med, record["lstar"]
        else:
            r_med = float(
                np.median(
                    response_full[yc - half : yc + half + 1, xc - half : xc + half + 1]
                )
            )
            record["response_linear"] = r_med
            record["response"] = float(lightness(np.array([r_med]))[0])
        record["quantity"] = quantity
        if rgb_full is not None:
            patch = rgb_full[yc - half : yc + half + 1, xc - half : xc + half + 1]
            record["rgb"] = [float(v) for v in np.median(patch.reshape(-1, 3), axis=0)]
        results.append(record)
    return results


# --------------------------------------------------------------------------- wedge analysis


@dataclass
class WedgeAnalysis:
    lut: Lut
    measured_in: np.ndarray  # one value in [0, 1] per wedge level
    measured_out: np.ndarray  # normalised response, 0 = paper white, 1 = max black
    density_range: float
    spikes: list[dict]
    paper_lstar: float
    black_lstar: float
    warnings: list[str]
    raw_patches: list[dict]
    #: Which quantity every number above was measured with. Recorded because a curve read
    #: through green is not comparable to one read in luminance, and nothing else says so.
    quantity: str = "lstar"

    def summary(self) -> str:
        lines = [
            f"measured in: {self.quantity}",
            f"density range (reflection, est.): {self.density_range:.2f}",
            f"paper white: {self.paper_lstar:.1f}   max black: {self.black_lstar:.1f}",
            f"spikes flagged: {len(self.spikes)}",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def analyze_wedge(
    scan_path: str | Path,
    sidecar_path: str | Path,
    *,
    space=None,
    quantity: str = "lstar",
) -> WedgeAnalysis:
    """Full wedge chain: scan + sidecar → correction LUT + quality report.

    ``quantity`` picks what the curve is built from. The default reads neutral lightness,
    which is right for a single-colorant cyanotype. A tricolour layer passes its own
    channel (``tricolour.RESPONSE_QUANTITY``), because a toned layer measured in luminance
    has its response diluted by two colorants it did not lay.
    """
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    if "seed" not in sidecar:
        raise AnalysisError(f"{sidecar_path} is not a step-wedge sidecar")
    scan = load_image(scan_path, space=space)
    frame = detect_fiducials(scan, sidecar)
    samples = sample_cells(scan, sidecar, frame, quantity=quantity)

    # Group copies by level.
    by_value: dict[int, list[dict]] = {}
    for s in samples:
        by_value.setdefault(s["value"], []).append(s)
    levels = sorted(by_value)
    if levels != list(range(sidecar["levels"])):
        raise AnalysisError("sidecar does not cover every level exactly — file mismatch?")

    # Polarity (see blocker.py): the top patch value is black in the negative, so it lays
    # maximum ink, blocks UV and leaves paper white. Value 0 is clear film → max black.
    top = sidecar["levels"] - 1
    paper_lstar = float(np.mean([s["response"] for s in by_value[top]]))
    black_lstar = float(np.mean([s["response"] for s in by_value[0]]))
    if paper_lstar - black_lstar < 5.0:
        raise AnalysisError(
            "paper-white and max-black references are nearly identical — "
            "wrong scan, or the print failed"
        )

    def normalise(lstar: float) -> float:
        """Normalised print lightness: 0 at max black, 1 at paper white."""
        return (lstar - black_lstar) / (paper_lstar - black_lstar)

    warnings: list[str] = []
    spikes: list[dict] = []
    measured_in = np.array([v / (sidecar["levels"] - 1) for v in levels])
    measured_out = np.empty(len(levels))
    for i, v in enumerate(levels):
        copies = np.array([normalise(s["response"]) for s in by_value[v]])
        keep = np.ones(len(copies), dtype=bool)

        if len(copies) > 2:
            # Robust rejection: a copy is an outlier against its own siblings, not against
            # a fixed spread. Enough copies here to identify *which* one is bad and drop
            # it, which is the whole point of the redundancy.
            median = float(np.median(copies))
            mad = float(np.median(np.abs(copies - median)))
            scale = max(mad * _MAD_TO_SIGMA, SPIKE_TOLERANCE / 4.0)  # floor: MAD can be 0
            keep = np.abs(copies - median) <= SPIKE_SIGMAS * scale
            if not keep.any():  # pathological; keep everything rather than lose the level
                keep = np.ones(len(copies), dtype=bool)
        elif len(copies) == 2 and abs(copies[0] - copies[1]) > SPIKE_TOLERANCE:
            # Two copies cannot say which one is wrong, so flag and keep both.
            keep = np.ones(2, dtype=bool)
            spikes.append(
                {
                    "value": v,
                    "responses": [round(float(c), 4) for c in copies],
                    "positions": [(s["row"], s["col"]) for s in by_value[v]],
                    "rejected": 0,
                }
            )

        if len(copies) > 2 and not keep.all():
            spikes.append(
                {
                    "value": v,
                    "responses": [round(float(c), 4) for c in copies[~keep]],
                    "positions": [
                        (s["row"], s["col"]) for s, k in zip(by_value[v], keep) if not k
                    ],
                    "rejected": int((~keep).sum()),
                }
            )

        measured_out[i] = float(copies[keep].mean())

    # From the same quantity the curve was built from, or the reported range would
    # describe a different measurement than the LUT beside it.
    y_paper = float(np.mean([s["response_linear"] for s in by_value[top]]))
    y_black = float(np.mean([s["response_linear"] for s in by_value[0]]))
    density_range = float(np.log10(max(y_paper, 1e-6) / max(y_black, 1e-6)))

    # DR_WINDOW is a Prussian-blue number, measured in luminance on a classic cyanotype.
    # A bleached layer is not that print and is not read in that quantity: yellow is a
    # pale Fe(OH)3 ochre measured through blue, and its range has no reason to land
    # between 1.2 and 1.4. Judging it against this window would fire on every yellow
    # analysis, and a warning that is always wrong is worse than none — it teaches the
    # reader to skip the warnings that are right.
    #
    # Not silently dropped, though: the range is still computed and still reported, it
    # just isn't graded against a window that does not describe it.
    if quantity != "lstar":
        warnings.append(
            f"density range {density_range:.2f} measured in {quantity}; the "
            f"{DR_WINDOW[0]}–{DR_WINDOW[1]} window describes luminance on an untoned "
            "Prussian-blue print, so it is not applied here — no comparable window has "
            "been measured for this layer yet"
        )
    elif density_range < DR_WINDOW[0]:
        warnings.append(
            f"density range {density_range:.2f} is below the {DR_WINDOW[0]}–{DR_WINDOW[1]} window — "
            "under-exposure, weak chemistry, or the blocker passes too much UV"
        )
    elif density_range > DR_WINDOW[1] + 0.1:
        warnings.append(
            f"density range {density_range:.2f} is above the {DR_WINDOW[0]}–{DR_WINDOW[1]} window — "
            "unusual for classic cyanotype; check the scan"
        )
    if len(spikes) > 12:
        rejected = sum(s.get("rejected", 0) for s in spikes)
        warnings.append(
            f"{len(spikes)} levels with outlier patches ({rejected} copies dropped from the "
            "averages) — inspect the print for coating/ink defects"
        )

    lut = derive_correction(measured_in, measured_out)
    raw_patches = [
        {"value": s["value"], "row": s["row"], "col": s["col"], "lstar": round(s["lstar"], 3),
         "y_linear": round(s["y_linear"], 6), "spread": round(s["spread"], 6),
         "response": round(s["response"], 3), "response_linear": round(s["response_linear"], 6),
         **({"rgb": [round(v, 6) for v in s["rgb"]]} if "rgb" in s else {})}
        for s in samples
    ]
    return WedgeAnalysis(
        lut=lut,
        measured_in=measured_in,
        measured_out=measured_out,
        density_range=density_range,
        spikes=spikes,
        paper_lstar=paper_lstar,
        black_lstar=black_lstar,
        warnings=warnings,
        raw_patches=raw_patches,
        quantity=quantity,
    )


# --------------------------------------------------------------------------- grid analysis


@dataclass
class GridAnalysis:
    ranking: list[dict]  # cells sorted best-first by lstar
    references: list[dict]
    best: dict
    recommended_saturation: float | None
    warnings: list[str]

    def summary(self) -> str:
        top = self.ranking[:5]
        lines = ["best UV blockers (higher L* = whiter = better):"]
        lines += [
            f"  hue {c['hue_deg']:>4g}°  sat {c['saturation']:.2f}  L* {c['lstar']:.1f}" for c in top
        ]
        if self.recommended_saturation is not None:
            lines.append(
                f"recommendation: hue {self.best['hue_deg']:g}°, "
                f"saturation {self.recommended_saturation:.2f}"
            )
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def analyze_grid(scan_path: str | Path, sidecar_path: str | Path, *, space=None) -> GridAnalysis:
    """Rank the blocker-grid cells by how well they held back UV.

    Best cell = highest L* (closest to paper white) at saturation 1.0. The recommended
    saturation is the lowest at the winning hue whose L* stays within 2 L* of that hue's
    best — the Harmon lever: don't burn more ink than the paper can use.
    """
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    if "hues_deg" not in sidecar:
        raise AnalysisError(f"{sidecar_path} is not a blocker-grid sidecar")
    scan = load_image(scan_path, space=space)
    frame = detect_fiducials(scan, sidecar)
    samples = sample_cells(scan, sidecar, frame)

    ref_sidecar = {**sidecar, "cells": sidecar["references"]}
    ref_samples = sample_cells(scan, ref_sidecar, frame)

    ranking = sorted(samples, key=lambda s: -s["lstar"])
    full_sat = [s for s in ranking if s["saturation"] == max(sidecar["saturations"])]
    best = full_sat[0]

    warnings: list[str] = []
    clear_ref = next((r for r in ref_samples if r.get("ref") == "clear"), None)
    if clear_ref is not None and best["lstar"] - clear_ref["lstar"] < 20:
        warnings.append(
            "even the best blocker is close to the clear-film reference — "
            "exposure may be too long, or this ink set blocks poorly"
        )
    black_ref = next((r for r in ref_samples if r.get("ref") == "black"), None)
    if black_ref is not None and black_ref["lstar"] >= best["lstar"] - 1:
        warnings.append(
            "composite black blocks as well as the best hue — consider black as the blocker"
        )

    same_hue = sorted(
        (s for s in samples if s["hue_deg"] == best["hue_deg"]), key=lambda s: s["saturation"]
    )
    hue_best_lstar = max(s["lstar"] for s in same_hue)
    recommended = None
    for s in same_hue:  # ascending saturation: the first one close enough wins
        if s["lstar"] >= hue_best_lstar - 2.0:
            recommended = s["saturation"]
            break

    return GridAnalysis(
        ranking=ranking,
        references=ref_samples,
        best=best,
        recommended_saturation=recommended,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- zone grid


@dataclass
class ZoneAnalysis:
    """Best blocker hue at each density zone, ready to become zone control points."""

    zones: list[dict]  # [{"n": ..., "hue_deg": ..., "rgb": [...], "lstar": ...}, ...]
    hue_varies: bool
    warnings: list[str]

    def control_points(self) -> list[dict]:
        """Zone control points for a ``zone_hue`` profile blocker."""
        points = [{"n": 0.0, "rgb": [255, 255, 255]}]
        points += [{"n": z["n"], "rgb": list(z["rgb"])} for z in self.zones]
        return points

    def summary(self) -> str:
        lines = ["best blocking hue per density zone:"]
        lines += [
            f"  n {z['n']:.2f}   hue {z['hue_deg']:>4g}°   L* {z['lstar']:.1f}" for z in self.zones
        ]
        lines.append(
            "hue varies across the scale — the zone model is justified"
            if self.hue_varies
            else "one hue wins at every zone — stay with the simpler fixed-hue model"
        )
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def analyze_zone_grid(scan_path: str | Path, sidecar_path: str | Path, *, space=None) -> ZoneAnalysis:
    """Find the best-blocking hue at each density zone.

    Reports whether the winning hue actually changes across the scale. If it does not,
    the zone model buys nothing over fixed-hue and the analysis says so — PLAN.md only
    licenses the upgrade "if measurements justify it".
    """
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    if "zones" not in sidecar or "hues_deg" not in sidecar:
        raise AnalysisError(f"{sidecar_path} is not a zone-grid sidecar")
    scan = load_image(scan_path, space=space)
    frame = detect_fiducials(scan, sidecar)
    samples = sample_cells(scan, sidecar, frame)

    by_zone: dict[float, list[dict]] = {}
    for s in samples:
        by_zone.setdefault(s["zone_n"], []).append(s)

    zones = []
    for n in sorted(by_zone):
        best = max(by_zone[n], key=lambda s: s["lstar"])
        zones.append(
            {
                "n": float(n),
                "hue_deg": best["hue_deg"],
                "rgb": best["rgb"],
                "lstar": round(best["lstar"], 2),
            }
        )

    hues = {z["hue_deg"] for z in zones}
    warnings: list[str] = []
    if len(zones) < 2:
        warnings.append("fewer than two density zones read — cannot judge hue variation")
    lstars = [z["lstar"] for z in zones]
    if max(lstars) - min(lstars) < 2.0:
        warnings.append(
            "density zones barely differ in blocking — check exposure; the wedge may be more useful here"
        )
    return ZoneAnalysis(zones=zones, hue_varies=len(hues) > 1, warnings=warnings)


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    from . import use_utf8_console

    use_utf8_console()
    parser = argparse.ArgumentParser(prog="cyanoneg.analyze", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    wedge = sub.add_parser("wedge", help="derive a correction LUT from a scanned wedge print")
    wedge.add_argument("scan", type=Path)
    wedge.add_argument("sidecar", type=Path)
    wedge.add_argument("--space", default=None, help="override scan colour space if untagged")
    wedge.add_argument("--export", type=Path, default=None, help="write .cube and .acv beside this stem")

    grid = sub.add_parser("grid", help="rank blocker hues from a scanned grid print")
    grid.add_argument("scan", type=Path)
    grid.add_argument("sidecar", type=Path)
    grid.add_argument("--space", default=None)

    zone = sub.add_parser("zone-grid", help="find the best blocker hue per density zone")
    zone.add_argument("scan", type=Path)
    zone.add_argument("sidecar", type=Path)
    zone.add_argument("--space", default=None)

    args = parser.parse_args(argv)
    if args.command == "zone-grid":
        print(analyze_zone_grid(args.scan, args.sidecar, space=args.space).summary())
    elif args.command == "wedge":
        result = analyze_wedge(args.scan, args.sidecar, space=args.space)
        print(result.summary())
        if args.export:
            print(f"wrote {result.lut.export_cube(args.export.with_suffix('.cube'))}")
            print(f"wrote {result.lut.export_acv(args.export.with_suffix('.acv'))}")
    else:
        print(analyze_grid(args.scan, args.sidecar, space=args.space).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
