"""Calibration target generators.

Three targets (PLAN.md), each written as a print-ready TIFF plus a **sidecar JSON**
recording seed, geometry and every patch's intended value — Phase 2's ``analyze.py`` reads
the layout from the sidecar rather than re-deriving it, so a target and its reading can
never disagree about what was printed where.

Targets are colour-blocked but **never LUT-corrected** (they print through
``profiles/linear.json``): a target pushed through a paper curve would measure the process
plus that curve.

Orientation: like every negative, targets are flipped horizontally on export (ink side
against the emulsion). The contact print un-mirrors them, so sidecar coordinates are in
**print orientation** — pre-flip generator space — which is what a flatbed scan of the
cyanotype print shows. The wedge's fiducials are asymmetric, so a scan of the film itself
is detectable and can be unflipped.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from . import __version__
from .blocker import apply_blocker, hue_to_rgb
from .imageio import DEFAULT_SPACE, Image, save_tiff

PPI = 360  # printer-native input resolution

#: Hue sweep for the blocker grid: red through amber/yellow to green — the region with
#: dye-printer UV-blocking precedent. Wider sweep costs patches; this range covers every
#: candidate the research surfaced.
GRID_HUES = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150)
GRID_SATURATIONS = (1.0, 0.9, 0.8, 0.65, 0.5, 0.35)


def _mm(mm: float) -> int:
    return int(round(mm / 25.4 * PPI))


def _font(px: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def apply_frame(
    canvas: np.ndarray,
    border_px: int,
    border_rgb: tuple[float, float, float],
) -> tuple[np.ndarray, dict]:
    """Wrap ``canvas`` in a dense border with clear-film corner fiducials.

    The border prints light (its colour blocks UV) and absorbs edge-etch; the fiducials
    are clear film, printing as the darkest marks on the sheet, which is what
    ``analyze.py`` keys its detection on. Three are solid; the top-left one is hollow,
    breaking every rotation/mirror symmetry.

    Returns the framed canvas and the fiducial geometry (coordinates on the framed
    canvas, print orientation).
    """
    h, w = canvas.shape[:2]
    framed = np.empty((h + 2 * border_px, w + 2 * border_px, 3), dtype=np.float32)
    framed[:, :] = np.asarray(border_rgb, dtype=np.float32)
    framed[border_px : border_px + h, border_px : border_px + w] = canvas

    fh, fw = framed.shape[:2]
    fid = max(8, border_px // 2)
    inset = (border_px - fid) // 2
    positions = {
        "top_left": (inset, inset),
        "top_right": (inset, fw - inset - fid),
        "bottom_left": (fh - inset - fid, inset),
        "bottom_right": (fh - inset - fid, fw - inset - fid),
    }
    for name, (y, x) in positions.items():
        framed[y : y + fid, x : x + fid] = 1.0  # clear film
        if name == "top_left":  # hollow: refill the centre with border colour
            q = fid // 3
            framed[y + q : y + fid - q, x + q : x + fid - q] = np.asarray(border_rgb, dtype=np.float32)

    fiducials = {
        name: {"y_px": int(y), "x_px": int(x), "size_px": fid, "hollow": name == "top_left"}
        for name, (y, x) in positions.items()
    }
    return framed, fiducials


@dataclass
class Target:
    """A generated target: film image (RGB float, film orientation) + sidecar dict."""

    name: str
    film: np.ndarray  # flipped, ready to print
    sidecar: dict

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tif = save_tiff(
            directory / f"{self.name}.tif",
            Image(self.film, DEFAULT_SPACE, ppi=PPI),
        )
        side = directory / f"{self.name}.json"
        side.write_text(json.dumps(self.sidecar, indent=2) + "\n", encoding="utf-8")
        return tif, side


def _finish(name: str, canvas: np.ndarray, sidecar: dict) -> Target:
    """Common export path: record shared metadata, then flip to film orientation."""
    sidecar = {
        "generator": f"cyanoneg {__version__}",
        "target": name,
        "ppi": PPI,
        "working_space": DEFAULT_SPACE,
        "coordinates": "print orientation (pre-flip); film TIFF is mirrored horizontally",
        **sidecar,
    }
    film = np.ascontiguousarray(canvas[:, ::-1])
    return Target(name=name, film=film, sidecar=sidecar)


# --------------------------------------------------------------------------- exposure strip


def exposure_strip(
    zones: int = 8,
    zone_mm: float = 22.0,
    height_mm: float = 30.0,
    blocker_rgb: tuple[int, int, int] = (255, 0, 0),
) -> Target:
    """Strip for finding the Standard Printing Exposure.

    Each numbered zone is half clear film (prints toward max black), half full blocker
    (must stay paper-white). Cover zones progressively with card during exposure; SPE is
    the shortest time whose clear half matches the next zone's — max black through film
    base, no wasted exposure.
    """
    w, h = _mm(zone_mm) * zones, _mm(height_mm)
    # Mono negative, blocker convention: 1 = white = clear film, 0 = black = full blocker.
    n = np.ones((h, w), dtype=np.float32)
    half = h // 2
    n[half:, :] = 0.0  # bottom half: full blocker reference

    divider = max(2, _mm(0.5))
    for z in range(1, zones):
        x = z * _mm(zone_mm)
        n[:, x - divider // 2 : x + divider // 2] = 0.0

    rgb = apply_blocker(n, blocker_rgb, 1.0)

    # Zone numbers, printed in blocker colour on the clear half so they are visible on
    # both the film and the finished print.
    pil = PILImage.fromarray((rgb * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    font = _font(_mm(4))
    colour = tuple(int(v) for v in np.array(blocker_rgb))
    for z in range(zones):
        draw.text((z * _mm(zone_mm) + _mm(2), _mm(2)), str(z + 1), fill=colour, font=font)
    canvas = np.asarray(pil, dtype=np.float32) / 255.0

    sidecar = {
        "zones": zones,
        "zone_mm": zone_mm,
        "height_mm": height_mm,
        "blocker_rgb": list(blocker_rgb),
        "layout": [
            {
                "zone": z + 1,
                "x_px": z * _mm(zone_mm),
                "w_px": _mm(zone_mm),
                "clear_half": "top",
                "blocker_half": "bottom",
            }
            for z in range(zones)
        ],
        "usage": "cover zones progressively; SPE = shortest time whose clear half matches the next",
    }
    return _finish("exposure_strip", canvas, sidecar)


# --------------------------------------------------------------------------- blocker grid


def blocker_grid(
    hues: tuple[float, ...] = GRID_HUES,
    saturations: tuple[float, ...] = GRID_SATURATIONS,
    cell_mm: float = 9.0,
    margin_mm: float = 4.0,
    border_mm: float = 8.0,
) -> Target:
    """Hue × saturation sweep at full coverage, for finding the best UV blocker.

    Every cell is printed at maximum density (n = 1) in its own colour. After exposure at
    SPE, the cell that stays closest to paper-white is the best blocker; within that hue,
    the lowest saturation still holding paper-white sets the density-range scalar.

    Includes two references: CLEAR (film base — prints max black) and BLACK (RGB 0,0,0 —
    tests whether composite black outperforms any single hue, answered empirically rather
    than assumed).

    Framed like the wedge — composite-black border with clear corner fiducials — so
    ``analyze.py`` reads both scans through one detection path.
    """
    cell, margin = _mm(cell_mm), _mm(margin_mm)
    border = _mm(border_mm)
    cols, rows = len(hues), len(saturations)
    label = _mm(8)
    w = margin * 2 + label + cols * cell + cell  # + one reference column
    h = margin * 2 + label + rows * cell
    canvas = np.ones((h, w, 3), dtype=np.float32)

    cells = []
    for r, sat in enumerate(saturations):
        for c, hue in enumerate(hues):
            rgb = hue_to_rgb(hue, 1.0, 1.0)
            # zeros = black in the negative = maximum ink, i.e. full coverage.
            block = apply_blocker(np.zeros((cell, cell), dtype=np.float32), rgb, sat)
            y, x = margin + label + r * cell, margin + label + c * cell
            canvas[y : y + cell, x : x + cell] = block
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "hue_deg": hue,
                    "saturation": sat,
                    "rgb": [int(round(float(v) * 255)) for v in block[0, 0]],
                    "x_px": x,
                    "y_px": y,
                    "w_px": cell,
                    "h_px": cell,
                }
            )

    # Reference column: CLEAR (top) and BLACK (bottom).
    xr = margin + label + cols * cell + cell // 4
    refs = [
        {"ref": "clear", "rgb": [255, 255, 255]},
        {"ref": "black", "rgb": [0, 0, 0]},
    ]
    for i, ref in enumerate(refs):
        y = margin + label + i * cell * 2
        canvas[y : y + cell, xr : xr + cell // 2 + cell // 4] = np.array(ref["rgb"], dtype=np.float32) / 255.0
        ref.update({"x_px": xr, "y_px": y, "w_px": cell // 2 + cell // 4, "h_px": cell})

    # Edge labels in neutral dark grey (prints regardless of which hue wins).
    pil = PILImage.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    font = _font(_mm(3))
    grey = (60, 60, 60)
    for c, hue in enumerate(hues):
        draw.text((margin + label + c * cell + cell // 3, margin), f"{hue:g}", fill=grey, font=font)
    for r, sat in enumerate(saturations):
        draw.text((margin, margin + label + r * cell + cell // 3), f"{sat:g}", fill=grey, font=font)
    canvas = np.asarray(pil, dtype=np.float32) / 255.0

    canvas, fiducials = apply_frame(canvas, border, (0.0, 0.0, 0.0))
    for item in cells + refs:
        item["x_px"] += border
        item["y_px"] += border

    sidecar = {
        "hues_deg": list(hues),
        "saturations": list(saturations),
        "cell_mm": cell_mm,
        "border_mm": border_mm,
        "border_rgb": [0, 0, 0],
        "fiducials": fiducials,
        "cells": cells,
        "references": refs,
        "usage": "expose at SPE; best blocker = cell closest to paper-white; "
        "lowest saturation still paper-white sets the DR scalar",
    }
    return _finish("blocker_grid", canvas, sidecar)


# --------------------------------------------------------------------------- zone grid

#: Density zones sampled by the zone grid. Skips n = 0 (clear film blocks nothing) and
#: concentrates where cyanotype actually separates tone.
ZONE_LEVELS = (0.25, 0.45, 0.65, 0.85, 1.0)


def zone_blocker_grid(
    hues: tuple[float, ...] = GRID_HUES,
    zones: tuple[float, ...] = ZONE_LEVELS,
    cell_mm: float = 9.0,
    margin_mm: float = 4.0,
    border_mm: float = 8.0,
) -> Target:
    """Hue sweep repeated at several densities, for the zone-varying blocker.

    Each row is one density zone; each column one hue. The ink laid down at n = 0.25 is a
    different mixture from the same hue at n = 1.0, and it need not block UV equally well —
    this target measures whether that is true for *this* printer and film, rather than
    assuming the EDN result transfers.

    Read with :func:`cyanoneg.analyze.analyze_zone_grid`, which returns the best hue per
    zone; those become the zone control points.
    """
    cell, margin, border = _mm(cell_mm), _mm(margin_mm), _mm(border_mm)
    label = _mm(8)
    cols, rows = len(hues), len(zones)
    w = margin * 2 + label + cols * cell
    h = margin * 2 + label + rows * cell
    canvas = np.ones((h, w, 3), dtype=np.float32)

    cells = []
    for r, n in enumerate(zones):
        for c, hue in enumerate(hues):
            rgb = hue_to_rgb(hue, 1.0, 1.0)
            # n is ink density; the blocker takes negative value, so pass 1 - n.
            block = apply_blocker(np.full((cell, cell), 1.0 - float(n), dtype=np.float32), rgb, 1.0)
            y, x = margin + label + r * cell, margin + label + c * cell
            canvas[y : y + cell, x : x + cell] = block
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "hue_deg": hue,
                    "zone_n": float(n),
                    "rgb": [int(round(float(v) * 255)) for v in block[0, 0]],
                    "x_px": x,
                    "y_px": y,
                    "w_px": cell,
                    "h_px": cell,
                }
            )

    pil = PILImage.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    font = _font(_mm(3))
    grey = (60, 60, 60)
    for c, hue in enumerate(hues):
        draw.text((margin + label + c * cell + cell // 3, margin), f"{hue:g}", fill=grey, font=font)
    for r, n in enumerate(zones):
        draw.text((margin, margin + label + r * cell + cell // 3), f"{n:g}", fill=grey, font=font)
    canvas = np.asarray(pil, dtype=np.float32) / 255.0

    canvas, fiducials = apply_frame(canvas, border, (0.0, 0.0, 0.0))
    for cell_meta in cells:
        cell_meta["x_px"] += border
        cell_meta["y_px"] += border

    sidecar = {
        "hues_deg": list(hues),
        "zones": list(zones),
        "cell_mm": cell_mm,
        "border_mm": border_mm,
        "fiducials": fiducials,
        "cells": cells,
        "usage": "expose at SPE; per density row the whitest cell names that zone's best hue",
    }
    return _finish("zone_grid", canvas, sidecar)


# --------------------------------------------------------------------------- step wedge

#: Levels and copies for the wedge. Deliberately few levels, heavily replicated.
#:
#: The first build used 256 levels with 2 copies, following PLAN.md. On real paper two
#: copies of one level, printed at different places on a hand-coated sheet, disagreed by a
#: median of ~1.0 L*, while 256 levels across a ~58 L* scale puts neighbouring levels only
#: 0.23 L* apart. Raw per-level readings were therefore mostly noise: the measured response
#: reversed direction 117 times and 50 levels tripped the spike detector on a print with
#: nothing physically wrong with it.
#:
#: **This does not much affect the derived curve**, and it is worth being precise about
#: that. ``lut.derive_correction`` fits 21 PCHIP knots regardless of how many points it is
#: given, so it was already averaging that noise away. Re-binning the measured 256x2 sheets
#: into 32x16 moves the resulting LUT by a mean of 0.006 and a worst case of 0.045 — real,
#: but not the difference between a working calibration and a broken one.
#:
#: What the redesign actually buys, which is why it is still the default:
#:  - Spike detection becomes meaningful. At k=2 a flag means "two noisy readings differ",
#:    which fired on a fifth of all levels and told you nothing. At k=16 an outlier is
#:    visible against 15 siblings, so a flag implicates the print.
#:  - The density-range references stop resting on two patches each.
#:  - Nothing is spent asking the process for precision it cannot deliver.
#:
#: 21-step tablets have been the alt-process convention for decades for the same reasons.
DEFAULT_LEVELS = 32
DEFAULT_REDUNDANCY = 16


def _grid_shape(count: int, target_aspect: float = 2.0) -> tuple[int, int]:
    """Columns and rows for ``count`` patches, closest to ``target_aspect`` (width:height).

    Only exact factorisations are considered, so every level keeps the same number of
    copies and the grid has no gaps.
    """
    best = None
    for cols in range(1, count + 1):
        if count % cols:
            continue
        rows = count // cols
        score = abs((cols / rows) - target_aspect)
        if best is None or score < best[0]:
            best = (score, cols, rows)
    assert best is not None
    return best[1], best[2]


#: Default shuffle seed for the step wedge. A named constant because the seed is not a
#: preference: it is the only thing that maps a scanned patch back to the level it was
#: printed at. Callers that record their configuration need to record *this* number, so it
#: has to be referable rather than buried in a signature default.
WEDGE_SEED = 20260725


def step_wedge(
    blocker_rgb: tuple[int, int, int],
    saturation: float = 1.0,
    seed: int = WEDGE_SEED,
    redundancy: int = DEFAULT_REDUNDANCY,
    levels: int = DEFAULT_LEVELS,
    patch_mm: float = 5.5,
    border_mm: float = 8.0,
) -> Target:
    """Randomised step tablet with anti-spike redundancy.

    - ``levels`` levels × ``redundancy`` copies, positions shuffled by a recorded seed, so
      an ink spike corrupts one copy of one level rather than a run of neighbours, and so
      the copies of a level sample different parts of the sheet. Copies are averaged, which
      is what converts coating variation from a bias into a shrinking error term.
    - A thick **full-blocker border**: on the print this edge stays unexposed and washes
      out, absorbing the lateral chemistry migration (edge-etch/peptization) that would
      otherwise corrupt edge patches.
    - Corner fiducials — clear-film squares set into the border, three solid and one
      (top-left) hollow, so detection also recovers orientation and mirroring.

    The wedge is generated in the measured blocker colour: calibration must exercise the
    same colours a real negative uses.

    See :data:`DEFAULT_LEVELS` for why the defaults favour few levels and many copies.
    """
    if levels < 2:
        raise ValueError(f"levels must be at least 2, got {levels}")
    if redundancy < 1:
        raise ValueError(f"redundancy must be at least 1, got {redundancy}")

    count = levels * redundancy
    cols, rows = _grid_shape(count)

    patch, border = _mm(patch_mm), _mm(border_mm)

    rng = np.random.default_rng(seed)
    values = np.repeat(np.arange(levels), redundancy)
    rng.shuffle(values)

    # Mono negative patch area; the dense border comes from apply_frame.
    n = np.empty((rows * patch, cols * patch), dtype=np.float32)
    cells = []
    for i, value in enumerate(values):
        r, c = divmod(i, cols)
        y, x = r * patch, c * patch
        # Patch value is the *positive* input level; the negative is its inversion.
        n[y : y + patch, x : x + patch] = 1.0 - value / (levels - 1)
        cells.append(
            {
                "index": i,
                "row": int(r),
                "col": int(c),
                "value": int(value),
                "x_px": int(x + border),
                "y_px": int(y + border),
                "w_px": patch,
                "h_px": patch,
            }
        )

    full_blocker = apply_blocker(np.zeros((1, 1), dtype=np.float32), blocker_rgb, saturation)[0, 0]
    canvas, fiducials = apply_frame(apply_blocker(n, blocker_rgb, saturation), border, tuple(full_blocker))

    sidecar = {
        "seed": seed,
        "redundancy": redundancy,
        "levels": levels,
        "grid": {"cols": cols, "rows": rows},
        "patch_mm": patch_mm,
        "border_mm": border_mm,
        "blocker_rgb": list(blocker_rgb),
        "saturation": saturation,
        "fiducials": fiducials,
        "cells": cells,
        "usage": "print through linear.json only; scan the finished cyanotype print for analyze.py",
    }
    return _finish("step_wedge", canvas, sidecar)


# --------------------------------------------------------------------------- CLI


# --------------------------------------------------------------------------- combined sheet

#: A4 in millimetres. The page exists so the blocker grid and the step wedge can be coated,
#: exposed, washed and dried as one piece of paper.
A4_MM = (210.0, 297.0)


def calibration_page(
    blocker_rgb: tuple[int, int, int],
    saturation: float = 1.0,
    seed: int = 20260725,
    gap_mm: float = 20.0,
    page_mm: tuple[float, float] = A4_MM,
    **wedge_kwargs,
) -> Target:
    """Blocker grid and step wedge on one sheet, stacked and centred.

    Both targets on one piece of paper share a coating pass, an exposure, a wash and a
    drying — which is the point. Measured sheet-to-sheet variation was 0.19 log units of
    density range on identical stimuli, dwarfing everything else in the calibration; two
    targets on separate sheets cannot be combined into one profile without carrying that
    error. Sharing a sheet removes it by construction.

    What sharing costs is evenness: the two targets sit at different distances from the
    lamp. Centred on the pair at a 350 mm lamp distance that difference is 0.005 log units
    between their centres — a fortieth of what it replaces. **Centre the page under the
    lamp**; the trade only holds while it is centred.

    Each target keeps its own fiducials and its own references, so analysis is unchanged:
    scan the two halves separately (or crop one scan) and feed each to ``analyze.py`` with
    its existing per-target sidecar. The page sidecar records where each one sits, in print
    orientation, so a single scan can also be split programmatically.
    """
    grid = blocker_grid()
    wedge = step_wedge(blocker_rgb, saturation=saturation, seed=seed, **wedge_kwargs)

    # Work in print orientation and mirror once at the end, so the sidecar offsets are
    # directly comparable with each target's own coordinates.
    parts = [("step_wedge", wedge), ("blocker_grid", grid)]  # wedge on top
    page_w, page_h = _mm(page_mm[0]), _mm(page_mm[1])
    gap = _mm(gap_mm)

    heights = [p.film.shape[0] for _, p in parts]
    widths = [p.film.shape[1] for _, p in parts]
    block_w, block_h = max(widths), sum(heights) + gap * (len(parts) - 1)
    if block_w > page_w or block_h > page_h:
        raise ValueError(
            f"targets need {block_w / PPI * 25.4:.0f} x {block_h / PPI * 25.4:.0f} mm but the "
            f"page is {page_mm[0]:.0f} x {page_mm[1]:.0f} mm"
        )

    # Paper white is 1.0 in the negative's space: unprinted film, which the paper sees as
    # full exposure. The page margin is never measured, so its value only has to be legal.
    canvas = np.ones((page_h, page_w, 3), dtype=np.float32)

    y = (page_h - block_h) // 2
    placement = {}
    for name, part in parts:
        print_view = np.ascontiguousarray(part.film[:, ::-1])  # undo the per-target mirror
        h, w = print_view.shape[:2]
        x = (page_w - w) // 2
        canvas[y : y + h, x : x + w] = print_view
        placement[name] = {
            "x_px": int(x),
            "y_px": int(y),
            "w_px": int(w),
            "h_px": int(h),
            "x_mm": round(x / PPI * 25.4, 1),
            "y_mm": round(y / PPI * 25.4, 1),
            "sidecar": f"{name}.json",
        }
        y += h + gap

    sidecar = {
        "page_mm": list(page_mm),
        "gap_mm": gap_mm,
        "placement": placement,
        "blocker_rgb": list(blocker_rgb),
        "saturation": saturation,
        "usage": (
            "print through linear.json only; coat, expose, wash and dry as one sheet, "
            "centred under the lamp. Scan each target separately (or crop one scan) and "
            "analyse it with its own per-target sidecar."
        ),
    }
    return _finish("calibration_page", canvas, sidecar)


def main(argv: list[str] | None = None) -> int:
    from . import use_utf8_console

    use_utf8_console()
    parser = argparse.ArgumentParser(prog="cyanoneg.targets", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("targets"))
    parser.add_argument("--all", action="store_true", help="generate all three targets")
    parser.add_argument("--exposure", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--zone-grid", action="store_true", help="hue sweep per density zone (phase 3)")
    parser.add_argument("--wedge", action="store_true")
    parser.add_argument(
        "--page",
        action="store_true",
        help="blocker grid and step wedge on one A4 sheet, so both share a coating and a wash",
    )
    parser.add_argument(
        "--blocker",
        type=str,
        default="255,0,0",
        help="blocker R,G,B for the exposure strip and wedge (default placeholder red)",
    )
    parser.add_argument("--saturation", type=float, default=1.0, help="wedge blocker saturation")
    parser.add_argument("--seed", type=int, default=WEDGE_SEED, help="wedge randomisation seed")
    parser.add_argument(
        "--levels",
        type=int,
        default=DEFAULT_LEVELS,
        help=f"wedge tone levels (default {DEFAULT_LEVELS}); more levels means finer steps "
        "but fewer copies of each to average coating variation out of",
    )
    parser.add_argument(
        "--redundancy",
        type=int,
        default=DEFAULT_REDUNDANCY,
        help=f"copies of each wedge level (default {DEFAULT_REDUNDANCY})",
    )
    args = parser.parse_args(argv)

    rgb = tuple(int(v) for v in args.blocker.split(","))
    if len(rgb) != 3:
        parser.error("--blocker must be R,G,B")

    wanted = []
    if args.all or args.exposure:
        wanted.append(exposure_strip(blocker_rgb=rgb))
    if args.all or args.grid:
        wanted.append(blocker_grid())
    if args.zone_grid:  # not in --all: only needed when upgrading to the zone model
        wanted.append(zone_blocker_grid())
    if args.all or args.wedge:
        wanted.append(
            step_wedge(
                rgb,
                saturation=args.saturation,
                seed=args.seed,
                levels=args.levels,
                redundancy=args.redundancy,
            )
        )
    if args.page:
        wanted.append(
            calibration_page(
                rgb,
                saturation=args.saturation,
                seed=args.seed,
                levels=args.levels,
                redundancy=args.redundancy,
            )
        )
    if not wanted:
        parser.error(
            "nothing to do: pass --all or one of --exposure/--grid/--zone-grid/--wedge/--page"
        )

    for target in wanted:
        tif, side = target.save(args.out)
        px = target.film.shape
        print(f"{tif}  ({px[1]}x{px[0]} px, {px[1]/PPI*25.4:.0f}x{px[0]/PPI*25.4:.0f} mm)  + {side.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
