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
    # Build as a mono negative: 0 = clear film, 1 = full blocker.
    n = np.zeros((h, w), dtype=np.float32)
    half = h // 2
    n[half:, :] = 1.0  # bottom half: full blocker reference

    divider = max(2, _mm(0.5))
    for z in range(1, zones):
        x = z * _mm(zone_mm)
        n[:, x - divider // 2 : x + divider // 2] = 1.0

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
    margin_mm: float = 8.0,
) -> Target:
    """Hue × saturation sweep at full coverage, for finding the best UV blocker.

    Every cell is printed at maximum density (n = 1) in its own colour. After exposure at
    SPE, the cell that stays closest to paper-white is the best blocker; within that hue,
    the lowest saturation still holding paper-white sets the density-range scalar.

    Includes two references: CLEAR (film base — prints max black) and BLACK (RGB 0,0,0 —
    tests whether composite black outperforms any single hue, answered empirically rather
    than assumed).
    """
    cell, margin = _mm(cell_mm), _mm(margin_mm)
    cols, rows = len(hues), len(saturations)
    label = _mm(8)
    w = margin * 2 + label + cols * cell + cell  # + one reference column
    h = margin * 2 + label + rows * cell
    canvas = np.ones((h, w, 3), dtype=np.float32)

    cells = []
    for r, sat in enumerate(saturations):
        for c, hue in enumerate(hues):
            rgb = hue_to_rgb(hue, 1.0, 1.0)
            block = apply_blocker(np.ones((cell, cell), dtype=np.float32), rgb, sat)
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

    sidecar = {
        "hues_deg": list(hues),
        "saturations": list(saturations),
        "cell_mm": cell_mm,
        "cells": cells,
        "references": refs,
        "usage": "expose at SPE; best blocker = cell closest to paper-white; "
        "lowest saturation still paper-white sets the DR scalar",
    }
    return _finish("blocker_grid", canvas, sidecar)


# --------------------------------------------------------------------------- 256-step wedge


def step_wedge(
    blocker_rgb: tuple[int, int, int],
    saturation: float = 1.0,
    seed: int = 20260725,
    redundancy: int = 2,
    patch_mm: float = 5.5,
    border_mm: float = 8.0,
) -> Target:
    """Randomised 256-step tablet with anti-spike redundancy.

    - 256 levels × ``redundancy`` copies, positions shuffled by a recorded seed, so an ink
      spike corrupts one copy of one level, not a run of neighbours; copies are averaged.
    - A thick **full-blocker border**: on the print this edge stays unexposed and washes
      out, absorbing the lateral chemistry migration (edge-etch/peptization) that would
      otherwise corrupt edge patches.
    - Corner fiducials — clear-film squares set into the border, three solid and one
      (top-left) hollow, so detection also recovers orientation and mirroring.

    The wedge is generated in the measured blocker colour: calibration must exercise the
    same colours a real negative uses.
    """
    count = 256 * redundancy
    cols = 32
    rows = count // cols
    if cols * rows != count:
        raise ValueError(f"patch count {count} does not fill a {cols}-wide grid")

    patch, border = _mm(patch_mm), _mm(border_mm)
    w = border * 2 + cols * patch
    h = border * 2 + rows * patch

    rng = np.random.default_rng(seed)
    values = np.repeat(np.arange(256), redundancy)
    rng.shuffle(values)

    # Mono negative canvas: border at full density.
    n = np.ones((h, w), dtype=np.float32)
    cells = []
    for i, value in enumerate(values):
        r, c = divmod(i, cols)
        y, x = border + r * patch, border + c * patch
        # Patch value is the *positive* input level; the negative is its inversion.
        n[y : y + patch, x : x + patch] = 1.0 - value / 255.0
        cells.append(
            {
                "index": i,
                "row": int(r),
                "col": int(c),
                "value": int(value),
                "x_px": int(x),
                "y_px": int(y),
                "w_px": patch,
                "h_px": patch,
            }
        )

    # Fiducials: clear squares inset in the border, hollow one at top-left.
    fid = max(8, border // 2)
    inset = (border - fid) // 2
    positions = {
        "top_left": (inset, inset),
        "top_right": (inset, w - inset - fid),
        "bottom_left": (h - inset - fid, inset),
        "bottom_right": (h - inset - fid, w - inset - fid),
    }
    for name, (y, x) in positions.items():
        n[y : y + fid, x : x + fid] = 0.0
        if name == "top_left":  # hollow: refill the centre
            q = fid // 3
            n[y + q : y + fid - q, x + q : x + fid - q] = 1.0

    canvas = apply_blocker(n, blocker_rgb, saturation)

    sidecar = {
        "seed": seed,
        "redundancy": redundancy,
        "levels": 256,
        "grid": {"cols": cols, "rows": rows},
        "patch_mm": patch_mm,
        "border_mm": border_mm,
        "blocker_rgb": list(blocker_rgb),
        "saturation": saturation,
        "fiducials": {
            name: {"y_px": int(y), "x_px": int(x), "size_px": fid, "hollow": name == "top_left"}
            for name, (y, x) in positions.items()
        },
        "cells": cells,
        "usage": "print through linear.json only; scan the finished cyanotype print for analyze.py",
    }
    return _finish("step_wedge", canvas, sidecar)


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cyanoneg.targets", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("targets"))
    parser.add_argument("--all", action="store_true", help="generate all three targets")
    parser.add_argument("--exposure", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--wedge", action="store_true")
    parser.add_argument(
        "--blocker",
        type=str,
        default="255,0,0",
        help="blocker R,G,B for the exposure strip and wedge (default placeholder red)",
    )
    parser.add_argument("--saturation", type=float, default=1.0, help="wedge blocker saturation")
    parser.add_argument("--seed", type=int, default=20260725, help="wedge randomisation seed")
    args = parser.parse_args(argv)

    rgb = tuple(int(v) for v in args.blocker.split(","))
    if len(rgb) != 3:
        parser.error("--blocker must be R,G,B")

    wanted = []
    if args.all or args.exposure:
        wanted.append(exposure_strip(blocker_rgb=rgb))
    if args.all or args.grid:
        wanted.append(blocker_grid())
    if args.all or args.wedge:
        wanted.append(step_wedge(rgb, saturation=args.saturation, seed=args.seed))
    if not wanted:
        parser.error("nothing to do: pass --all or one of --exposure/--grid/--wedge")

    for target in wanted:
        tif, side = target.save(args.out)
        px = target.film.shape
        print(f"{tif}  ({px[1]}x{px[0]} px, {px[1]/PPI*25.4:.0f}x{px[0]/PPI*25.4:.0f} mm)  + {side.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
