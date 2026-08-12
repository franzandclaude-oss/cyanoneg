"""Generate print #1: three registered negatives, a manifest and a darkroom sheet.

Deliberately a script rather than a GUI mode. The Process tab is the largest and least
tested surface in the tricolour change, and print #1 needs none of it — every value that
matters ends up in the manifest, where it can be read and checked before any film is
committed to paper.

Two steps, because seeding is a one-off:

    # once — clone the measured profile into three provisional layer profiles + a set
    python scripts/make_print1.py seed

    # each time — generate the negatives from a positive
    python scripts/make_print1.py make PHOTO.tif

Then, before coating anything:

    python scripts/make_print1.py check

which runs the numeric channel test from the plan's verification section. Do not skip it:
a wrong channel permutation produces three entirely plausible orange transparencies, and
the eyeball version of the check passes on all of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyanoneg import use_utf8_console  # noqa: E402
from cyanoneg.blocker import recover_coverage  # noqa: E402
from cyanoneg.imageio import load_image  # noqa: E402
from cyanoneg.pipeline import PrintSize  # noqa: E402
from cyanoneg.profiles import PROFILE_DIR, Profile  # noqa: E402
from cyanoneg.targets import _mm  # noqa: E402
from cyanoneg.tricolour import (  # noqa: E402
    CHANNEL_INDEX,
    PRINT_ORDER,
    SOURCE_CHANNEL,
    TricolourSet,
    make_tricolour,
    seed_provisional_set,
)

#: 130 mm on the long edge is a paper constraint, not a layout one. Cotton rag grows
#: 0.5-1% wet, so a 240 mm print moves 1-2 mm across three wet/dry cycles and no fiducial
#: fixes that. Pre-shrink the sheets and keep print #1 small.
PRINT_SIZE = PrintSize(130.0, 100.0)

BASE_PROFILE = "CassArt 300 Sm"
SET_NAME = "CassArt 300 Sm — Tricolour"
SET_FILE = PROFILE_DIR / "CassArt 300 Sm — Tricolour.json"


def cmd_seed(args: argparse.Namespace) -> int:
    base = Profile.load(PROFILE_DIR / f"{args.base}.json")
    tset, clones = seed_provisional_set(base, SET_NAME, saturation_boost=args.saturation)

    for clone in clones:
        path = clone.save(PROFILE_DIR / f"{clone.name}.json")
        print(f"  wrote {path.name}")
    print(f"  wrote {tset.save(SET_FILE).name}")

    print()
    print("All three carry the measured LUT unchanged and spe_seconds unmultiplied.")
    print("What differentiates them on print #1 is the exposure multiplier — the one")
    print("per-layer quantity the sources actually give as a number.")
    return 0


def cmd_make(args: argparse.Namespace) -> int:
    tset = TricolourSet.load(SET_FILE)
    result = make_tricolour(
        args.source,
        tset,
        PRINT_SIZE,
        output_dir=args.out,
        stem=args.stem or Path(args.source).stem,
        wedges=not args.no_wedges,
    )

    for role in PRINT_ORDER:
        layer = result.manifest["layers"][role]
        print(
            f"  {layer['print_order']}. {role:<8} {layer['negative']:<16}"
            f" expose {layer['exposure']['instruction_display']}"
            f"  ({layer['exposure']['instruction_seconds']} s)"
        )
    print()
    print(f"  manifest    {args.out}/{result.manifest['output']['stem']}_tricolour.json")
    print(f"  wall sheet  {args.out}/{result.manifest['output']['stem']}_tricolour.md")

    if result.warnings:
        print()
        for warning in result.warnings:
            print(f"  ! {warning}")

    print()
    print("Now run the channel check before coating anything:")
    print(f"  python scripts/make_print1.py check --out {args.out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Numeric channel verification. The eyeball version passes on a wrong permutation."""
    from PIL import Image as PILImage

    manifest = json.loads(
        next(Path(args.out).glob("*_tricolour.json")).read_text(encoding="utf-8")
    )
    source = load_image(manifest["source"]["path"])
    if source.is_mono:
        print("  source is mono — nothing to separate")
        return 1

    rect = manifest["placement"]["picture"]
    border = _mm(manifest["set"]["border_mm"])
    blocker = {
        "model": "fixed_hue",
        "rgb": manifest["blocker"]["rgb"],
        "saturation": manifest["blocker"]["saturation"],
    }

    # How separable this source is at all. A natural photograph has RGB channels
    # correlated at ~0.9, which caps every score below and is why the criterion is
    # "highest", not "much higher": the ceiling is set by the source, not the separation.
    pairs = {
        "red-green": (0, 1), "red-blue": (0, 2), "green-blue": (1, 2),
    }
    inter = {
        name: abs(np.corrcoef(source.data[..., a].ravel(), source.data[..., b].ravel())[0, 1])
        for name, (a, b) in pairs.items()
    }
    print("  source inter-channel correlation: " + "  ".join(
        f"{n} {v:.2f}" for n, v in inter.items()
    ))
    print(f"  (the closer to 1.00, the less this test can discriminate)")
    print()

    print(f"  {'negative':<10}" + "".join(f"{n:>11}" for n in ("src red", "src green", "src blue")))
    ok = True
    thin: list[str] = []
    for role in PRINT_ORDER:
        film = load_image(Path(args.out) / manifest["layers"][role]["negative"])
        block = film.data[:, ::-1][
            rect["y_px"] : rect["y_px"] + rect["h_px"],
            rect["x_px"] : rect["x_px"] + rect["w_px"],
        ]
        # The border is a constant ~28% of the block; leaving it in drags every
        # correlation towards zero uniformly and hides the difference being measured.
        cover = recover_coverage(block[border:-border, border:-border], blocker)
        ch, cw = cover.shape

        scores = []
        for i in range(3):
            plane = np.ascontiguousarray(source.data[..., i])
            resized = np.asarray(
                PILImage.fromarray(plane, mode="F").resize(
                    (cw, ch), PILImage.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            scores.append(abs(np.corrcoef(cover.ravel(), resized.ravel())[0, 1]))

        own = scores[CHANNEL_INDEX[role]]
        runner_up = max(sorted(scores)[:2])
        passed = own == max(scores)
        ok &= passed
        if passed and own - runner_up < 0.02:
            thin.append(f"{role} leads {SOURCE_CHANNEL[role]} by only {own - runner_up:.3f}")
        marks = "".join(
            f"{v:>10.3f}" + ("*" if i == CHANNEL_INDEX[role] else " ")
            for i, v in enumerate(scores)
        )
        print(f"  {role:<10}{marks}  {'ok' if passed else 'FAIL'}")

    print()
    print("  * = the channel this layer must derive from")
    if not ok:
        print("  FAIL — the separation is wrong. Do not print.")
        for role in PRINT_ORDER:
            print(f"    {role} must come from {SOURCE_CHANNEL[role]}")
        return 1

    print("  PASS — every layer tracks its own channel most strongly")
    if thin:
        print()
        print("  Margins are thin, so this run is weak evidence rather than none:")
        for note in thin:
            print(f"    - {note}")
        print("  Re-run against a source with less correlated channels to confirm.")
    print()
    print("  Next: print to film and overlay on a light table before coating.")
    return 0


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(prog="make_print1", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="clone a measured profile into a provisional set")
    seed.add_argument("--base", default=BASE_PROFILE)
    seed.add_argument("--saturation", type=float, default=1.35)
    seed.set_defaults(func=cmd_seed)

    make = sub.add_parser("make", help="generate the three negatives from a positive")
    make.add_argument("source", help="RGB positive: .tif, .png or .jpg")
    make.add_argument("--out", default="print1")
    make.add_argument("--stem", default=None)
    make.add_argument("--no-wedges", action="store_true")
    make.set_defaults(func=cmd_make)

    check = sub.add_parser("check", help="numeric channel verification — run before coating")
    check.add_argument("--out", default="print1")
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
