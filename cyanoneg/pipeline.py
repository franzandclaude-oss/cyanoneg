"""Orchestration: positive scan → print-ready negative.

The order below is mandated by PLAN.md (per Reeder/Anderson and Ware) and asserted by
test — the correction curve applies to the **positive, before inversion**, and the mirror
flip comes last. Getting either wrong silently produces wrong tones or an unprintable
negative, which is exactly the failure mode this tool exists to remove.

    1. load positive          (16-bit TIFF preferred, float32 internally)
    2. mono conversion        (channel mixer)
    3. apply paper LUT        (in the profile's working_space, on the positive)
    4. resize                 (to print size at output ppi, resampled in linear light)
    5. invert
    6. colour block
    7. flip horizontal        (ink side against emulsion)
    8. export                 (16-bit TIFF)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from . import imageio as cio
from .blocker import apply_blocker, apply_zone_blocker
from .imageio import Image
from .mono import DEFAULT_WEIGHTS, to_mono
from .profiles import Profile

DEFAULT_OUTPUT_PPI = 360  # Epson's native input resolution


@dataclass(frozen=True)
class PrintSize:
    """Target print dimensions in millimetres. The image is fitted within, never cropped."""

    width_mm: float
    height_mm: float

    def pixels(self, ppi: float) -> tuple[int, int]:
        return (
            int(round(self.width_mm / 25.4 * ppi)),
            int(round(self.height_mm / 25.4 * ppi)),
        )

    def fit_scale(self, src_width: int, src_height: int, ppi: float) -> float:
        box_w, box_h = self.pixels(ppi)
        return min(box_w / src_width, box_h / src_height)

    def oriented_for(self, src_width: int, src_height: int) -> PrintSize:
        """Swap width and height if that fits the image larger.

        Entering "240 × 180" means a sheet of that size, not a commitment to landscape.
        Without this, a portrait image in a landscape box fits to the 180 mm height and
        the long edge silently comes out at 180 mm instead of 240 — a smaller print than
        asked for, with no warning.
        """
        if src_width <= 0 or src_height <= 0 or self.width_mm == self.height_mm:
            return self
        swapped = PrintSize(self.height_mm, self.width_mm)
        # ppi cancels out of the comparison; any positive value gives the same answer.
        if swapped.fit_scale(src_width, src_height, 360) > self.fit_scale(src_width, src_height, 360):
            return swapped
        return self


# --------------------------------------------------------------------------- steps
# Each step is a named function so the ordering test can patch and record them.


def step_mono(image: Image, weights: tuple[float, float, float]) -> Image:
    return to_mono(image, weights)


def step_apply_lut(image: Image, profile: Profile) -> Image:
    """Apply the paper curve to the *positive*, in the profile's working space."""
    data = image.data
    if image.space != profile.working_space:
        data = cio.convert_space(data, image.space, profile.working_space)
    return Image(
        data=profile.lut.apply(data),
        space=profile.working_space,
        ppi=image.ppi,
        bit_depth=image.bit_depth,
    )


def step_resize(image: Image, size: PrintSize, ppi: float, auto_orient: bool = True) -> Image:
    """Fit within the print size, resampling in linear light to avoid halo artefacts."""
    src_w, src_h = image.size
    if auto_orient:
        size = size.oriented_for(src_w, src_h)
    box_w, box_h = size.pixels(ppi)
    scale = min(box_w / src_w, box_h / src_h)
    dst = (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))))

    linear = cio.to_linear(image.data, image.space)
    pil = PILImage.fromarray(linear.astype(np.float32), mode="F")
    resized = np.asarray(pil.resize(dst, PILImage.Resampling.LANCZOS), dtype=np.float32)
    encoded = cio.from_linear(np.clip(resized, 0.0, 1.0), image.space)
    return Image(data=encoded, space=image.space, ppi=ppi, bit_depth=image.bit_depth)


def step_invert(image: Image) -> Image:
    return image.replace(1.0 - image.data)


def step_blocker(image: Image, profile: Profile) -> Image:
    if profile.blocker.get("model") == "zone_hue":
        zones = profile.blocker.get("zones")
        if not zones:
            raise ValueError(
                f"profile {profile.name!r} declares a zone blocker but carries no zones — "
                "print and measure the zone blocker grid first"
            )
        return image.replace(apply_zone_blocker(image.data, zones))

    rgb = profile.blocker.get("rgb")
    sat = profile.blocker.get("saturation")
    if rgb is None or sat is None:
        raise ValueError(
            f"profile {profile.name!r} has no blocker colour yet — "
            "print and measure the HSB blocker grid first"
        )
    return image.replace(apply_blocker(image.data, tuple(rgb), float(sat)))


def step_flip(image: Image) -> Image:
    return image.replace(np.ascontiguousarray(image.data[:, ::-1]))


# --------------------------------------------------------------------------- pipeline


def make_negative(
    source: str | Path | Image,
    profile: Profile,
    print_size: PrintSize,
    *,
    output_path: str | Path | None = None,
    output_ppi: float = DEFAULT_OUTPUT_PPI,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    space: cio.Space | None = None,
    raw_scan: bool = False,
    auto_orient: bool = True,
) -> Image:
    """Run the full positive → negative pipeline.

    ``raw_scan=True`` is the flagged path for un-inverted lab scans: the source is a
    negative, so it is inverted to a positive immediately after loading, and the rest of
    the pipeline is unchanged.

    ``auto_orient`` (default on) lets the print size follow the image's orientation, so a
    portrait image asked to print at 240 × 180 mm gets 240 mm on its long edge rather than
    being fitted to the 180 mm height.

    Returns the final negative; writes it to ``output_path`` when given.
    """
    image = source if isinstance(source, Image) else cio.load_image(source, space=space)

    if raw_scan:
        image = step_invert(image)  # raw scan → positive; the tonal pipeline needs a positive

    image = step_mono(image, weights)  # 2
    image = step_apply_lut(image, profile)  # 3 — on the positive, before inversion
    image = step_resize(image, print_size, output_ppi, auto_orient)  # 4
    image = step_invert(image)  # 5
    image = step_blocker(image, profile)  # 6
    image = step_flip(image)  # 7

    if output_path is not None:
        cio.save_tiff(output_path, image)  # 8
    return image


# --------------------------------------------------------------------------- batch


SOURCE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass
class BatchResult:
    """Outcome of one batch run. Failures are collected, never raised mid-run."""

    written: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.written) + len(self.failed)

    def summary(self) -> str:
        lines = [f"{len(self.written)} of {self.total} negatives written"]
        lines += [f"  FAILED {path.name}: {why}" for path, why in self.failed]
        return "\n".join(lines)


def batch_negatives(
    sources: Iterable[str | Path] | str | Path,
    profile: Profile,
    print_size: PrintSize,
    output_dir: str | Path,
    *,
    suffix: str = "_negative",
    progress=None,
    **kwargs,
) -> BatchResult:
    """Process many positives with one profile.

    ``sources`` may be a directory (scanned non-recursively for image files) or an
    explicit iterable of paths. One bad file does not stop the run: its error is recorded
    and the rest continue, because discovering at file 200 that file 3 was untagged is
    not a reason to lose the other 197.

    ``progress`` is called as ``progress(index, total, path)`` before each file.
    """
    if isinstance(sources, (str, Path)) and Path(sources).is_dir():
        paths = sorted(p for p in Path(sources).iterdir() if p.suffix.lower() in SOURCE_SUFFIXES)
    elif isinstance(sources, (str, Path)):
        paths = [Path(sources)]
    else:
        paths = [Path(p) for p in sources]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = BatchResult()

    for i, path in enumerate(paths):
        if progress is not None:
            progress(i, len(paths), path)
        destination = output_dir / f"{path.stem}{suffix}.tif"
        if destination.resolve() == path.resolve():
            result.failed.append((path, "output would overwrite the source"))
            continue
        try:
            make_negative(path, profile, print_size, output_path=destination, **kwargs)
        except Exception as e:  # noqa: BLE001 - one bad file must not end the run
            result.failed.append((path, str(e)))
        else:
            result.written.append(destination)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .profiles import PROFILE_DIR

    parser = argparse.ArgumentParser(prog="cyanoneg.pipeline", description="Batch-process positives")
    parser.add_argument("sources", type=Path, help="a directory of positives, or one image")
    parser.add_argument("--profile", type=Path, required=True, help="profile .json (or a name in profiles/)")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--width", type=float, required=True, help="print width in mm")
    parser.add_argument("--height", type=float, required=True, help="print height in mm")
    parser.add_argument("--ppi", type=float, default=DEFAULT_OUTPUT_PPI)
    parser.add_argument("--space", default=None, help="override source colour space if untagged")
    parser.add_argument("--raw-scan", action="store_true", help="sources are un-inverted negatives")
    parser.add_argument(
        "--no-auto-orient",
        action="store_true",
        help="keep the print box as given instead of matching each image's orientation",
    )
    args = parser.parse_args(argv)

    profile_path = args.profile if args.profile.exists() else PROFILE_DIR / f"{args.profile}.json"
    profile = Profile.load(profile_path)
    if profile.provisional:
        print(f"note: {profile.name!r} is provisional — tones are a starting point, not a calibration")

    def show(i: int, total: int, path: Path) -> None:
        print(f"[{i + 1}/{total}] {path.name}")

    result = batch_negatives(
        args.sources,
        profile,
        PrintSize(args.width, args.height),
        args.out,
        progress=show,
        output_ppi=args.ppi,
        space=args.space,
        raw_scan=args.raw_scan,
        auto_orient=not args.no_auto_orient,
    )
    print(result.summary())
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
