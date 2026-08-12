"""Tricolour cyanotype: three registered negatives from one RGB positive.

Three sequential cyanotype layers on one sheet, each from its own negative, each
chemically transformed to a different colour, building a subtractive CMY image.

The channel mapping and the physical print order are process requirements, not design
choices, so they live here as constants rather than as configuration. See
``TRICOLOUR_CONVERTER_PLAN_2026-08-11.md`` for the derivation and the measurements behind
them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from . import imageio as cio
from .blocker import apply_blocker
from .imageio import Image
from .profiles import PROFILE_DIR, Profile
from .targets import A4_MM, PPI, Target, _font, _mm, apply_frame

#: Rec.709 luma weights, the grey axis saturation is scaled about.
_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: Source channel each layer is derived from: red→cyan, green→magenta, blue→yellow.
#: A wrong permutation still yields three plausible negatives, so it is pinned by test.
CHANNEL_INDEX = {"cyan": 0, "magenta": 1, "yellow": 2}

#: The same mapping in words, for the set file and the manifest a human reads.
SOURCE_CHANNEL = {"cyan": "red", "magenta": "green", "yellow": "blue"}

#: Physical print order. Cyan is last because the alkaline carbonate bleach that the
#: magenta and yellow layers depend on destroys Prussian blue; printing cyan earlier
#: would erase it. Not configurable — a set that reorders this is not a valid set.
PRINT_ORDER = ("magenta", "yellow", "cyan")

#: The channel each layer's wedge must be *measured* through — its own complement.
#:
#: Not the same question as :data:`CHANNEL_INDEX`, which says where a layer's image comes
#: from. ``analyze_wedge`` normalises on L*, which is right for Prussian blue on white and
#: wrong for yellow, where L* barely moves across the whole density range: the curve would
#: come out of noise and look like a measurement.
RESPONSE_QUANTITY = {"cyan": "lstar_r", "magenta": "lstar_g", "yellow": "lstar_b"}

#: Bounds on a layer's ``scale``. Cotton rag moves 0.5-1% across a wet/dry cycle, so a
#: value outside this band is a typo or a units error rather than a shrinkage measurement.
SCALE_BOUNDS = (0.95, 1.05)


class TricolourSetError(ValueError):
    """Raised when a set file is structurally invalid or contradicts a process constant."""


#: Profile fields the three layers of a set are *allowed* to differ in.
#:
#: Stated as an exemption list rather than a list of fields that must agree, so the check
#: fails safe: adding a field to :class:`~cyanoneg.profiles.Profile` makes it mandatory to
#: agree by default. A whitelist of must-agree fields would instead go quietly stale.
ALLOWED_TO_DIFFER = frozenset(
    {
        "name",  # each layer is its own named profile
        "chemistry",  # the toning is what makes the layer that colour
        "provisional",  # one layer may be measured before the others are
        "lut",  # measured per layer; differing curves are the point
        "measurements",  # the raw patches behind that layer's own curve
    }
)

#: Everything else. These describe the shared apparatus and materials: if they differ, the
#: layers were not calibrated for the same print and the set is a fiction.
MUST_AGREE = tuple(f.name for f in fields(Profile) if f.name not in ALLOWED_TO_DIFFER)


def check_profile_agreement(profiles: dict[str, Profile]) -> list[str]:
    """Return a list of fields on which the layers' profiles disagree.

    A set whose layers were measured under different driver settings, film batches or
    working spaces is not a set — each curve describes a different process. ``blocker`` is
    included for a second, structural reason: the page background and the masks over the
    two non-owner wedge slots are one colour, so layers with different blockers cannot
    share a sheet at all.
    """
    problems: list[str] = []
    roles = list(profiles)
    if len(roles) < 2:
        return problems
    reference = profiles[roles[0]]
    for name in MUST_AGREE:
        expected = getattr(reference, name)
        for role in roles[1:]:
            actual = getattr(profiles[role], name)
            if actual != expected:
                problems.append(
                    f"layers disagree on {name}: {roles[0]} has {expected!r}, "
                    f"{role} has {actual!r}"
                )
    return problems


@dataclass
class TricolourLayer:
    """What genuinely varies between the three layers.

    ``role``, ``source_channel`` and ``print_order`` are deliberately absent: the role is
    the dictionary key that holds this layer, and the other two follow from it via
    :data:`SOURCE_CHANNEL` and :data:`PRINT_ORDER`. Storing them here would allow a set to
    represent a state the process cannot have, such as magenta printed third from red.
    """

    profile: str
    exposure_multiplier: float
    sensitizer: str = ""
    chemistry: str = ""
    #: Linear scale applied to this layer's *picture block only*, to meet paper that has
    #: already shrunk. 1.0 for print #1, which is the print that measures the real figure.
    #:
    #: The wedges and the control region are deliberately left unscaled: each is sampled
    #: through a homography built from its own four fiducials, which absorbs a uniform
    #: scale change for free. Shrinkage threatens registration of the picture and nothing
    #: else.
    scale: float = 1.0


@dataclass
class TricolourSet:
    """The three layers of one tricolour print, plus the settings shared across them."""

    name: str
    saturation_boost: float = 1.0
    border_mm: float = 10.0
    registration: dict[str, Any] = field(
        default_factory=lambda: {"fiducials": True, "letter": True}
    )
    layers: dict[str, TricolourLayer] = field(default_factory=dict)

    # ------------------------------------------------------------------ validation

    def validate(self) -> list[str]:
        """Return a list of problems; empty means valid."""
        problems: list[str] = []
        if not self.name.strip():
            problems.append("set has no name")
        if not isinstance(self.saturation_boost, (int, float)) or self.saturation_boost <= 0:
            problems.append(f"saturation_boost must be positive, got {self.saturation_boost!r}")
        if not isinstance(self.border_mm, (int, float)) or self.border_mm < 0:
            problems.append(f"border_mm must not be negative, got {self.border_mm!r}")

        for role in PRINT_ORDER:
            if role not in self.layers:
                problems.append(f"set is missing its {role} layer")
        for role in self.layers:
            if role not in PRINT_ORDER:
                problems.append(f"unknown layer {role!r}; expected one of {PRINT_ORDER}")

        for role, layer in self.layers.items():
            if not layer.profile.strip():
                problems.append(f"{role} layer names no profile")
            m = layer.exposure_multiplier
            if not isinstance(m, (int, float)) or m <= 0:
                problems.append(f"{role} exposure_multiplier must be positive, got {m!r}")
            s = layer.scale
            lo, hi = SCALE_BOUNDS
            if not isinstance(s, (int, float)) or not (lo <= s <= hi):
                problems.append(
                    f"{role} scale must lie within {SCALE_BOUNDS} — paper moves 0.5-1%, so "
                    f"{s!r} is a mistake rather than a measurement"
                )
        return problems

    def resolve(self, profile_dir: str | Path = PROFILE_DIR) -> dict[str, Profile]:
        """Load the three layers' profiles, refusing a set whose layers do not match.

        Resolution is the first moment the layers can be compared, so it is where the
        disagreement has to be caught. Returning three profiles that describe three
        different processes would put the error into the negatives instead.
        """
        problems = self.validate()
        if problems:
            raise TricolourSetError("invalid set: " + "; ".join(problems))

        profile_dir = Path(profile_dir)
        resolved: dict[str, Profile] = {}
        for role in PRINT_ORDER:
            name = self.layers[role].profile
            path = profile_dir / f"{name}.json"
            if not path.is_file():
                raise TricolourSetError(
                    f"{role} layer names profile {name!r}, but {path} does not exist"
                )
            resolved[role] = Profile.load(path)

        disagreements = check_profile_agreement(resolved)
        if disagreements:
            raise TricolourSetError(
                "the three layers were not calibrated for the same print: "
                + "; ".join(disagreements)
            )
        return resolved

    # ------------------------------------------------------------------ JSON round-trip

    def to_dict(self) -> dict[str, Any]:
        """Serialise, writing the derived invariants out for a human reader.

        ``source_channel`` and ``print_order`` are redundant with the constants by
        construction. They are written because the file is read at the enlarger by someone
        deciding which sheet to expose next, and re-checked on load so the redundancy can
        never become a lie.
        """
        return {
            "name": self.name,
            "saturation_boost": self.saturation_boost,
            "border_mm": self.border_mm,
            "registration": self.registration,
            "layers": {
                role: {
                    "print_order": PRINT_ORDER.index(role) + 1,
                    "source_channel": SOURCE_CHANNEL[role],
                    "profile": layer.profile,
                    "exposure_multiplier": layer.exposure_multiplier,
                    "sensitizer": layer.sensitizer,
                    "chemistry": layer.chemistry,
                    "scale": layer.scale,
                }
                for role, layer in self.layers.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TricolourSet:
        try:
            layers = {}
            for role, ld in (d.get("layers") or {}).items():
                if role in PRINT_ORDER:
                    _check_derived(role, ld)
                layers[role] = TricolourLayer(
                    profile=ld["profile"],
                    exposure_multiplier=ld["exposure_multiplier"],
                    sensitizer=ld.get("sensitizer", ""),
                    chemistry=ld.get("chemistry", ""),
                    scale=ld.get("scale", 1.0),
                )
            return cls(
                name=d["name"],
                saturation_boost=d.get("saturation_boost", 1.0),
                border_mm=d.get("border_mm", 10.0),
                registration=d.get("registration", {"fiducials": True, "letter": True}),
                layers=layers,
            )
        except KeyError as e:
            raise TricolourSetError(f"set is missing required field {e}") from e

    def save(self, path: str | Path) -> Path:
        problems = self.validate()
        if problems:
            raise TricolourSetError("refusing to save an invalid set: " + "; ".join(problems))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> TricolourSet:
        path = Path(path)
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise TricolourSetError(f"{path.name} is not valid JSON: {e}") from e
        tset = cls.from_dict(d)
        problems = tset.validate()
        if problems:
            raise TricolourSetError(f"{path.name} is invalid: " + "; ".join(problems))
        return tset


def _check_derived(role: str, layer: dict[str, Any]) -> None:
    """Refuse a layer whose written-out invariants disagree with the process constants.

    Silently preferring the constants would leave the operator holding a file that does not
    describe what the code did.
    """
    channel = layer.get("source_channel")
    if channel is not None and channel != SOURCE_CHANNEL[role]:
        raise TricolourSetError(
            f"{role} layer claims source_channel {channel!r}, but the process derives "
            f"{role} from {SOURCE_CHANNEL[role]!r}"
        )
    order = layer.get("print_order")
    expected = PRINT_ORDER.index(role) + 1
    if order is not None and order != expected:
        raise TricolourSetError(
            f"{role} layer claims print_order {order!r}, but the print order is "
            f"{PRINT_ORDER} so {role} is number {expected}"
        )


def format_seconds(seconds: int) -> str:
    """Whole seconds as ``mm:ss`` — the units the darkroom timer is actually set in.

    ``HANDOFF.md`` records the measured SPE as "13:30 = 810 s". Emitting only the seconds
    would leave that conversion to be done by hand, at the one moment it is most expensive
    to get wrong.
    """
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def layer_exposure(profile: Profile, layer: TricolourLayer) -> dict[str, Any]:
    """One layer's working exposure. The only place a multiplier is ever applied.

    Exposure has exactly one owner per value. The measured ``spe_seconds`` lives in the
    layer's profile and is identical across all three, because it is a property of paper,
    lamp and distance rather than of the layer — ``MUST_AGREE`` enforces that structurally,
    since ``exposure`` is not on the exemption list. The multiplier lives in the set. The
    product exists only here and in the manifest, so there is no second copy to disagree
    with the first, and no way for a later refactor to apply the multiplier twice.

    ``computed_seconds`` is rounded to 0.1 s purely to keep float noise out of an artefact
    a person reads and diffs: 810 x 1.1 is 891.0000000000001 in float64.
    """
    try:
        base = profile.exposure["spe_seconds"]
    except (KeyError, TypeError) as e:
        raise TricolourSetError(
            f"profile {profile.name!r} records no spe_seconds, so its working exposure "
            "cannot be computed — measure the standard printing exposure first"
        ) from e
    if not isinstance(base, (int, float)) or base <= 0:
        raise TricolourSetError(
            f"profile {profile.name!r} has a nonsensical spe_seconds {base!r}"
        )

    multiplier = layer.exposure_multiplier
    computed = base * multiplier
    instruction = math.floor(computed + 0.5)
    return {
        "base_spe_seconds": base,
        "exposure_multiplier": multiplier,
        "computed_seconds": round(computed, 1),
        "instruction_seconds": instruction,
        "instruction_display": format_seconds(instruction),
    }


def step_saturate(image: Image, amount: float) -> tuple[Image, float]:
    """Boost saturation by ``amount``, returning the image and the clipped fraction.

    Scales each pixel's distance from the Rec.709 grey axis, in linear light. The sources
    ask for "boost saturation 30–50%"; this is one interpretation of that instruction, not
    a transcription of it, and it will not match a perceptual saturation slider in
    Photoshop. It is chosen because it is defined, reversible and independent of the
    measured LUT — targets are printed through ``linear.json`` and never saturated, so the
    curve describes the process and the boost is a creative transform on the image alone.

    ``clipped_fraction`` is the fraction of pixels for which **at least one** component
    fell outside [0, 1] after scaling and before clipping. Boosting saturation drives
    out-of-gamut colour silently, and a print made from clipped data cannot be recovered
    by re-processing at a lower boost — the highlights are already flat.
    """
    if amount == 1.0:
        # A true no-op. Decoding and re-encoding at 1.0 would shift low code values and
        # put a silent difference between tricolour at boost 1.0 and the mono baseline.
        return image, 0.0
    if image.is_mono:
        raise ValueError("saturation needs three channels; got a mono image")

    linear = cio.to_linear(image.data, image.space)
    luma = (linear @ _REC709)[..., None]
    scaled = luma + (linear - luma) * np.float32(amount)

    outside = (scaled < 0.0) | (scaled > 1.0)
    clipped_fraction = float(outside.any(axis=-1).mean())

    encoded = cio.from_linear(np.clip(scaled, 0.0, 1.0), image.space)
    return image.replace(encoded), clipped_fraction


def extract_channel(image: Image, layer: str) -> Image:
    """Take ``layer``'s own source channel as a direct slice.

    Deliberately not routed through :func:`cyanoneg.mono.to_mono` with unit weights: that
    normalises its weights and round-trips through linear light, so the result would
    correlate with the channel without equalling it.
    """
    if layer not in CHANNEL_INDEX:
        raise ValueError(f"unknown layer {layer!r}; expected one of {tuple(CHANNEL_INDEX)}")
    if image.is_mono:
        raise ValueError(
            "cannot separate a mono image into tricolour layers — it has no colour to "
            "separate, and the three negatives would come out identical"
        )
    return image.replace(image.data[..., CHANNEL_INDEX[layer]])


# --------------------------------------------------------------------------- frame

#: The letter stamped into each layer's border. Three near-identical orange
#: transparencies are otherwise told apart only by holding them up to the light.
GLYPH = {"cyan": "C", "magenta": "M", "yellow": "Y"}

#: How much ink the glyph lays, as a fraction of full blocker.
#:
#: Not clear film. ``detect_fiducials`` thresholds on the dark tail of the scan, and a
#: clear-film glyph would print as dark as a fiducial. It is belt-and-braces rather than
#: the only defence — candidates must also pass ``squareness > 0.55``, ``fill > 0.45`` and
#: a size band, which a letterform is unlikely to satisfy — but it costs nothing, and the
#: placement below removes the risk structurally anyway.
GLYPH_COVERAGE = 0.5


def step_frame(
    image: Image,
    layer: str,
    border_px: int,
    blocker_rgb: tuple[int, int, int],
    saturation: float = 1.0,
    *,
    glyph: bool = True,
) -> tuple[Image, dict]:
    """Wrap a layer's picture in a border with fiducials, and stamp it with its letter.

    The border blocks UV at the sheet edge, leaving clean white paper that absorbs
    edge-etch, and its fiducials print as the darkest marks on the sheet. Layer 1 prints
    them onto the paper; layers 2 and 3 align to what is already there. Identical absolute
    positions on all three mean a fringed mark *is* misregistration — the check is a light
    table, not software.

    The glyph is drawn on its own sub-canvas, **mirrored**, and then pasted in. The page is
    composed in print orientation and flipped once on the way to film, so a letter drawn
    the right way round here would come out reversed on the film — which is the side
    actually read in a dim room.

    It is centred on the top border, between the two upper fiducials rather than beyond
    them. ``detect_fiducials`` takes the bounding box of all candidate centres and picks
    the blob nearest each corner of it, so a mark *inside* that box cannot redefine the
    frame even if it were detected. Placement, not just coverage, is what makes this safe.
    """
    if layer not in GLYPH:
        raise ValueError(f"unknown layer {layer!r}; expected one of {tuple(GLYPH)}")

    blocked = full_blocker_value(blocker_rgb, saturation)
    framed, fiducials = apply_frame(
        image.data, border_px, tuple(float(v) for v in blocked)
    )
    geometry: dict[str, Any] = {"fiducials": fiducials, "border_px": int(border_px)}

    if not glyph:
        geometry["glyph"] = None
        return Image(framed, image.space, ppi=image.ppi, bit_depth=image.bit_depth), geometry

    stamp = _glyph_stamp(GLYPH[layer], border_px, blocked)
    gh, gw = stamp.shape[:2]
    fid_size = fiducials["top_left"]["size_px"]
    fid_right = fiducials["top_right"]["x_px"]
    fid_left = fiducials["top_left"]["x_px"] + fid_size
    if gw >= fid_right - fid_left:
        raise ValueError(
            "the glyph is wider than the gap between the upper fiducials; it would sit "
            "outside their span and could redefine the detected frame"
        )
    x = (framed.shape[1] - gw) // 2
    y = (border_px - gh) // 2
    framed[y : y + gh, x : x + gw] = stamp

    geometry["glyph"] = {
        "letter": GLYPH[layer],
        "x_px": int(x),
        "y_px": int(y),
        "w_px": int(gw),
        "h_px": int(gh),
        "coverage": GLYPH_COVERAGE,
        "mirrored": True,
    }
    return Image(framed, image.space, ppi=image.ppi, bit_depth=image.bit_depth), geometry


def _glyph_stamp(letter: str, border_px: int, blocked: np.ndarray) -> np.ndarray:
    """Draw ``letter`` on a border-coloured tile, then mirror it for the film flip."""
    from PIL import ImageDraw

    h = max(8, int(border_px * 0.7))
    w = h
    tile = np.empty((h, w, 3), dtype=np.float32)
    tile[:, :] = blocked

    pil = PILImage.fromarray((tile * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil)
    ink = blocked + GLYPH_COVERAGE * (1.0 - blocked)
    font = _font(max(6, int(h * 0.8)))
    box = draw.textbbox((0, 0), letter, font=font)
    draw.text(
        ((w - (box[2] - box[0])) // 2 - box[0], (h - (box[3] - box[1])) // 2 - box[1]),
        letter,
        fill=tuple(int(round(v * 255.0)) for v in ink),
        font=font,
    )
    drawn = np.asarray(pil, dtype=np.float32) / 255.0
    return np.ascontiguousarray(drawn[:, ::-1])  # mirror now; the page flips once, later


# --------------------------------------------------------------------------- page

#: Size of the blocked control region, and the border that carries its fiducials.
CONTROL_MM = 40.0
CONTROL_BORDER_MM = 8.0


def full_blocker_value(
    blocker_rgb: tuple[int, int, int], saturation: float = 1.0
) -> np.ndarray:
    """The pixel value that lays maximum ink — what "blocked" means on this film.

    Taken from :func:`~cyanoneg.blocker.apply_blocker` rather than written down, so the
    page background can never drift from the value the wedges compute for themselves.
    """
    return apply_blocker(np.zeros((1, 1), dtype=np.float32), blocker_rgb, saturation)[0, 0]


def _scale_block(block: np.ndarray, factor: float, space: str) -> np.ndarray:
    """Resample an RGB block by ``factor``, in linear light.

    Per-channel, because the resize primitive works on single-channel float planes.
    Resampling encoded values would darken edges; the mono pipeline resamples in linear
    light for the same reason.
    """
    if factor == 1.0:
        return block
    from PIL import Image as PILImage

    h, w = block.shape[:2]
    dst = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    linear = cio.to_linear(block, space)
    planes = [
        np.asarray(
            PILImage.fromarray(linear[..., c].astype(np.float32), mode="F").resize(
                dst, PILImage.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        for c in range(3)
    ]
    return cio.from_linear(np.clip(np.stack(planes, axis=-1), 0.0, 1.0), space)


def control_region(
    blocker_rgb: tuple[int, int, int],
    saturation: float = 1.0,
    size_mm: float = CONTROL_MM,
    border_mm: float = CONTROL_BORDER_MM,
) -> tuple[np.ndarray, dict]:
    """A region held at full blocker on *every* layer, framed so it can be cropped.

    Nothing ever exposes it, which is what makes it worth printing. After the full
    M -> Y -> C cycle it answers three questions at once:

    - does the blocker stay opaque at 2.75x SPE, where it has never been characterised?
    - what do three coat-and-wash cycles deposit on paper that was never exposed?
    - and therefore, can the isolated wedges be read after cyan at all — since that stain
      floor is exactly what would corrupt them?

    The third is why this is cheap insurance rather than a nicety: it settles by
    measurement a question the plan otherwise settles by assertion.

    Read it in all three channels. A stain invisible in L* can be large in blue, which is
    the layer most at risk.

    Its fiducials are clear film and do print. That is deliberate: they locate the crop,
    and they double as a max-exposure reference at that spot on the sheet.
    """
    blocked = full_blocker_value(blocker_rgb, saturation)
    interior_px = _mm(size_mm - 2 * border_mm)
    interior = np.empty((interior_px, interior_px, 3), dtype=np.float32)
    interior[:, :] = blocked
    return apply_frame(interior, _mm(border_mm), tuple(float(v) for v in blocked))


def tricolour_page(
    picture: Image,
    wedges: dict[str, Target] | None,
    owner: str,
    blocker_rgb: tuple[int, int, int],
    saturation: float = 1.0,
    *,
    scale: float = 1.0,
    page_mm: tuple[float, float] = A4_MM,
    picture_gap_mm: float = 20.0,
    control_gap_mm: float = 5.0,
    slot_gap_mm: float = 5.0,
) -> tuple[Image, dict]:
    """Compose one layer's sheet, in print orientation.

    The background is **full blocker, not clear film**. ``calibration_page`` uses white,
    which is right there — its margins are never coated and never measured. Here it would
    be wrong three times over: clear film passes UV, so every uncoated margin would take a
    full exposure on three successive layers, and anything coated outside the picture would
    go to Dmax, then be bleached and toned, then have Prussian blue laid over it.

    Wedge slots are **isolated, not stacked**: ``owner``'s wedge is drawn and the other two
    slots are left at background, so each layer's wedge measures that layer alone. A
    stacked patch read through green carries magenta's absorption plus yellow's plus
    cyan's, and a curve derived from it linearises the neutral stack — three such curves
    are one curve read three ways, not three per-layer curves. Leaving the non-owner slots
    alone is the same operation as "background", so the masking costs nothing.

    ``scale`` moves the picture block only. The wedges and the control are sampled through
    homographies built from their own fiducials, which absorb a uniform scale change for
    free, so shrinkage threatens registration of the picture and nothing else.

    Returns the page and a placement dict recording every croppable region in print
    orientation, which is the orientation a scan is measured in.
    """
    if owner not in PRINT_ORDER:
        raise ValueError(f"unknown owner {owner!r}; expected one of {PRINT_ORDER}")
    if wedges is not None and set(wedges) != set(PRINT_ORDER):
        raise ValueError(
            "wedges must cover every layer so all three sheets share one geometry; "
            f"got {sorted(wedges)}"
        )

    blocked = full_blocker_value(blocker_rgb, saturation)
    page_w, page_h = _mm(page_mm[0]), _mm(page_mm[1])

    picture_block = _scale_block(picture.data, scale, picture.space)
    pic_h, pic_w = picture_block.shape[:2]

    control, control_fiducials = control_region(blocker_rgb, saturation)
    ctl_h, ctl_w = control.shape[:2]

    slot_views: dict[str, np.ndarray] = {}
    slot_h = slot_w = 0
    rows: list[tuple[str, int, int]] = [("picture", pic_h, pic_w)]
    if wedges is not None:
        slot_views = {
            role: np.ascontiguousarray(w.film[:, ::-1]) for role, w in wedges.items()
        }
        shapes = {v.shape[:2] for v in slot_views.values()}
        if len(shapes) != 1:
            raise ValueError(
                "the three wedges differ in size, so the slots would not line up across "
                "layers — generate them with identical levels and redundancy"
            )
        slot_h, slot_w = shapes.pop()
        rows.append(("wedges", slot_h, 3 * slot_w + 2 * _mm(slot_gap_mm)))
    rows.append(("control", ctl_h, ctl_w))

    gaps = [_mm(picture_gap_mm), _mm(control_gap_mm)][: len(rows) - 1]
    block_h = sum(h for _, h, _ in rows) + sum(gaps)
    block_w = max(w for _, _, w in rows)
    if block_w > page_w or block_h > page_h:
        raise ValueError(
            f"the sheet needs {block_w / PPI * 25.4:.0f} x {block_h / PPI * 25.4:.0f} mm "
            f"but the page is {page_mm[0]:.0f} x {page_mm[1]:.0f} mm"
        )

    canvas = np.empty((page_h, page_w, 3), dtype=np.float32)
    canvas[:, :] = blocked

    def _record(x: int, y: int, w: int, h: int, **extra: Any) -> dict:
        return {
            "x_px": int(x),
            "y_px": int(y),
            "w_px": int(w),
            "h_px": int(h),
            "x_mm": round(x / PPI * 25.4, 1),
            "y_mm": round(y / PPI * 25.4, 1),
            **extra,
        }

    placement: dict[str, Any] = {}
    y = (page_h - block_h) // 2
    for i, (kind, h, w) in enumerate(rows):
        x = (page_w - w) // 2
        if kind == "picture":
            canvas[y : y + h, x : x + w] = picture_block
            placement["picture"] = _record(x, y, w, h, scale=scale)
        elif kind == "wedges":
            slots = {}
            for role in PRINT_ORDER:
                if role == owner:
                    canvas[y : y + slot_h, x : x + slot_w] = slot_views[role]
                slots[role] = _record(
                    x, y, slot_w, slot_h, owned=role == owner, sidecar=f"wedge_{role}.json"
                )
                x += slot_w + _mm(slot_gap_mm)
            placement["wedges"] = slots
        else:
            canvas[y : y + h, x : x + w] = control
            placement["control"] = _record(
                x, y, w, h, fiducials=control_fiducials, read_in="all three channels"
            )
        if i < len(gaps):
            y += h + gaps[i]

    return Image(canvas, picture.space, ppi=PPI, bit_depth=picture.bit_depth), placement
