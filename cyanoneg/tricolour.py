"""Tricolour cyanotype: three registered negatives from one RGB positive.

Three sequential cyanotype layers on one sheet, each from its own negative, each
chemically transformed to a different colour, building a subtractive CMY image.

The channel mapping and the physical print order are process requirements, not design
choices, so they live here as constants rather than as configuration. See
``TRICOLOUR_CONVERTER_PLAN_2026-08-11.md`` for the derivation and the measurements behind
them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from . import __version__
from . import imageio as cio
from . import pipeline
from .blocker import apply_blocker
from .imageio import Image
from .pipeline import PrintSize
from .profiles import PROFILE_DIR, Profile
from .targets import A4_MM, PPI, WEDGE_SEED, Target, _font, _mm, apply_frame, step_wedge

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


def shared_calibration_identity(profile: Profile) -> dict[str, Any]:
    """The part of a profile that must be identical across the three layers.

    Derived from :data:`MUST_AGREE`, so it is the same projection the agreement check uses
    and cannot drift from it.
    """
    d = profile.to_dict()
    return {name: d[name] for name in MUST_AGREE}


def calibration_fingerprint(profile: Profile) -> str:
    """A short stable digest of the apparatus a profile was measured on.

    The manifest records a profile *name*, and a name does not preserve a run — the file
    behind it can be recalibrated or overwritten the same afternoon. The fingerprint lets a
    later reader tell whether the profile on disk is still the one that made the negatives.
    """
    payload = json.dumps(
        shared_calibration_identity(profile), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    the right way round here would come out reversed on the film.

    Which side "correct" refers to is a real choice, confirmed rather than assumed: the
    letter reads properly when the film is held up **ink-side towards you**, the way you
    pick a sheet up to see which one it is. It therefore reads reversed when the film is
    lying face-down in printing position — as does the picture, which is the entire point
    of the flip. The glyph exists to stop three near-identical orange transparencies being
    confused while being handled, so it is matched to handling, not to exposure.

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


#: Film base: blocks nothing. What the page falls back to outside ``blocker_extent_mm``.
CLEAR_FILM = 1.0

#: The frame every ``placement`` rectangle is expressed in.
#:
#: Composition happens in print orientation and the flip comes last, so these coordinates
#: match a flatbed scan of the finished cyanotype — the contact print un-mirrors the film.
#: That is the frame the measurements are taken in, and it is the right default.
PRINT_FRAME = "print"

#: The frame the exported TIFFs are in: print orientation mirrored left-to-right.
FILM_FRAME = "film"


def film_rect(rect: dict[str, Any], page_w_px: int) -> dict[str, Any]:
    """Mirror one ``placement`` rectangle from print orientation into film orientation.

    Exists because the alternative is writing ``page_w - x - w`` by hand at each call site,
    and getting that wrong is close to invisible here: the layout is centred, so a rectangle
    cropped in the wrong frame lands within a millimetre of the right place and returns a
    picture that looks entirely correct. It is off by a mirror, not by a mile.

    Only ``x`` moves. The export flips horizontally, so ``y``, ``w`` and ``h`` are the same
    in both frames — which is exactly why the mistake survives a casual look at the numbers.

    Prefer this to un-mirroring the image: the rectangle is four integers and the film is a
    72 MB array.

    This locates the crop; it does not orient its contents. Pixels cut out with a mirrored
    rectangle are still mirrored, so anything compared against the source positive — or
    against a scan of the print — needs :func:`to_print_orientation` applied to the crop.
    """
    if not 0 <= rect["x_px"] <= page_w_px - rect["w_px"]:
        raise ValueError(
            f"rectangle at x={rect['x_px']} w={rect['w_px']} does not fit a "
            f"{page_w_px}px page — mirroring it would put it off the sheet"
        )
    mirrored = dict(rect)
    mirrored["x_px"] = int(page_w_px - rect["x_px"] - rect["w_px"])
    mirrored["x_mm"] = round(mirrored["x_px"] / PPI * 25.4, 1)
    mirrored["frame"] = FILM_FRAME
    return mirrored


def to_print_orientation(film: np.ndarray) -> np.ndarray:
    """Un-mirror a film-orientation array so ``placement`` rectangles index it directly.

    The named inverse of the export flip. Copies, so use :func:`film_rect` instead when the
    array is large and only a crop is wanted.
    """
    return np.ascontiguousarray(film[:, ::-1])


def tricolour_page(
    picture: Image,
    wedges: dict[str, Target] | None,
    owner: str,
    blocker_rgb: tuple[int, int, int],
    saturation: float = 1.0,
    *,
    scale: float = 1.0,
    page_mm: tuple[float, float] = A4_MM,
    #: How far the blocker extends past the outermost element, in mm. ``None`` floods the
    #: whole sheet, which is the safe default and what R3 asks for.
    #:
    #: A number trades ink for a promise: everything beyond that margin becomes clear film,
    #: and clear film passes UV. The promise is that no *coated* paper lies out there. That
    #: is why the region is recorded in the manifest and printed on the wall sheet as a
    #: coating limit — the guard has to be somewhere a person will actually read it, not
    #: only in the head of whoever chose the number.
    blocker_extent_mm: float | None = None,
    picture_gap_mm: float = 20.0,
    #: Wider than the 5 mm between wedge slots, and for a different reason. The wedge's
    #: lower fiducials and the control's upper ones both sit ~2 mm inside their own
    #: borders, so a 5 mm gap left only ~9 mm between two sets of detectable marks. The
    #: manifest crops are exact, but a skewed scan has little room before one target's
    #: furniture lands inside another's crop.
    control_gap_mm: float = 10.0,
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

    ``blocker_extent_mm`` bounds the flood. The bound is taken over **every** element
    including the two wedge slots this sheet does not own, so all three sheets end up with
    an identical blocked region. Deriving it from the owner's slot alone would save more
    ink and would be a serious mistake: a strip blocked on magenta and clear on yellow
    takes a full unfiltered 2228 s exposure through the second negative, which is the
    precise failure the blocked background exists to prevent.

    Returns the page and a placement dict recording every croppable region in print
    orientation, which is the orientation a scan is measured in.
    """
    if owner not in PRINT_ORDER:
        raise ValueError(f"unknown owner {owner!r}; expected one of {PRINT_ORDER}")
    if blocker_extent_mm is not None and blocker_extent_mm < 0:
        raise ValueError(
            f"blocker_extent_mm must be >= 0, got {blocker_extent_mm}; a negative margin "
            "would clear film over the elements themselves"
        )
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

    if blocker_extent_mm is not None:
        placement["blocker_region"] = _bound_blocker(
            canvas, placement, _mm(blocker_extent_mm), blocker_extent_mm, _record
        )

    return Image(canvas, picture.space, ppi=PPI, bit_depth=picture.bit_depth), placement


def _bound_blocker(
    canvas: np.ndarray,
    placement: dict[str, Any],
    pad: int,
    extent_mm: float,
    record: Any,
) -> dict[str, Any]:
    """Clear the film outside ``pad`` pixels of the outermost element, in place.

    Written as four edge slices rather than a mask because the four strips are provably
    disjoint from every element rectangle: the region is the bounding box of those
    rectangles, grown outwards. Nothing an element occupies can be cleared, whatever the
    layout does next.
    """
    page_h, page_w = canvas.shape[:2]
    rects = [placement["picture"], placement["control"], *placement.get("wedges", {}).values()]
    x0 = max(0, min(r["x_px"] for r in rects) - pad)
    y0 = max(0, min(r["y_px"] for r in rects) - pad)
    x1 = min(page_w, max(r["x_px"] + r["w_px"] for r in rects) + pad)
    y1 = min(page_h, max(r["y_px"] + r["h_px"] for r in rects) + pad)

    canvas[:y0, :] = CLEAR_FILM
    canvas[y1:, :] = CLEAR_FILM
    canvas[y0:y1, :x0] = CLEAR_FILM
    canvas[y0:y1, x1:] = CLEAR_FILM

    cleared = 1.0 - ((x1 - x0) * (y1 - y0)) / float(page_w * page_h)
    # Measured, not assumed. The wall sheet turns this into "coat W x H centred on the
    # sheet", which is only a safe instruction if the region really is centred; the rows
    # are centred individually, so it is, but a future layout change could break that
    # silently and the sheet would keep giving the old instruction.
    off_x = abs((x0 + x1) - page_w) / 2.0
    off_y = abs((y0 + y1) - page_h) / 2.0
    return record(
        x0,
        y0,
        x1 - x0,
        y1 - y0,
        extent_mm=extent_mm,
        w_mm=round((x1 - x0) / PPI * 25.4, 1),
        h_mm=round((y1 - y0) / PPI * 25.4, 1),
        cleared_fraction=round(cleared, 4),
        offset_from_centre_mm=[round(off_x / PPI * 25.4, 2), round(off_y / PPI * 25.4, 2)],
        centred=bool(max(off_x, off_y) <= _mm(0.5)),
        note=(
            "blocker inside this rectangle, clear film outside; identical on all three "
            "sheets. Coated paper must stay inside it — clear film passes UV."
        ),
    )


# --------------------------------------------------------------------------- run

#: Fraction of clipped pixels above which the saturation boost is worth a warning.
CLIP_WARN = 0.02


@dataclass
class TricolourResult:
    """What one run produced, and everything needed to explain it later."""

    paths: dict[str, Path]
    manifest: dict[str, Any]
    fiducials: dict[str, Any]
    clipped_fraction: float
    warnings: list[str] = field(default_factory=list)


def output_name(stem: str, role: str) -> str:
    """``<stem>_1M`` / ``_2Y`` / ``_3C`` — numbered so the files sort into printing order.

    The order is the one thing about this process that cannot be recovered from the files
    themselves, and getting it wrong destroys the print rather than degrading it.
    """
    return f"{stem}_{PRINT_ORDER.index(role) + 1}{role[0].upper()}"


def make_tricolour(
    source: str | Path | Image,
    tset: TricolourSet,
    print_size: PrintSize,
    *,
    output_dir: str | Path,
    stem: str,
    profile_dir: str | Path = PROFILE_DIR,
    wedges: bool = True,
    wedge_levels: int = 16,
    wedge_redundancy: int = 4,
    #: Explicit rather than left to the generator's default. The patch layout is shuffled
    #: by this number, so it is the difference between a scan that can be read and a
    #: 60 mm square of unattributable greys.
    wedge_seed: int = WEDGE_SEED,
    space: cio.Space | None = None,
    raw_scan: bool = False,
    output_ppi: float = PPI,
    auto_orient: bool = True,
    page_mm: tuple[float, float] = A4_MM,
    blocker_extent_mm: float | None = None,
) -> TricolourResult:
    """One RGB positive in, three registered negatives and a manifest out.

    The step order is fixed by real dependencies, not preference: saturation is a
    three-channel operation so it precedes the channel split; the frame writes into an
    ``(h, w, 3)`` canvas so it follows the blocker; and framing and composition both
    precede the flip, so the emitted geometry is in print orientation — which is the
    orientation a scan is measured in.

    Saturation is applied **once**, to the shared positive, before the split. Doing it per
    layer would apply a three-channel operation to three separate monochrome planes, which
    is not the same transform and not the one the sources describe.

    The print size is resolved once and reused, so a portrait source cannot orient one
    layer differently from another and quietly break registration.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    problems = tset.validate()
    if problems:
        raise TricolourSetError("refusing to run an invalid set: " + "; ".join(problems))
    profiles = tset.resolve(profile_dir)

    image = source if isinstance(source, Image) else cio.load_image(source, space=space)
    source_name = None if isinstance(source, Image) else str(source)
    if raw_scan:
        image = pipeline.step_invert(image)
    if image.is_mono:
        raise ValueError(
            "a mono positive has no colour to separate — the three negatives would come "
            "out identical and three darkroom sessions would produce a grey image"
        )

    src_w, src_h = image.size
    boosted, clipped = step_saturate(image, tset.saturation_boost)
    if clipped > CLIP_WARN:
        warnings.append(
            f"saturation boost {tset.saturation_boost} clipped {clipped:.1%} of pixels; "
            "colour driven out of gamut here cannot be recovered by re-processing"
        )

    # Resolved once, then handed to all three layers.
    oriented = print_size.oriented_for(src_w, src_h) if auto_orient else print_size

    reference = profiles[PRINT_ORDER[0]]
    blocker_rgb = tuple(reference.blocker["rgb"])
    saturation = float(reference.blocker["saturation"])

    wedge_targets = None
    wedge_manifest: dict[str, Any] | None = None
    if wedges:
        wedge_targets = {
            role: step_wedge(
                blocker_rgb,
                saturation=saturation,
                levels=wedge_levels,
                redundancy=wedge_redundancy,
                seed=wedge_seed,
            )
            for role in PRINT_ORDER
        }
        # Written, not just named. The patch layout is shuffled, so a scan of a wedge is
        # unreadable without the sidecar that says which level each cell holds, and
        # ``analyze_wedge`` takes that file as an argument. The manifest has always
        # pointed at ``wedge_<role>.json``; until now nothing created them.
        #
        # The three are identical while the seed is shared, and are still written per
        # role: the manifest addresses them per role, and a set that ever varies the seed
        # per layer would otherwise silently alias three layers onto one layout.
        for role, target in wedge_targets.items():
            (output_dir / f"wedge_{role}.json").write_text(
                json.dumps(target.sidecar, indent=2) + "\n", encoding="utf-8"
            )
        wedge_manifest = {
            "levels": wedge_levels,
            "redundancy": wedge_redundancy,
            "seed": wedge_seed,
            "sidecars": {role: f"wedge_{role}.json" for role in PRINT_ORDER},
            "note": (
                "the sidecar maps each cell to the level it was printed at; the layout is "
                "shuffled by seed, so a scan cannot be read without it"
            ),
        }

    border_px = _mm(tset.border_mm)
    paths: dict[str, Path] = {}
    layer_manifest: dict[str, Any] = {}
    fiducials: dict[str, Any] = {}
    placements: dict[str, dict[str, Any]] = {}

    for role in PRINT_ORDER:
        layer, profile = tset.layers[role], profiles[role]

        plane = extract_channel(boosted, role)
        plane = pipeline.step_apply_lut(plane, profile)
        plane = pipeline.step_resize(plane, oriented, output_ppi, auto_orient=False)
        plane = pipeline.step_invert(plane)
        plane = pipeline.step_blocker(plane, profile)

        framed, geometry = step_frame(
            plane,
            role,
            border_px,
            blocker_rgb,
            saturation,
            glyph=bool(tset.registration.get("letter", True)),
        )
        page, placement = tricolour_page(
            framed,
            wedge_targets,
            role,
            blocker_rgb,
            saturation,
            scale=layer.scale,
            page_mm=page_mm,
            blocker_extent_mm=blocker_extent_mm,
        )
        film = pipeline.step_flip(page)

        path = output_dir / f"{output_name(stem, role)}.tif"
        cio.save_tiff(path, film)
        paths[role] = path
        fiducials[role] = geometry["fiducials"]
        placements[role] = placement

        layer_manifest[role] = {
            # This layer's own slot, recorded per layer because which slot a sheet owns
            # is the one thing about the page that is *not* shared between them. The
            # scan-back reads this to know what to crop.
            "wedge_slot": placement["wedges"][role] if wedge_targets else None,
            "print_order": PRINT_ORDER.index(role) + 1,
            "source_channel": SOURCE_CHANNEL[role],
            "response_quantity": RESPONSE_QUANTITY[role],
            "negative": path.name,
            "sensitizer": layer.sensitizer,
            "chemistry": layer.chemistry,
            "scale": layer.scale,
            "exposure": layer_exposure(profile, layer),
            "profile": {
                "name": profile.name,
                "provisional": profile.provisional,
                "calibration_fingerprint": calibration_fingerprint(profile),
                "calibration_identity": shared_calibration_identity(profile),
                "lut": {
                    "size": profile.lut.size,
                    "values": [round(float(v), 9) for v in profile.lut.values],
                },
            },
            "geometry": geometry,
        }

    # Page geometry is identical across the three layers by construction, and tested to be.
    # The per-layer ``owned`` flag is deliberately stripped here: it is a property of a
    # sheet, not of the page, and folding three sheets' flags into one shared record can
    # only ever be right for one of them.
    shared = placements[PRINT_ORDER[0]]
    page_geometry: dict[str, Any] = {
        "picture": shared["picture"],
        "control": shared["control"],
        "note": "identical on all three sheets; each layer's own slot is under layers.<role>",
        # Stated rather than implied. Every rectangle below is pre-flip, which is what a
        # scan of the finished print shows; the TIFFs are mirrored. Cropping a TIFF with
        # these numbers is off by a mirror and, on a centred layout, looks correct.
        "frame": PRINT_FRAME,
        "frame_note": (
            "print orientation (pre-flip) — matches a flatbed scan of the cyanotype. "
            "The exported TIFFs are mirrored: convert with tricolour.film_rect(rect, "
            "page_pixels[0]) before cropping film."
        ),
    }
    if "blocker_region" in shared:
        page_geometry["blocker_region"] = shared["blocker_region"]
        region = shared["blocker_region"]
        warnings.append(
            f"blocker bounded to {region['w_mm']:.0f} x {region['h_mm']:.0f} mm "
            f"({region['cleared_fraction']:.0%} of the sheet is clear film) — coated paper "
            "outside that rectangle takes an unfiltered exposure on all three layers"
        )
    if "wedges" in shared:
        page_geometry["wedges"] = {
            role: {k: v for k, v in slot.items() if k != "owned"}
            for role, slot in shared["wedges"].items()
        }

    provisional = [r for r in PRINT_ORDER if profiles[r].provisional]
    if provisional:
        warnings.append(
            f"{', '.join(provisional)} still provisional — this print is the experiment "
            "that measures them, not a calibrated result"
        )

    manifest = {
        "generator": f"cyanoneg {__version__}",
        "set": tset.to_dict(),
        "source": {
            "path": source_name,
            "pixels": [src_w, src_h],
            "raw_scan_inverted": raw_scan,
        },
        "output": {
            "stem": stem,
            "ppi": output_ppi,
            "print_size_mm": [oriented.width_mm, oriented.height_mm],
            "auto_orient": auto_orient,
            "page_mm": list(page_mm),
            "page_pixels": [int(page_mm[0] / 25.4 * PPI), int(page_mm[1] / 25.4 * PPI)],
            "flipped_for_film": True,
            "blocker_extent_mm": blocker_extent_mm,
        },
        "saturation": {
            "boost": tset.saturation_boost,
            "clipped_pixel_fraction": round(clipped, 6),
            "warn_above": CLIP_WARN,
        },
        "blocker": {"rgb": list(blocker_rgb), "saturation": saturation},
        "wedge": wedge_manifest,
        "placement": page_geometry,
        "layers": layer_manifest,
        "warnings": warnings,
    }

    (output_dir / f"{stem}_tricolour.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"{stem}_tricolour.md").write_text(
        darkroom_sheet(manifest), encoding="utf-8"
    )

    return TricolourResult(
        paths=paths,
        manifest=manifest,
        fiducials=fiducials,
        clipped_fraction=clipped,
        warnings=warnings,
    )


#: What the sources specify per layer, in numbers. Everything else about a seeded set is
#: cloned from the measured base profile rather than invented.
PROVISIONAL_LAYERS = {
    "magenta": {
        "exposure_multiplier": 1.5,
        "sensitizer": "10/10",
        "chemistry": "expose → wash → sodium carbonate bleach → madder root tone",
    },
    "yellow": {
        "exposure_multiplier": 2.75,
        "sensitizer": "10/10",
        "chemistry": "heavy overexpose → wash → carbonate bleach to Fe(III) hydroxide",
    },
    "cyan": {
        "exposure_multiplier": 1.1,
        "sensitizer": "5/5 (1:1 dilute)",
        "chemistry": "classic, untoned",
    },
}


def seed_provisional_set(
    base: Profile, name: str, saturation_boost: float = 1.35
) -> tuple[TricolourSet, list[Profile]]:
    """Clone a measured profile into three provisional layer profiles, plus a set.

    All three carry the base's **measured LUT, unchanged**. Nothing has measured how
    bleaching or madder toning reshapes the scale on this paper, and a curve invented from
    a plausible-looking shape would be worse than none: it would look authoritative while
    being fiction. The measured curve is not a guess — it is the true response of this ink,
    film, printer and paper through the exposure and wash, which the first two layers still
    share up to the point of bleaching. It is a baseline, not a claim about what happens at
    1.5x or 2.75x exposure; that is what print #1 is for.

    What differentiates the layers on print #1 is therefore the exposure multiplier, which
    is the one per-layer quantity the sources give numerically.

    ``spe_seconds`` is cloned unmodified. The multiplier lives only in the set, so the
    working exposure exists in exactly one place: the manifest.
    """
    if base.provisional:
        raise TricolourSetError(
            f"profile {base.name!r} is itself provisional — seed from a measured profile, "
            "or the three layers inherit a curve nothing has verified"
        )

    profiles: list[Profile] = []
    layers: dict[str, TricolourLayer] = {}
    for role in PRINT_ORDER:
        spec = PROVISIONAL_LAYERS[role]
        clone = dataclasses.replace(
            base,
            name=f"{base.paper or base.name} — {role.capitalize()}",
            chemistry=spec["chemistry"],
            provisional=True,
            measurements={"raw_patches": [], "scan_date": None},
        )
        profiles.append(clone)
        layers[role] = TricolourLayer(
            profile=clone.name,
            exposure_multiplier=spec["exposure_multiplier"],
            sensitizer=spec["sensitizer"],
            chemistry=spec["chemistry"],
        )

    tset = TricolourSet(name=name, saturation_boost=saturation_boost, layers=layers)
    return tset, profiles


#: Where each wedge slot sits on the finished print, left to right.
#:
#: The contact print un-mirrors the film, so the slots appear on paper in the same order
#: the manifest records them — which is print order. Worth stating rather than deriving in
#: someone's head at a scanner: the three squares are pixel-identical, and nothing
#: downstream can catch a mislabelled crop.
SLOT_POSITIONS = ("left", "middle", "right")


def slot_positions(manifest: dict[str, Any]) -> dict[str, str]:
    """Map each layer to its wedge slot's position on the print, or {} if there are none."""
    wedges = manifest["placement"].get("wedges")
    if not wedges or len(wedges) != len(SLOT_POSITIONS):
        return {}
    left_to_right = sorted(wedges, key=lambda role: wedges[role]["x_mm"])
    return dict(zip(left_to_right, SLOT_POSITIONS))


def analyze_layer_wedge(
    manifest: dict[str, Any] | str | Path,
    role: str,
    scan_path: str | Path,
    *,
    manifest_dir: str | Path | None = None,
    space: Any = None,
):
    """Read one layer's wedge back from a scan, through that layer's own quantity.

    ``scan_path`` is a scan cropped to the one wedge slot. It cannot be the whole sheet:
    ``detect_fiducials`` takes the bounding box of every candidate mark it finds, and a
    full sheet carries the picture's fiducials, the control's, and the printed wedge's, so
    the frame it derived would span the page rather than the target.

    Which slot to crop is not recoverable from the image. All three wedges are generated
    from one seed and are pixel-identical, so only position on the sheet identifies them —
    see :func:`slot_positions`, and the wall sheet, which names it per layer.

    The quantity comes from the manifest's own ``response_quantity`` rather than from an
    argument, so a layer cannot be read through the wrong channel by a caller who has not
    thought about it.
    """
    from .analyze import analyze_wedge

    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        manifest_dir = manifest_dir or path.parent
        manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest_dir is None:
        raise ValueError(
            "manifest_dir is required when the manifest is passed as a dict — the "
            "sidecars are found relative to it"
        )
    if role not in PRINT_ORDER:
        raise ValueError(f"unknown layer {role!r}; expected one of {PRINT_ORDER}")

    wedge = manifest.get("wedge")
    if not wedge:
        raise TricolourSetError(
            f"this run was generated without wedges, so {role} has nothing to read back"
        )
    sidecar = Path(manifest_dir) / wedge["sidecars"][role]
    if not sidecar.exists():
        raise TricolourSetError(
            f"{sidecar} is missing. The patch layout is shuffled by seed "
            f"{wedge['seed']}, so the scan cannot be read without it; regenerate with "
            "step_wedge(levels=%d, redundancy=%d, seed=%d)"
            % (wedge["levels"], wedge["redundancy"], wedge["seed"])
        )

    quantity = manifest["layers"][role]["response_quantity"]
    return analyze_wedge(scan_path, sidecar, space=space, quantity=quantity)


def darkroom_sheet(manifest: dict[str, Any]) -> str:
    """The wall sheet: the same run, arranged for someone standing at the enlarger.

    Deliberately a second artefact rather than a friendlier manifest. The JSON is
    optimised for reproducing a run months later; this is optimised for not making a
    mistake in the next ten minutes, and those pull in different directions.
    """
    out = [
        f"# TRICOLOUR — {manifest['set']['name']}",
        "",
        f"Negatives: `{manifest['output']['stem']}_1M` · `_2Y` · `_3C`",
        f"Sheet: {manifest['output']['page_mm'][0]:.0f} x "
        f"{manifest['output']['page_mm'][1]:.0f} mm · picture "
        f"{manifest['output']['print_size_mm'][0]:.0f} x "
        f"{manifest['output']['print_size_mm'][1]:.0f} mm · pre-shrink the paper",
        "",
        "## ORDER IS NOT NEGOTIABLE",
        "",
        "The carbonate bleach used by magenta and yellow destroys Prussian blue.",
        "Cyan is last. Always.",
        "",
    ]
    region = manifest["placement"].get("blocker_region")
    if region:
        where = (
            "centred on the sheet"
            if region["centred"]
            else f"with its top-left corner at {region['x_mm']:.0f}, {region['y_mm']:.0f} mm"
        )
        out += [
            "## COAT INSIDE THE BLOCKER",
            "",
            f"**Coat no larger than {region['w_mm']:.0f} x {region['h_mm']:.0f} mm, {where}.**",
            "",
            f"The blocker stops there. Outside it the film is clear "
            f"({region['cleared_fraction']:.0%} of the sheet), and clear film blocks",
            "nothing — sensitized paper reaching past that rectangle takes the full",
            "exposure on all three layers and comes out solid. Unsensitized paper out",
            "there is fine, which is the whole point: the ink was doing no work.",
            "",
        ]
    out += [
        "## DO NOT SKIP A SCAN",
        "",
        "Each layer's wedge is scanned after that layer dries and before the next is",
        "coated. Those scans are the measurement this whole print exists to produce.",
        "Whether a wedge can still be read after cyan is what this print is testing,",
        "so it cannot be relied on — a missed scan costs a full three-session cycle.",
        "",
    ]
    positions = slot_positions(manifest)
    for role in PRINT_ORDER:
        layer = manifest["layers"][role]
        e = layer["exposure"]
        where = f" — the **{positions[role]}** one of the three" if positions else ""
        out += [
            f"## {layer['print_order']}. {role.upper()} — `{layer['negative']}`",
            "",
            f"- **Expose {e['instruction_display']}**  ({e['instruction_seconds']} s"
            f" = {e['base_spe_seconds']} s SPE x {e['exposure_multiplier']})",
            f"- Sensitizer: {layer['sensitizer'] or '—'}",
            f"- Then: {layer['chemistry'] or '—'}",
            "- Dry flat (cold air only — heat warps the dimensional scale).",
            f"- **SCAN the {role} wedge slot now**{where}"
            + (
                "." if role == PRINT_ORDER[-1]
                else ", before coating the next layer."
            ),
            "  SilverFast raw, converted in Photoshop — the same path as the existing",
            "  calibration sheets, or the comparison means nothing.",
            "",
        ]
    if positions:
        left_to_right = sorted(positions.items(), key=lambda kv: SLOT_POSITIONS.index(kv[1]))
        out += [
            "## THE THREE WEDGE SLOTS LOOK IDENTICAL",
            "",
            "Left to right on the print: **"
            + "**, **".join(role for role, _ in left_to_right)
            + "**.",
            "",
            "They come from one seed and are pixel-identical, so position on the sheet is",
            "the only thing that says which layer a crop belongs to. Label each crop as",
            "you make it — a swapped pair gives two plausible curves and no error.",
            "",
        ]
    out += [
        "## After cyan — the experiment",
        "",
        "- Scan all three wedge slots **and** the blocked control region.",
        "- Read the control in **all three channels** — a stain invisible in L* can be",
        "  large in blue, which is the layer most at risk.",
        "",
        "The curves come from the scans taken between layers, not from these. These",
        "answer a different question: comparing them against the earlier scans, for the",
        "same physical patches, shows whether an isolated wedge survives the full",
        "remaining process. The control gives the stain floor that makes that readable —",
        "if a wedge shifted by about what the control shifted, the shift is stain rather",
        "than damage.",
        "",
        "If they do survive, later prints can drop the intermediate scans. That is a",
        "conclusion to earn from this print, not one to assume before it.",
        "",
        "## Measure the fiducial span",
        "",
        "While the sheet is on the scanner, measure the printed picture fiducials against",
        "their nominal spacing. That difference is the paper's shrinkage, and it is the",
        "number that fills each layer's `scale` field for print #2. Nothing else measures",
        "it, and it cannot be recovered later.",
        "",
    ]
    if manifest["warnings"]:
        out += ["## Warnings", ""] + [f"- {w}" for w in manifest["warnings"]] + [""]
    return "\n".join(out)
