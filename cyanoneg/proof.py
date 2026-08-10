"""Soft-proof: predict the cyanotype print from the negative.

The prediction is only as good as the measurement behind it, so this module refuses to
guess. A profile that has not been calibrated has no measured response, and a soft proof
invented from a plausible-looking curve would be worse than none — it would look
authoritative while being fiction. :func:`soft_proof` raises in that case.

Where the measurement does exist, the chain is the honest inverse of printing:

    negative → UV transmission → measured process response → print reflectance → Prussian blue

The response comes from ``profile.measurements.raw_patches``, written by ``analyze.py``
when the wedge scan was read. That is the same data the correction curve was derived
from, so the proof and the curve cannot drift apart.
"""

from __future__ import annotations

import numpy as np

from .imageio import Image, from_linear
from .lut import Lut, pchip_eval
from .profiles import Profile

#: Prussian blue endpoints in linear light: paper white → full cyanotype Dmax. These now
#: supply *hue* only — the lightness of a proofed tone comes from the profile's measured
#: patches, which know what the paper actually did.
PAPER_RGB = (0.97, 0.97, 0.94)
BLUE_RGB = (0.016, 0.055, 0.13)

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# CIE L* breakpoints, on relative luminance.
_L_KNEE_Y = 0.008856
_L_SLOPE = 903.3


class ProofUnavailable(RuntimeError):
    """Raised when a profile carries no measured response to proof against."""


def _to_lstar(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0.0, 1.0)
    return np.where(y > _L_KNEE_Y, 116.0 * np.cbrt(y) - 16.0, _L_SLOPE * y)


def _from_lstar(lstar: np.ndarray) -> np.ndarray:
    lstar = np.clip(lstar, 0.0, 100.0)
    return np.where(lstar > _L_SLOPE * _L_KNEE_Y, ((lstar + 16.0) / 116.0) ** 3, lstar / _L_SLOPE)


def measured_response(profile: Profile) -> Lut:
    """Recover the process response (input level → normalised print lightness) as a LUT.

    This is the *forward* process — the opposite direction to the correction curve.
    Built from the raw patch measurements so it reflects what the paper actually did.
    """
    patches = (profile.measurements or {}).get("raw_patches") or []
    if not patches:
        raise ProofUnavailable(
            f"profile {profile.name!r} has no measured patches — "
            "run the Calibrate wizard's wedge step first. A soft proof without "
            "measurements would be invented, not predicted."
        )

    by_value: dict[int, list[float]] = {}
    for p in patches:
        try:
            by_value.setdefault(int(p["value"]), []).append(float(p["lstar"]))
        except (KeyError, TypeError, ValueError) as e:
            raise ProofUnavailable(f"malformed patch record in {profile.name!r}: {p!r}") from e

    values = sorted(by_value)
    if len(values) < 3:
        raise ProofUnavailable(f"profile {profile.name!r} has too few distinct patches to proof")

    lstars = np.array([float(np.mean(by_value[v])) for v in values])
    # Polarity (see blocker.py): the lowest patch value is clear film → max black; the
    # highest lays maximum ink → paper white.
    black, paper = lstars[0], lstars[-1]
    if paper - black < 5.0:
        raise ProofUnavailable(
            f"profile {profile.name!r} has almost no measured tonal range — proof would be meaningless"
        )

    xs = np.array(values, dtype=np.float64) / max(values)
    ys = np.clip((lstars - black) / (paper - black), 0.0, 1.0)
    # Strictly increasing x for the interpolant; duplicates cannot occur after sorting
    # unique values, but guard anyway.
    keep = np.concatenate(([True], np.diff(xs) > 0))
    grid = np.linspace(0.0, 1.0, 256)
    return Lut(pchip_eval(xs[keep], ys[keep], grid)).enforce_monotonic()


def measured_colour(profile: Profile) -> tuple[np.ndarray, np.ndarray] | None:
    """Lightness fraction → the paper's own linear-light colour, from the wedge patches.

    ``None`` when the patches carry no colour, which is every profile measured before
    ``analyze.py`` started recording it.

    Prussian blue does not fade towards paper along a straight line between two endpoints.
    Measured on this paper, a tone at L* 50 is still about 180% as blue-minus-red as it is
    luminous, where a linear blend of the endpoints gives 30%. That is the difference
    between a proof that looks like a cyanotype and one that looks like a grey print, and
    no amount of choosing better endpoint constants fixes the shape between them. So the
    shape is measured rather than modelled — the wedge already prints 32 levels of exactly
    this, and reading its colour costs nothing.
    """
    patches = (profile.measurements or {}).get("raw_patches") or []
    with_rgb = [p for p in patches if isinstance(p.get("rgb"), (list, tuple)) and len(p["rgb"]) == 3]
    if len(with_rgb) < 3:
        return None

    by_value: dict[int, list[list[float]]] = {}
    lstar_by_value: dict[int, list[float]] = {}
    for p in with_rgb:
        by_value.setdefault(int(p["value"]), []).append([float(v) for v in p["rgb"]])
        lstar_by_value.setdefault(int(p["value"]), []).append(float(p["lstar"]))

    values = sorted(by_value)
    lstars = np.array([np.mean(lstar_by_value[v]) for v in values])
    rgbs = np.array([np.mean(by_value[v], axis=0) for v in values], dtype=np.float32)
    black, paper = lstars[0], lstars[-1]
    if paper - black < 5.0:
        return None

    fractions = np.clip((lstars - black) / (paper - black), 0.0, 1.0)
    order = np.argsort(fractions)
    fractions, rgbs = fractions[order], rgbs[order]
    keep = np.concatenate(([True], np.diff(fractions) > 1e-6))  # np.interp needs strict increase
    return fractions[keep].astype(np.float32), rgbs[keep]


def measured_endpoints(profile: Profile) -> tuple[float, float]:
    """The paper's measured (Dmax, paper-white) lightness in L*.

    :func:`measured_response` normalises these away to a 0–1 curve, which is right for the
    curve and wrong for the proof: predicting where a tone lands needs to know where the
    ends *are*. Measured Dmax on hand-coated paper runs around L* 36; the Prussian blue
    constant above is nearer 27, so proofing against the constant makes every shadow look
    deeper than the paper can actually go.
    """
    patches = (profile.measurements or {}).get("raw_patches") or []
    measured_response(profile)  # reuse its validation and error messages
    by_value: dict[int, list[float]] = {}
    for p in patches:
        by_value.setdefault(int(p["value"]), []).append(float(p["lstar"]))
    values = sorted(by_value)
    return (
        float(np.mean(by_value[values[0]])),  # clear film → maximum exposure → Dmax
        float(np.mean(by_value[values[-1]])),  # maximum ink → bare paper
    )


def soft_proof(negative: Image, profile: Profile) -> Image:
    """Render what ``negative`` should look like once printed on this paper.

    ``negative`` is the pipeline's output: colour-blocked, mirrored film. The proof
    un-mirrors it (the contact print reverses the flip), converts blocker colour to UV
    transmission, applies the measured response, and maps the result onto Prussian blue.
    """
    response = measured_response(profile)

    data = negative.data
    if data.ndim != 3:
        raise ValueError("soft_proof expects the colour-blocked RGB negative")
    print_view = data[:, ::-1]  # contact printing un-mirrors the film

    # Ink coverage on the film. The minimum channel is the best single proxy the RGB data
    # offers: clear film is white in every channel, while any blocker hue drives at least
    # one channel down. Ink equals the corrected positive level, which is what indexes the
    # measured response.
    #
    # This stays in the **encoded** working space on purpose. The response was measured
    # against wedge patch values, which are encoded code values, so converting to linear
    # light here would index the curve in the wrong space — the exact class of silent
    # error the explicit working_space parameter exists to prevent.
    ink = np.clip(1.0 - print_view.min(axis=-1), 0.0, 1.0)

    fraction = response.apply(ink.astype(np.float32))

    colour = measured_colour(profile)
    if colour is not None:
        # Best case: the paper's own colour, read off the wedge. Interpolating the measured
        # patches gives the right lightness and the right blue at once, with nothing modelled.
        fractions, rgbs = colour
        reflect = np.stack(
            [np.interp(fraction, fractions, rgbs[:, i]) for i in range(3)], axis=-1
        ).astype(np.float32)
    else:
        # Profiles measured before analyze.py recorded patch colour. Tone still comes from
        # the measurement; only the hue is approximated, and it will read too neutral in the
        # midtones. Re-run the wedge analysis to replace this with the real thing.
        #
        # The response is a fraction of the paper's measured *lightness* range, so it must be
        # interpolated in L* and only then turned back into light. Blending linear reflectance
        # with an L* fraction — which this did until two prints showed it up — puts a mid tone
        # at 50% luminance where L* says 18%, leaving the proof ~14 L* too light through the
        # midtones while matching perfectly at both ends.
        dmax_lstar, paper_lstar = measured_endpoints(profile)
        luminance = _from_lstar(dmax_lstar + fraction * (paper_lstar - dmax_lstar))
        paper = np.asarray(PAPER_RGB, dtype=np.float32)
        blue = np.asarray(BLUE_RGB, dtype=np.float32)
        hue = blue[None, None, :] + fraction[..., None] * (paper - blue)[None, None, :]
        reflect = hue * (luminance / np.maximum(hue @ _LUMA, 1e-6))[..., None]
    return Image(
        data=from_linear(np.clip(reflect, 0.0, 1.0), negative.space),
        space=negative.space,
        ppi=negative.ppi,
        bit_depth=negative.bit_depth,
    )


def can_proof(profile: Profile) -> bool:
    """True when this profile carries enough measurement to soft-proof."""
    try:
        measured_response(profile)
    except ProofUnavailable:
        return False
    return True
