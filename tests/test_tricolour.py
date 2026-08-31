"""Tricolour: three registered negatives from one RGB positive.

The failure modes guarded here are the expensive ones. A wrong channel permutation, a
wedge slot that is not actually isolated, or a page background that passes UV all produce
three plausible-looking orange transparencies, and the cost of finding out is a multi-day
darkroom cycle that cannot be re-run on the same sheet.
"""

import dataclasses
import json

import numpy as np
import pytest

from cyanoneg.imageio import DEFAULT_SPACE, Image
from pathlib import Path

from cyanoneg import pipeline
from cyanoneg.pipeline import PrintSize
from cyanoneg.profiles import PROFILE_DIR, Profile
from cyanoneg.targets import PPI, WEDGE_SEED, _mm, step_wedge
from cyanoneg.tricolour import (
    ALLOWED_TO_DIFFER,
    CHANNEL_INDEX,
    CLEAR_FILM,
    CONTROL_BORDER_MM,
    CONTROL_MM,
    FILM_FRAME,
    PRINT_FRAME,
    MIN_SEPARATION,
    SLOT_POSITIONS,
    analyze_layer_wedge,
    diagnose_layer,
    film_rect,
    slot_positions,
    to_print_orientation,
    MUST_AGREE,
    RESPONSE_QUANTITY,
    _glyph_stamp,
    calibration_fingerprint,
    control_region,
    format_seconds,
    full_blocker_value,
    layer_exposure,
    make_tricolour,
    output_name,
    seed_provisional_set,
    step_frame,
    tricolour_page,
    PRINT_ORDER,
    SOURCE_CHANNEL,
    TricolourLayer,
    TricolourSet,
    TricolourSetError,
    check_profile_agreement,
    extract_channel,
    step_saturate,
)


@pytest.fixture
def rgb() -> Image:
    """Three channels holding distinguishable, non-constant data.

    Deliberately not a grey ramp: if all three channels carried the same values, every
    channel permutation would pass.
    """
    h, w = 8, 16
    r = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    g = np.tile(np.linspace(0.25, 0.75, w, dtype=np.float32), (h, 1))
    b = np.tile(np.linspace(1.0, 0.0, w, dtype=np.float32), (h, 1))
    return Image(np.stack([r, g, b], axis=-1), "srgb", ppi=300)


class TestExtractChannel:
    @pytest.mark.parametrize(
        ("layer", "index"),
        [("cyan", 0), ("magenta", 1), ("yellow", 2)],
    )
    def test_extracts_the_layer_s_own_channel_byte_exact(self, rgb, layer, index):
        """Red→cyan, green→magenta, blue→yellow, and no arithmetic on the way through.

        Byte-exact against the raw slice because anything that merely correlates — a
        weighted mix, a round trip through linear light — would still look like a
        reasonable negative while being the wrong tonal scale.
        """
        result = extract_channel(rgb, layer)

        assert np.array_equal(result.data, rgb.data[..., index])
        assert CHANNEL_INDEX[layer] == index

    def test_mono_source_raises_rather_than_tripling(self, rgb):
        """A greyscale positive has no colour to separate.

        ``to_mono`` early-returns on mono input, so routing through it would have emitted
        three *identical* negatives and three darkroom sessions would have produced a grey
        image in three colours. Refusing is the only safe answer.
        """
        mono = rgb.replace(rgb.data[..., 0])
        assert mono.is_mono

        with pytest.raises(ValueError, match="mono"):
            extract_channel(mono, "magenta")


class TestSaturate:
    def test_amount_one_is_exactly_the_input(self, rgb):
        """No boost must mean no change at all, not a float32 round trip.

        Scaling happens in linear light, so a naive implementation would decode and
        re-encode even at 1.0 and shift low code values by a bit or two. That would put a
        silent difference between "tricolour at boost 1.0" and the mono path, which is the
        baseline the whole calibration is compared against.
        """
        result, clipped = step_saturate(rgb, 1.0)

        assert np.array_equal(result.data, rgb.data)
        assert clipped == 0.0

    def test_boost_widens_colour_and_leaves_grey_alone(self):
        """Saturation scales distance from the Rec.709 grey axis; grey has no distance.

        Asserted as two independent properties rather than against expected pixel values:
        a coloured pixel must move *away* from its own luma, and a neutral one must not
        move at all. An implementation that scaled about 0.5, or about the channel mean,
        would satisfy neither.
        """
        pixels = np.array([[[0.5, 0.25, 0.25], [0.4, 0.4, 0.4]]], dtype=np.float32)
        image = Image(pixels, "linear")

        result, _ = step_saturate(image, 1.5)

        coloured_before = pixels[0, 0]
        coloured_after = result.data[0, 0]
        assert np.ptp(coloured_after) > np.ptp(coloured_before)

        assert np.allclose(result.data[0, 1], pixels[0, 1], atol=1e-6)

    def test_clipped_fraction_counts_pixels_with_any_component_out_of_range(self):
        """Pinned to a definition, because a threshold on a vague number is not a warning.

        The definition is: the fraction of *pixels* for which at least one component fell
        outside [0, 1] after scaling and before clipping. Four pixels are constructed so
        the answer is known by hand — one grey (safe), one over 1.0 in red, one safely
        inside, one below 0.0 in red — so the expected value is exactly 0.5. Counting
        components rather than pixels, or counting after the clip, both give other answers.
        """
        pixels = np.array(
            [[
                [0.4, 0.4, 0.4],    # neutral: unmoved, in range
                [1.0, 0.0, 0.0],    # red goes to ~1.79, over
                [0.5, 0.5, 0.45],   # stays inside
                [0.0, 0.0, 0.1],    # red goes to ~-0.007, under
            ]],
            dtype=np.float32,
        )
        image = Image(pixels, "linear")

        _, clipped = step_saturate(image, 2.0)

        assert clipped == pytest.approx(0.5)


def _set(**overrides) -> TricolourSet:
    """A set whose three layers differ only in what genuinely varies between them."""
    defaults = dict(
        name="CassArt 300 Sm — Tricolour",
        saturation_boost=1.35,
        border_mm=10.0,
        layers={
            "magenta": TricolourLayer(profile="P — Magenta", exposure_multiplier=1.5),
            "yellow": TricolourLayer(profile="P — Yellow", exposure_multiplier=2.75),
            "cyan": TricolourLayer(profile="P — Cyan", exposure_multiplier=1.1),
        },
    )
    defaults.update(overrides)
    return TricolourSet(**defaults)


class TestProcessConstants:
    def test_print_order_is_magenta_yellow_cyan(self):
        """Cyan must be last: the carbonate bleach used for M and Y destroys Prussian blue.

        Pinned as a literal rather than derived from anything, because every other order
        produces a print and only this one produces the right print. Getting it wrong is
        found out three darkroom days later, on a sheet that cannot be reused.
        """
        assert PRINT_ORDER == ("magenta", "yellow", "cyan")

    def test_source_channel_names_agree_with_channel_index(self):
        """One mapping, two spellings — they must not be able to drift apart.

        ``CHANNEL_INDEX`` is what the array slicing uses; ``SOURCE_CHANNEL`` is what the
        set file and manifest say in words. If a future edit changed one and not the other,
        the negatives and the paperwork describing them would disagree silently.
        """
        names = ("red", "green", "blue")
        assert SOURCE_CHANNEL == {
            role: names[index] for role, index in CHANNEL_INDEX.items()
        }


class TestSetSerialisation:
    def test_round_trip_through_json_preserves_the_set(self, tmp_path):
        """Save then load must be an identity, or a set means something different tomorrow."""
        tset = _set()
        path = tset.save(tmp_path / "set.json")

        assert TricolourSet.load(path) == tset

    def test_saved_json_states_the_derived_invariants_for_a_human_reader(self, tmp_path):
        """Channel and order are written out even though they are derived.

        The file is read by a person standing at an enlarger deciding which sheet to expose
        next, and "print_order: 1" is the answer to their question. Derived-and-written is
        safe here only because loading re-checks it against the constants.
        """
        d = json.loads(_set().save(tmp_path / "set.json").read_text(encoding="utf-8"))

        assert d["layers"]["magenta"]["print_order"] == 1
        assert d["layers"]["magenta"]["source_channel"] == "green"
        assert d["layers"]["cyan"]["print_order"] == 3
        assert d["layers"]["cyan"]["source_channel"] == "red"

    def test_loading_rejects_a_file_that_contradicts_the_constants(self, tmp_path):
        """A hand-edited set claiming magenta comes from red must not load.

        This is the whole reason the derived fields are re-checked instead of ignored: a
        plausible-looking edit would otherwise be silently overridden by the constants, and
        the operator would trust a file that does not describe what the code does.
        """
        d = json.loads(_set().save(tmp_path / "set.json").read_text(encoding="utf-8"))
        d["layers"]["magenta"]["source_channel"] = "red"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(d), encoding="utf-8")

        with pytest.raises(ValueError, match="source_channel"):
            TricolourSet.load(path)

    def test_missing_layer_is_rejected(self):
        """Two negatives are not a tricolour set."""
        layers = dict(_set().layers)
        del layers["yellow"]

        problems = _set(layers=layers).validate()

        assert any("yellow" in p for p in problems)


def _layer_profile(name: str, **overrides) -> Profile:
    """A profile as one layer of a set: shared apparatus, its own name and chemistry."""
    defaults = dict(
        name=name,
        paper="CassArt 300 Sm",
        film="Fixxons",
        film_batch="2026-07",
        media_type="Premium Presentation Paper Matte",
        working_space="srgb",
        driver_settings={"quality": "high", "colour": "off"},
        blocker={"model": "fixed_hue", "rgb": [255, 64, 0], "saturation": 1.0},
        exposure={"spe_seconds": 810},
    )
    defaults.update(overrides)
    return Profile(**defaults)


def _layer_profiles(**overrides) -> dict[str, Profile]:
    profiles = {
        "magenta": _layer_profile("P — Magenta", chemistry="carbonate bleach, madder root"),
        "yellow": _layer_profile("P — Yellow", chemistry="carbonate bleach to Fe(III)"),
        "cyan": _layer_profile("P — Cyan", chemistry="classic, untoned"),
    }
    profiles.update(overrides)
    return profiles


class TestProfileAgreement:
    def test_layers_sharing_apparatus_agree(self):
        assert check_profile_agreement(_layer_profiles()) == []

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("film_batch", "2026-08"),
            ("driver_settings", {"quality": "draft"}),
            ("blocker", {"model": "fixed_hue", "rgb": [255, 80, 0], "saturation": 1.0}),
            ("paper", "Somerset Satin"),
            ("working_space", "adobergb"),
            ("exposure", {"spe_seconds": 600}),
        ],
    )
    def test_one_layer_out_of_step_is_rejected(self, field_name, value):
        """Each of these is a different print masquerading as one layer of this print.

        ``film_batch`` and ``driver_settings`` are the silent-error class: the calibration
        absorbed them, so a layer measured under different ones carries a curve that does
        not describe the sheet being printed. ``blocker`` matters for a second reason — the
        page background and the non-owner wedge masks are one colour, so three layers with
        different blockers cannot share a page at all. ``exposure`` must agree because SPE
        is the common base that the per-layer multiplier is applied to.
        """
        profiles = _layer_profiles(
            yellow=_layer_profile("P — Yellow", **{field_name: value})
        )

        problems = check_profile_agreement(profiles)

        assert any(field_name in p for p in problems), problems

    def test_layers_may_differ_in_name_and_chemistry(self):
        """Each layer is its own profile with its own toning, and its own measured curve."""
        assert "name" in ALLOWED_TO_DIFFER
        assert "chemistry" in ALLOWED_TO_DIFFER
        assert "lut" in ALLOWED_TO_DIFFER

    def test_every_profile_field_is_either_checked_or_deliberately_exempt(self):
        """The check must fail safe as the profile schema grows.

        A handwritten list of fields-that-must-agree goes stale silently: someone adds
        ``uv_source`` to ``Profile``, nobody updates the list, and sets with mismatched UV
        sources start validating. Inverting it — every field must agree unless named in
        ``ALLOWED_TO_DIFFER`` — makes the stale case a loud test failure here instead.
        """
        declared = {f.name for f in dataclasses.fields(Profile)}

        assert ALLOWED_TO_DIFFER <= declared
        assert declared - ALLOWED_TO_DIFFER == set(MUST_AGREE)


class TestResolve:
    def test_resolve_loads_each_layer_s_profile_by_name(self, tmp_path):
        for profile in _layer_profiles().values():
            profile.save(tmp_path / f"{profile.name}.json")

        resolved = _set().resolve(tmp_path)

        assert set(resolved) == set(PRINT_ORDER)
        assert resolved["magenta"].name == "P — Magenta"

    def test_resolve_reports_the_missing_profile_by_name(self, tmp_path):
        profiles = _layer_profiles()
        del profiles["cyan"]
        for profile in profiles.values():
            profile.save(tmp_path / f"{profile.name}.json")

        with pytest.raises(TricolourSetError, match="P — Cyan"):
            _set().resolve(tmp_path)

    def test_resolve_refuses_layers_that_do_not_share_apparatus(self, tmp_path):
        """Resolution is where the disagreement becomes findable, so it must refuse there."""
        profiles = _layer_profiles(
            cyan=_layer_profile("P — Cyan", film_batch="2026-08")
        )
        for profile in profiles.values():
            profile.save(tmp_path / f"{profile.name}.json")

        with pytest.raises(TricolourSetError, match="film_batch"):
            _set().resolve(tmp_path)


class TestResponseQuantity:
    def test_each_layer_is_read_through_its_complement(self):
        """Magenta absorbs green, yellow absorbs blue, cyan absorbs red.

        Reading a yellow wedge in L* is the failure this constant exists to prevent: L*
        barely moves across yellow's whole density range, so the derived curve would be
        shaped by noise while looking like a measurement.
        """
        assert RESPONSE_QUANTITY == {
            "cyan": "lstar_r",
            "magenta": "lstar_g",
            "yellow": "lstar_b",
        }

    def test_every_layer_has_one(self):
        assert set(RESPONSE_QUANTITY) == set(PRINT_ORDER)


class TestLayerExposure:
    def test_worked_values_for_the_measured_paper(self):
        profile = _layer_profile("P", exposure={"spe_seconds": 810})
        got = {
            role: layer_exposure(profile, _set().layers[role])["instruction_seconds"]
            for role in PRINT_ORDER
        }
        assert got == {"magenta": 1215, "yellow": 2228, "cyan": 891}

    def test_display_matches_the_recorded_spe(self):
        """810 s renders as 13:30, which is how HANDOFF records the measured SPE.

        A self-check on the formatter: if mm:ss were wrong, the one exposure we have an
        independently recorded rendering of would disagree.
        """
        assert format_seconds(810) == "13:30"
        assert format_seconds(1215) == "20:15"
        assert format_seconds(2228) == "37:08"
        assert format_seconds(891) == "14:51"

    def test_half_second_rounds_up(self):
        """810 x 2.75 is 2227.5 exactly; the timer takes whole seconds."""
        block = layer_exposure(
            _layer_profile("P"), TricolourLayer(profile="P", exposure_multiplier=2.75)
        )
        assert block["computed_seconds"] == 2227.5
        assert block["instruction_seconds"] == 2228

    def test_float_noise_is_kept_out_of_the_record(self):
        """810 x 1.1 is 891.0000000000001 in float64; a manifest a person diffs should not say so."""
        block = layer_exposure(
            _layer_profile("P"), TricolourLayer(profile="P", exposure_multiplier=1.1)
        )
        assert block["computed_seconds"] == 891.0
        assert repr(block["computed_seconds"]) == "891.0"

    def test_base_spe_is_reported_unmultiplied(self):
        """The profile's own value travels with the result, so double application is visible."""
        block = layer_exposure(
            _layer_profile("P"), TricolourLayer(profile="P", exposure_multiplier=2.75)
        )
        assert block["base_spe_seconds"] == 810
        assert block["exposure_multiplier"] == 2.75

    def test_missing_spe_is_refused_by_name(self):
        with pytest.raises(TricolourSetError, match="spe_seconds"):
            layer_exposure(
                _layer_profile("P", exposure={}),
                TricolourLayer(profile="P", exposure_multiplier=1.5),
            )

    def test_nonsense_spe_is_refused(self):
        with pytest.raises(TricolourSetError):
            layer_exposure(
                _layer_profile("P", exposure={"spe_seconds": 0}),
                TricolourLayer(profile="P", exposure_multiplier=1.5),
            )

    def test_multiplier_is_never_stored_in_a_profile(self):
        """Single source of truth: the profile keeps the base, the set keeps the multiplier.

        Guards the future — a later refactor that helpfully caches the working exposure on
        the profile would make the multiplier applicable twice.
        """
        profile = _layer_profile("P", exposure={"spe_seconds": 810})
        layer_exposure(profile, _set().layers["yellow"])
        assert profile.exposure == {"spe_seconds": 810}


class TestLayerScale:
    def test_defaults_to_one(self):
        assert all(layer.scale == 1.0 for layer in _set().layers.values())
        assert _set().validate() == []

    @pytest.mark.parametrize("bad", [0.5, 1.5, 0.0, -1.0])
    def test_rejects_implausible_values(self, bad):
        """Cotton rag moves 0.5-1%. Anything beyond that is a typo, not a measurement."""
        tset = _set()
        tset.layers["yellow"] = dataclasses.replace(tset.layers["yellow"], scale=bad)
        problems = tset.validate()
        assert any("scale" in p for p in problems)

    @pytest.mark.parametrize("good", [0.99, 1.0, 1.01])
    def test_accepts_real_shrinkage(self, good):
        tset = _set()
        tset.layers["yellow"] = dataclasses.replace(tset.layers["yellow"], scale=good)
        assert tset.validate() == []

    def test_survives_the_set_round_trip(self, tmp_path):
        tset = _set()
        tset.layers["magenta"] = dataclasses.replace(tset.layers["magenta"], scale=0.995)
        out = tset.save(tmp_path / "set.json")
        assert TricolourSet.load(out).layers["magenta"].scale == 0.995


# --------------------------------------------------------------------------- frame

BLOCKER_RGB = (255, 64, 0)
SATURATION = 1.0


class TestFrame:
    def test_border_is_added_on_every_side(self):
        framed, _ = step_frame(
            _picture(130, 100), "magenta", _mm(10), BLOCKER_RGB, SATURATION
        )
        h, w = framed.data.shape[:2]
        assert (round(w / PPI * 25.4), round(h / PPI * 25.4)) == (150, 120)

    def test_fiducials_are_identical_across_layers(self):
        """Layer 1 prints these onto the paper; 2 and 3 align to them. If they moved
        between layers, a perfectly registered sheet would still show a fringe."""
        geometries = [
            step_frame(_picture(), role, _mm(10), BLOCKER_RGB, SATURATION)[1]["fiducials"]
            for role in PRINT_ORDER
        ]
        assert all(g == geometries[0] for g in geometries)

    def test_glyph_is_not_clear_film(self):
        """R6: a clear-film glyph prints as dark as a fiducial."""
        framed, geo = step_frame(_picture(), "yellow", _mm(10), BLOCKER_RGB, SATURATION)
        g = geo["glyph"]
        region = framed.data[
            g["y_px"] : g["y_px"] + g["h_px"], g["x_px"] : g["x_px"] + g["w_px"]
        ]
        assert not np.isclose(region, 1.0).all(axis=-1).any(), "glyph reaches clear film"

    def test_glyph_sits_inside_the_upper_fiducial_span(self):
        """The structural guarantee, stronger than the coverage one.

        ``detect_fiducials`` takes the bounding box of all candidate centres and picks the
        blob nearest each corner. A mark between the two upper fiducials cannot move that
        box, so even a detected glyph could not redefine the frame.
        """
        _, geo = step_frame(_picture(), "cyan", _mm(10), BLOCKER_RGB, SATURATION)
        f, g = geo["fiducials"], geo["glyph"]
        left = f["top_left"]["x_px"] + f["top_left"]["size_px"]
        right = f["top_right"]["x_px"]
        assert left < g["x_px"]
        assert g["x_px"] + g["w_px"] < right

    def test_glyph_reads_correctly_on_the_film(self):
        """Drawn mirrored, because the page is flipped once on the way to film.

        Checked on 'C', whose spine is on the left: after the flip the ink mass must lie
        left of centre. A glyph drawn the right way round here would fail this.
        """
        blocked = full_blocker_value(BLOCKER_RGB, SATURATION)
        stamp = _glyph_stamp("C", _mm(10), blocked)
        ink = ~np.isclose(stamp, blocked, atol=1e-3).all(axis=-1)
        assert not np.array_equal(ink, ink[:, ::-1]), "a symmetric stamp proves nothing"
        film = ink[:, ::-1]
        half = film.shape[1] // 2
        assert film[:, :half].sum() > film[:, half:].sum()

    def test_glyph_can_be_switched_off(self):
        _, geo = step_frame(
            _picture(), "magenta", _mm(10), BLOCKER_RGB, SATURATION, glyph=False
        )
        assert geo["glyph"] is None

    def test_unknown_layer_is_refused(self):
        with pytest.raises(ValueError, match="unknown layer"):
            step_frame(_picture(), "green", _mm(10), BLOCKER_RGB, SATURATION)


# --------------------------------------------------------------------------- page


def _wedges() -> dict:
    """One wedge per layer, identical geometry — 16 x k4, the settled print #1 target."""
    return {
        role: step_wedge(BLOCKER_RGB, saturation=SATURATION, levels=16, redundancy=4)
        for role in PRINT_ORDER
    }


def _picture(w_mm: float = 150.0, h_mm: float = 120.0) -> Image:
    data = np.full((_mm(h_mm), _mm(w_mm), 3), 0.5, dtype=np.float32)
    return Image(data, DEFAULT_SPACE, ppi=PPI)


class TestPageBackground:
    def test_background_is_full_blocker_not_clear_film(self):
        """Clear film passes UV. calibration_page uses white because its margins are never
        coated; here the same choice would give every uncoated margin a full exposure on
        three successive layers, and take anything coated outside the picture to Dmax."""
        page, _ = tricolour_page(_picture(), _wedges(), "magenta", BLOCKER_RGB, SATURATION)
        blocked = full_blocker_value(BLOCKER_RGB, SATURATION)
        for corner in (page.data[0, 0], page.data[0, -1], page.data[-1, 0], page.data[-1, -1]):
            assert np.allclose(corner, blocked)
        assert not np.allclose(page.data[0, 0], 1.0), "clear film would expose the margin"

    def test_page_is_a4_at_printer_resolution(self):
        page, _ = tricolour_page(_picture(), _wedges(), "cyan", BLOCKER_RGB, SATURATION)
        h, w = page.data.shape[:2]
        assert (round(w / PPI * 25.4), round(h / PPI * 25.4)) == (210, 297)


class TestOrientationFrames:
    """``placement`` is print orientation; the TIFFs are mirrored. Crossing the two is
    the quietest mistake in this module — on a centred layout a rectangle cropped in the
    wrong frame lands within a millimetre of the right place and looks entirely correct."""

    def test_mirroring_twice_is_identity(self):
        page, placement = tricolour_page(
            _picture(), _wedges(), "magenta", BLOCKER_RGB, SATURATION
        )
        w = page.data.shape[1]
        rect = placement["picture"]
        there = film_rect(rect, w)
        back = film_rect(there, w)
        assert back["x_px"] == rect["x_px"]
        assert (back["w_px"], back["h_px"], back["y_px"]) == (
            rect["w_px"], rect["h_px"], rect["y_px"]
        )

    def test_only_x_moves(self):
        """y, w and h are identical in both frames, which is why the error hides."""
        page, placement = tricolour_page(
            _picture(), _wedges(), "cyan", BLOCKER_RGB, SATURATION
        )
        rect = placement["control"]
        moved = film_rect(rect, page.data.shape[1])
        assert moved["y_px"] == rect["y_px"]
        assert (moved["w_px"], moved["h_px"]) == (rect["w_px"], rect["h_px"])
        assert moved["frame"] == FILM_FRAME

    def test_the_rectangle_actually_finds_the_element_on_film(self):
        """The property that matters: crop the exported film with the converted rectangle
        and, once un-mirrored, get exactly what the same rectangle cuts from the page."""
        page, placement = tricolour_page(
            _picture(), _wedges(), "yellow", BLOCKER_RGB, SATURATION
        )
        film = pipeline.step_flip(page)
        w = page.data.shape[1]

        for rect in (placement["picture"], placement["control"],
                     placement["wedges"]["yellow"]):
            want = page.data[
                rect["y_px"] : rect["y_px"] + rect["h_px"],
                rect["x_px"] : rect["x_px"] + rect["w_px"],
            ]
            c = film_rect(rect, w)
            got = to_print_orientation(
                film.data[c["y_px"] : c["y_px"] + c["h_px"], c["x_px"] : c["x_px"] + c["w_px"]]
            )
            assert np.array_equal(got, want)

    def test_cropping_in_the_wrong_frame_is_detectably_wrong(self):
        """Guards the guard. If a mirrored crop happened to equal the print-frame crop,
        every test above would pass while proving nothing — so assert the two differ."""
        page, placement = tricolour_page(
            _picture(), _wedges(), "magenta", BLOCKER_RGB, SATURATION
        )
        film = pipeline.step_flip(page)
        rect = placement["wedges"]["magenta"]
        naive = film.data[
            rect["y_px"] : rect["y_px"] + rect["h_px"],
            rect["x_px"] : rect["x_px"] + rect["w_px"],
        ]
        want = page.data[
            rect["y_px"] : rect["y_px"] + rect["h_px"],
            rect["x_px"] : rect["x_px"] + rect["w_px"],
        ]
        assert not np.array_equal(naive, want), (
            "the wrong-frame crop matches the right one, so these tests prove nothing "
            "about this layout"
        )

    def test_a_rectangle_that_cannot_fit_is_refused(self):
        with pytest.raises(ValueError, match="does not fit"):
            film_rect({"x_px": 900, "y_px": 0, "w_px": 200, "h_px": 10}, 1000)

    def test_manifest_states_its_frame(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="FR", profile_dir=pdir, wedges=False,
        )
        placement = result.manifest["placement"]
        assert placement["frame"] == PRINT_FRAME
        assert "film_rect" in placement["frame_note"]


class TestBlockerExtent:
    """Bounding the flood trades ink for a promise that no coated paper lies outside it."""

    EXTENT = 5.0

    def _page(self, owner: str, extent):
        return tricolour_page(
            _picture(), _wedges(), owner, BLOCKER_RGB, SATURATION,
            blocker_extent_mm=extent,
        )

    def test_default_floods_the_sheet_unchanged(self):
        """``None`` must leave the page exactly as it was before this option existed."""
        page, placement = self._page("magenta", None)
        blocked = full_blocker_value(BLOCKER_RGB, SATURATION)
        assert "blocker_region" not in placement
        for corner in (page.data[0, 0], page.data[0, -1], page.data[-1, 0], page.data[-1, -1]):
            assert np.allclose(corner, blocked)

    def test_region_is_identical_on_all_three_sheets(self):
        """The one mistake here that destroys a print rather than degrading it.

        The bound is taken over every element, including the two wedge slots a sheet does
        not own. Bounding to the owner's slot alone would save more ink and would leave
        paper that is blocked under magenta sitting under clear film on yellow — a full
        unfiltered 2228 s exposure straight onto a coated sheet, which is precisely what
        the blocked background exists to prevent. Compared as pixels, not just as records,
        because the records agreeing would not prove the canvases do.

        Comparing "every clear-film pixel" across the three sheets is the obvious check
        and it is wrong: a wedge's own fiducials and its n = 0 patch are clear film too,
        and by R2 they appear only on the sheet that owns that slot. The three masks
        therefore differ *inside* the slots by design. What has to match is the film
        outside the blocked region, which is what bounding actually creates.
        """
        pages, regions = {}, {}
        for role in PRINT_ORDER:
            page, placement = self._page(role, self.EXTENT)
            pages[role] = page.data
            regions[role] = placement["blocker_region"]

        first = regions[PRINT_ORDER[0]]
        for role in PRINT_ORDER[1:]:
            assert regions[role] == first, f"{role} bounded the blocker differently"

        h, w = pages[PRINT_ORDER[0]].shape[:2]
        outside = np.ones((h, w), dtype=bool)
        outside[
            first["y_px"] : first["y_px"] + first["h_px"],
            first["x_px"] : first["x_px"] + first["w_px"],
        ] = False
        for role in PRINT_ORDER:
            assert np.all(pages[role][outside] == CLEAR_FILM), (
                f"{role} leaves blocker outside the shared region — the three sheets "
                "would protect different paper"
            )

    def test_nothing_inside_the_region_changes(self):
        """Bounding may only remove background. Everything inside must be byte-identical."""
        flooded, _ = self._page("yellow", None)
        bounded, placement = self._page("yellow", self.EXTENT)
        r = placement["blocker_region"]
        sl = (slice(r["y_px"], r["y_px"] + r["h_px"]), slice(r["x_px"], r["x_px"] + r["w_px"]))
        assert np.array_equal(bounded.data[sl], flooded.data[sl])

    def test_outside_is_clear_film_and_the_elements_are_not(self):
        page, placement = self._page("cyan", self.EXTENT)
        r = placement["blocker_region"]
        assert np.allclose(page.data[0, 0], CLEAR_FILM)
        assert np.allclose(page.data[-1, -1], CLEAR_FILM)
        for name in ("picture", "control"):
            e = placement[name]
            assert e["x_px"] >= r["x_px"] and e["y_px"] >= r["y_px"]
            assert e["x_px"] + e["w_px"] <= r["x_px"] + r["w_px"]
            assert e["y_px"] + e["h_px"] <= r["y_px"] + r["h_px"]
        for slot in placement["wedges"].values():
            assert slot["x_px"] >= r["x_px"]
            assert slot["x_px"] + slot["w_px"] <= r["x_px"] + r["w_px"]

    def test_zero_extent_still_encloses_every_element(self):
        """The boundary case: no margin at all must still not clip an element."""
        _, placement = self._page("magenta", 0.0)
        r = placement["blocker_region"]
        for slot in placement["wedges"].values():
            assert slot["x_px"] >= r["x_px"]
            assert slot["x_px"] + slot["w_px"] <= r["x_px"] + r["w_px"]

    def test_extent_larger_than_the_page_is_the_flood_again(self):
        page, placement = self._page("magenta", 500.0)
        blocked = full_blocker_value(BLOCKER_RGB, SATURATION)
        assert placement["blocker_region"]["cleared_fraction"] == 0.0
        assert np.allclose(page.data[0, 0], blocked)

    def test_negative_extent_refused(self):
        with pytest.raises(ValueError, match="blocker_extent_mm must be >= 0"):
            self._page("magenta", -1.0)

    def test_centred_is_measured_not_assumed(self):
        """The wall sheet says "centred on the sheet"; that has to be true, not asserted."""
        _, placement = self._page("magenta", self.EXTENT)
        r = placement["blocker_region"]
        assert r["centred"] is True
        assert max(r["offset_from_centre_mm"]) < 0.5


class TestWedgeIsolation:
    """R2: each layer's wedge measures that layer alone.

    A stacked patch read through green carries magenta's absorption plus yellow's plus
    cyan's, so a curve derived from it linearises the neutral stack. Three such curves are
    one curve read three ways.
    """

    @pytest.mark.parametrize("owner", PRINT_ORDER)
    def test_only_the_owner_draws_its_wedge(self, owner):
        page, place = tricolour_page(_picture(), _wedges(), owner, BLOCKER_RGB, SATURATION)
        blocked = full_blocker_value(BLOCKER_RGB, SATURATION)
        for role, slot in place["wedges"].items():
            region = page.data[
                slot["y_px"] : slot["y_px"] + slot["h_px"],
                slot["x_px"] : slot["x_px"] + slot["w_px"],
            ]
            if role == owner:
                assert slot["owned"]
                assert not np.allclose(region, blocked), "the owner's wedge is missing"
            else:
                assert not slot["owned"]
                assert np.allclose(region, blocked), f"{role} slot is not isolated"

    def test_slot_geometry_is_identical_across_layers(self):
        """The three sheets must register. Differing slots would misalign the whole page."""
        places = [
            tricolour_page(_picture(), _wedges(), owner, BLOCKER_RGB, SATURATION)[1]
            for owner in PRINT_ORDER
        ]
        for role in PRINT_ORDER:
            rects = {
                (p["wedges"][role]["x_px"], p["wedges"][role]["y_px"],
                 p["wedges"][role]["w_px"], p["wedges"][role]["h_px"])
                for p in places
            }
            assert len(rects) == 1, f"{role} slot moves between layers"

    def test_mismatched_wedges_are_refused(self):
        wedges = _wedges()
        wedges["yellow"] = step_wedge(BLOCKER_RGB, levels=32, redundancy=4)
        with pytest.raises(ValueError, match="differ in size"):
            tricolour_page(_picture(), wedges, "magenta", BLOCKER_RGB, SATURATION)

    def test_partial_wedge_set_is_refused(self):
        wedges = _wedges()
        del wedges["cyan"]
        with pytest.raises(ValueError, match="every layer"):
            tricolour_page(_picture(), wedges, "magenta", BLOCKER_RGB, SATURATION)


class TestControlRegion:
    """P1: one region, blocked on every layer, answering three questions at once."""

    def test_is_forty_millimetres_with_a_uniform_interior(self):
        control, _ = control_region(BLOCKER_RGB, SATURATION)
        h, w = control.shape[:2]
        assert (round(w / PPI * 25.4), round(h / PPI * 25.4)) == (CONTROL_MM, CONTROL_MM)
        b = _mm(CONTROL_BORDER_MM)
        blocked = full_blocker_value(BLOCKER_RGB, SATURATION)
        assert np.allclose(control[b:-b, b:-b], blocked)

    def test_carries_four_fiducials_one_hollow(self):
        """It has to be locatable and croppable, like a wedge."""
        _, fiducials = control_region(BLOCKER_RGB, SATURATION)
        assert set(fiducials) == {"top_left", "top_right", "bottom_left", "bottom_right"}
        assert [k for k, v in fiducials.items() if v["hollow"]] == ["top_left"]

    @pytest.mark.parametrize("owner", PRINT_ORDER)
    def test_no_layer_ever_exposes_it(self, owner):
        """Not just the owning layer — nothing may print there, on any sheet."""
        page, place = tricolour_page(_picture(), _wedges(), owner, BLOCKER_RGB, SATURATION)
        c = place["control"]
        b = _mm(CONTROL_BORDER_MM)
        interior = page.data[
            c["y_px"] + b : c["y_px"] + c["h_px"] - b,
            c["x_px"] + b : c["x_px"] + c["w_px"] - b,
        ]
        assert np.allclose(interior, full_blocker_value(BLOCKER_RGB, SATURATION))

    def test_placement_is_recorded_and_stable(self):
        rects = set()
        for owner in PRINT_ORDER:
            _, place = tricolour_page(_picture(), _wedges(), owner, BLOCKER_RGB, SATURATION)
            c = place["control"]
            rects.add((c["x_px"], c["y_px"], c["w_px"], c["h_px"]))
            assert c["read_in"] == "all three channels"
        assert len(rects) == 1


class TestPageScale:
    def test_scale_moves_the_picture_and_leaves_the_furniture(self):
        """P2: the wedges absorb a uniform scale change through their own homographies."""
        _, base = tricolour_page(_picture(), _wedges(), "magenta", BLOCKER_RGB, SATURATION)
        _, shrunk = tricolour_page(
            _picture(), _wedges(), "magenta", BLOCKER_RGB, SATURATION, scale=0.99
        )
        assert shrunk["picture"]["w_px"] < base["picture"]["w_px"]
        assert shrunk["picture"]["scale"] == 0.99
        for role in PRINT_ORDER:
            assert shrunk["wedges"][role]["w_px"] == base["wedges"][role]["w_px"]
            assert shrunk["wedges"][role]["h_px"] == base["wedges"][role]["h_px"]
        assert shrunk["control"]["w_px"] == base["control"]["w_px"]

    def test_scale_of_one_is_a_true_no_op(self):
        page_a, _ = tricolour_page(_picture(), _wedges(), "cyan", BLOCKER_RGB, SATURATION)
        page_b, _ = tricolour_page(
            _picture(), _wedges(), "cyan", BLOCKER_RGB, SATURATION, scale=1.0
        )
        assert np.array_equal(page_a.data, page_b.data)


class TestPageFit:
    def test_settled_layout_fits_a4_with_room(self):
        """picture 120 + gap 20 + wedges 60 + gap 10 + control 40 = 250 of 297.

        This is the *worst* case: a picture that exactly fills its 100 mm box. A real
        source fits within the print size rather than filling it, so the block is usually
        shorter — HPTressII.jpg composes to 238 mm.
        """
        _, place = tricolour_page(_picture(), _wedges(), "magenta", BLOCKER_RGB, SATURATION)
        top = place["picture"]["y_px"]
        bottom = place["control"]["y_px"] + place["control"]["h_px"]
        assert round((bottom - top) / PPI * 25.4) == 250
        assert top / PPI * 25.4 == pytest.approx(23.5, abs=1)

    def test_control_furniture_clears_the_wedge_furniture(self):
        """Two targets' fiducials must not sit close enough to fall inside one crop.

        The manifest crops are exact, but a skewed scan needs margin. Both sets sit ~2 mm
        inside their own borders, so the clearance is the gap plus ~4 mm.
        """
        _, place = tricolour_page(_picture(), _wedges(), "magenta", BLOCKER_RGB, SATURATION)
        wedge = place["wedges"]["magenta"]
        gap_mm = (
            place["control"]["y_px"] - (wedge["y_px"] + wedge["h_px"])
        ) / PPI * 25.4
        assert gap_mm == pytest.approx(10.0, abs=0.2)

    def test_oversized_picture_raises_rather_than_crops(self):
        with pytest.raises(ValueError, match="but the page is"):
            tricolour_page(_picture(200, 260), _wedges(), "magenta", BLOCKER_RGB, SATURATION)

    def test_unknown_owner_is_refused(self):
        with pytest.raises(ValueError, match="unknown owner"):
            tricolour_page(_picture(), _wedges(), "green", BLOCKER_RGB, SATURATION)


# --------------------------------------------------------------------------- run


def _measured_base() -> Profile:
    return Profile.load(PROFILE_DIR / "CassArt 300 Sm.json")


def _seeded(tmp_path) -> tuple[TricolourSet, Path]:
    """A provisional set on disk, seeded from the shipped measured profile."""
    pdir = tmp_path / "profiles"
    pdir.mkdir(exist_ok=True)
    tset, clones = seed_provisional_set(_measured_base(), "CassArt 300 Sm — Tricolour")
    for clone in clones:
        clone.save(pdir / f"{clone.name}.json")
    return tset, pdir


def _independent_rgb(w: int = 400, h: int = 300) -> Image:
    """Three channels that are genuinely independent.

    Not gradients: a diagonal is a linear combination of a horizontal and a vertical one,
    and an anti-correlated channel gives the *same* |correlation| as the channel it mirrors
    — so a wrong permutation would score identically. Learned the hard way.
    """
    from PIL import Image as PILImage

    def field(seed: int) -> np.ndarray:
        small = np.random.default_rng(seed).random((15, 20)).astype(np.float32)
        return np.asarray(
            PILImage.fromarray(small, mode="F").resize((w, h), PILImage.Resampling.BICUBIC),
            dtype=np.float32,
        )

    data = np.clip(np.stack([field(1), field(2), field(3)], axis=-1) * 0.8 + 0.1, 0.0, 1.0)
    return Image(data, "srgb", ppi=300)


class TestSeedProvisionalSet:
    def test_clones_carry_the_measured_lut_unchanged(self, tmp_path):
        """R1: nothing has measured how bleaching or toning reshapes this paper.

        A shaping curve invented from a plausible shape would be indistinguishable in the
        output from a measured one, while being fiction.
        """
        base = _measured_base()
        _, clones = seed_provisional_set(base, "T")
        assert len(clones) == 3
        for clone in clones:
            assert np.array_equal(clone.lut.values, base.lut.values)

    def test_clones_are_provisional_and_carry_no_measurements(self):
        _, clones = seed_provisional_set(_measured_base(), "T")
        for clone in clones:
            assert clone.provisional
            assert clone.measurements["raw_patches"] == []

    def test_spe_is_cloned_unmultiplied(self):
        """B1: the multiplier lives in the set. A pre-multiplied profile applies it twice."""
        _, clones = seed_provisional_set(_measured_base(), "T")
        for clone in clones:
            assert clone.exposure["spe_seconds"] == 810

    def test_clones_keep_the_scan_path(self):
        """F5: seeding is exactly where an undeclared field would have been lost."""
        base = _measured_base()
        _, clones = seed_provisional_set(base, "T")
        for clone in clones:
            assert clone.scan_settings == base.scan_settings
            assert clone.scan_settings != {}

    def test_seeded_layers_agree_with_each_other(self):
        _, clones = seed_provisional_set(_measured_base(), "T")
        by_role = dict(zip(PRINT_ORDER, clones))
        assert check_profile_agreement(by_role) == []

    def test_refuses_to_seed_from_a_provisional_profile(self):
        base = dataclasses.replace(_measured_base(), provisional=True)
        with pytest.raises(TricolourSetError, match="provisional"):
            seed_provisional_set(base, "T")


class TestOutputNaming:
    def test_files_sort_into_printing_order(self):
        names = sorted(output_name("IMG", role) for role in PRINT_ORDER)
        assert names == ["IMG_1M", "IMG_2Y", "IMG_3C"]


class TestMakeTricolour:
    def test_writes_three_negatives_a_manifest_and_a_wall_sheet(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100),
            output_dir=tmp_path / "out", stem="P1", profile_dir=pdir, wedges=False,
            page_mm=(210.0, 297.0),
        )
        assert sorted(p.name for p in result.paths.values()) == [
            "P1_1M.tif", "P1_2Y.tif", "P1_3C.tif"
        ]
        assert (tmp_path / "out" / "P1_tricolour.json").exists()
        assert (tmp_path / "out" / "P1_tricolour.md").exists()

    def test_channel_mapping_is_verified_numerically(self, tmp_path):
        """Plan §11.4 — the eyeball version of this passes on a wrong permutation.

        Two things the plan's wording gets wrong, both found by running it:

        The border must be excluded from the crop. It is a constant ~28% of the picture
        block, and leaving it in drags every correlation towards zero uniformly, hiding
        the difference being measured.

        And the criterion is *highest*, not "far more strongly than the other two". A
        natural photograph has RGB channels correlated at ~0.9 — HPTressII.jpg measures
        0.77 to 0.90 — so a large margin is capped by the source rather than earned by the
        separation, and demanding one fails every real image. A permutation error moves
        which channel is the maximum, which is detectable regardless. The margin is
        asserted here only because this synthetic source has independent channels; that is
        what makes it a stronger test than any photograph can be.
        """
        from PIL import Image as PILImage
        from cyanoneg.blocker import recover_coverage
        from cyanoneg.imageio import load_image

        source = _independent_rgb()
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            source, tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="CHK", profile_dir=pdir, wedges=False,
        )
        rect = result.manifest["placement"]["picture"]
        border = _mm(tset.border_mm)
        blocker = _measured_base().blocker

        for role in PRINT_ORDER:
            film = load_image(result.paths[role])
            print_view = film.data[:, ::-1]  # undo the film flip
            block = print_view[
                rect["y_px"] : rect["y_px"] + rect["h_px"],
                rect["x_px"] : rect["x_px"] + rect["w_px"],
            ]
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
            others = sorted(scores)[:2]
            assert own == max(scores), f"{role} does not track {SOURCE_CHANNEL[role]}"
            assert own > 2.0 * max(others), (
                f"{role} correlates {own:.2f} with {SOURCE_CHANNEL[role]} but "
                f"{max(others):.2f} with another channel — too close to call"
            )

    def test_all_three_share_one_page_geometry(self, tmp_path):
        """Different dimensions between layers would be misregistration by construction."""
        from cyanoneg.imageio import load_image

        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="G", profile_dir=pdir, wedges=False,
        )
        shapes = {load_image(p).data.shape for p in result.paths.values()}
        assert len(shapes) == 1
        assert len({json.dumps(f, sort_keys=True) for f in result.fiducials.values()}) == 1

    def test_mono_source_is_refused(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        grey = _independent_rgb()
        with pytest.raises(ValueError, match="mono"):
            make_tricolour(
                grey.replace(grey.data[..., 0]), tset, PrintSize(130, 100),
                output_dir=tmp_path / "out", stem="M", profile_dir=pdir, wedges=False,
            )

    def test_provisional_layers_are_warned_about(self, tmp_path):
        """A seeded set is an experiment, not a calibrated result. Say so."""
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="W", profile_dir=pdir, wedges=False,
        )
        assert any("provisional" in w for w in result.warnings)


class TestManifest:
    def test_records_exposure_computed_once(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="E", profile_dir=pdir, wedges=False,
        )
        got = {
            role: result.manifest["layers"][role]["exposure"]["instruction_display"]
            for role in PRINT_ORDER
        }
        assert got == {"magenta": "20:15", "yellow": "37:08", "cyan": "14:51"}

    def test_survives_the_named_profile_being_changed_afterwards(self, tmp_path):
        """A6: a profile name does not preserve a run — the file can be overwritten.

        The manifest must still identify the calibration that actually made the negatives.
        """
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="R", profile_dir=pdir, wedges=False,
        )
        recorded = result.manifest["layers"]["magenta"]["profile"]
        name = recorded["name"]

        altered = dataclasses.replace(
            Profile.load(pdir / f"{name}.json"), film_batch="a-different-box"
        )
        altered.save(pdir / f"{name}.json")

        assert recorded["calibration_identity"]["film_batch"] == _measured_base().film_batch
        assert calibration_fingerprint(altered) != recorded["calibration_fingerprint"]

    def test_each_layer_records_the_slot_it_actually_owns(self, tmp_path):
        """Found by looking at the composed sheets rather than by a test.

        The page geometry is identical across layers, so it is tempting to record one
        placement for the run. But ``owned`` is a property of a *sheet*: folding three
        sheets' flags into one record is right for exactly one of them, and it was the
        last one written. The scan-back reads this to know which slot to crop, so two
        layers out of three would have been cropped to a slot that was never printed.
        """
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="S", profile_dir=pdir, wedges=True,
        )
        for role in PRINT_ORDER:
            slot = result.manifest["layers"][role]["wedge_slot"]
            assert slot["owned"] is True, f"{role} does not own its own slot"
            shared = result.manifest["placement"]["wedges"][role]
            assert (slot["x_px"], slot["y_px"]) == (shared["x_px"], shared["y_px"])

        # The shared record must not carry a flag that can only be true for one sheet.
        for slot in result.manifest["placement"]["wedges"].values():
            assert "owned" not in slot

    def test_embeds_the_lut_that_ran(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="L", profile_dir=pdir, wedges=False,
        )
        lut = result.manifest["layers"]["cyan"]["profile"]["lut"]
        assert lut["size"] == _measured_base().lut.size
        assert len(lut["values"]) == lut["size"]


class TestBlockerExtentReaches_the_darkroom:
    """A bound the printer honours but nobody reads is not a guard."""

    def _run(self, tmp_path, extent, stem):
        tset, pdir = _seeded(tmp_path)
        return make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem=stem, profile_dir=pdir, blocker_extent_mm=extent,
        )

    def test_manifest_records_the_region_and_the_setting(self, tmp_path):
        result = self._run(tmp_path, 5.0, "BE")
        assert result.manifest["output"]["blocker_extent_mm"] == 5.0
        region = result.manifest["placement"]["blocker_region"]
        assert region["w_mm"] > 0 and region["h_mm"] > 0
        assert 0.0 < region["cleared_fraction"] < 1.0

    def test_the_coating_limit_is_on_the_wall_sheet(self, tmp_path):
        """The number a person needs while holding a brush, not only in the JSON."""
        self._run(tmp_path, 5.0, "BW")
        sheet = (tmp_path / "out" / "BW_tricolour.md").read_text(encoding="utf-8")
        assert "COAT INSIDE THE BLOCKER" in sheet
        assert "Coat no larger than" in sheet
        assert "clear film blocks" in sheet

    def test_a_flooded_sheet_says_nothing_about_coating_limits(self, tmp_path):
        """No bound, no guard to state — and no phantom limit to obey."""
        result = self._run(tmp_path, None, "BF")
        assert result.manifest["output"]["blocker_extent_mm"] is None
        assert "blocker_region" not in result.manifest["placement"]
        sheet = (tmp_path / "out" / "BF_tricolour.md").read_text(encoding="utf-8")
        assert "COAT INSIDE THE BLOCKER" not in sheet

    def test_bounding_raises_a_warning(self, tmp_path):
        """It surfaces wherever warnings surface: script output and the wall sheet."""
        result = self._run(tmp_path, 5.0, "BX")
        assert any("unfiltered exposure" in w for w in result.warnings)
        sheet = (tmp_path / "out" / "BX_tricolour.md").read_text(encoding="utf-8")
        assert "unfiltered exposure" in sheet


class TestDarkroomSheet:
    def test_states_the_order_and_the_reason(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="D", profile_dir=pdir, wedges=False,
        )
        sheet = (tmp_path / "out" / "D_tricolour.md").read_text(encoding="utf-8")
        assert "ORDER IS NOT NEGOTIABLE" in sheet
        assert "destroys Prussian blue" in sheet
        assert sheet.index("1. MAGENTA") < sheet.index("2. YELLOW") < sheet.index("3. CYAN")

    def test_gives_times_the_way_the_timer_is_set(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="T", profile_dir=pdir, wedges=False,
        )
        sheet = (tmp_path / "out" / "T_tricolour.md").read_text(encoding="utf-8")
        assert "37:08" in sheet and "2228 s" in sheet
        assert "all three channels" in sheet  # the control region instruction

    def test_scan_protocol_is_stated_not_deferred(self, tmp_path):
        """B2 is settled: the between-layer scans are the production measurement.

        The sheet carried a pending-B2 marker while the question was open, on the grounds
        that a wall sheet quietly stating the wrong protocol is worse than one with a
        visible gap. That marker must now be gone, and the instruction real.
        """
        tset, pdir = _seeded(tmp_path)
        make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="B2", profile_dir=pdir, wedges=False,
        )
        sheet = (tmp_path / "out" / "B2_tricolour.md").read_text(encoding="utf-8")

        assert "pending B2" not in sheet
        assert "<!--" not in sheet, "no unresolved markers left on a darkroom sheet"
        assert "DO NOT SKIP A SCAN" in sheet
        for role in PRINT_ORDER:
            assert f"SCAN the {role} wedge slot now" in sheet
        # Cyan is last; there is no layer after it to coat.
        assert "SCAN the cyan wedge slot now, before coating" not in sheet
        assert sheet.count(", before coating the next layer.") == 2
        # The post-cyan scan is the experiment, not the measurement.
        assert "come from the scans taken between layers" in sheet
        # P2's number has exactly one chance to be taken.
        assert "fiducial span" in sheet.lower()


class TestWedgeSidecars:
    """The manifest named ``wedge_<role>.json`` from the start; nothing wrote them.

    The patch layout is shuffled by a seed, so a scanned wedge without its sidecar is a
    60 mm square of unattributable greys. The films for print #1 were made, measured and
    about to be exposed before this was noticed, which is exactly how a dangling
    reference survives: nothing reads it until the one moment it is needed."""

    def _run(self, tmp_path, **kw):
        tset, pdir = _seeded(tmp_path)
        return make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="S", profile_dir=pdir, **kw
        ), tmp_path / "out"

    def test_sidecars_exist_where_the_manifest_says(self, tmp_path):
        result, out = self._run(tmp_path)
        for role in PRINT_ORDER:
            named = result.manifest["layers"][role]["wedge_slot"]["sidecar"]
            assert (out / named).exists(), f"{named} is referenced but absent"

    def test_sidecar_is_a_readable_wedge(self, tmp_path):
        """analyze_wedge refuses anything without a seed, so this is its acceptance test."""
        _, out = self._run(tmp_path)
        side = json.loads((out / "wedge_magenta.json").read_text(encoding="utf-8"))
        assert "seed" in side and side["cells"]
        assert side["levels"] == 16
        assert len(side["cells"]) == 16 * 4

    def test_manifest_records_enough_to_regenerate(self, tmp_path):
        """Reproducing the sidecar must not depend on remembering the call site."""
        result, _ = self._run(tmp_path)
        w = result.manifest["wedge"]
        assert (w["levels"], w["redundancy"], w["seed"]) == (16, 4, WEDGE_SEED)

    def test_a_custom_seed_is_honoured_and_recorded(self, tmp_path):
        result, out = self._run(tmp_path, wedge_seed=4242)
        assert result.manifest["wedge"]["seed"] == 4242
        assert json.loads((out / "wedge_cyan.json").read_text(encoding="utf-8"))["seed"] == 4242

    def test_no_wedges_means_no_dangling_reference(self, tmp_path):
        result, out = self._run(tmp_path, wedges=False)
        assert result.manifest["wedge"] is None
        assert not list(out.glob("wedge_*.json"))


class TestSlotIdentity:
    """Three pixel-identical squares; only position says which layer a crop is."""

    def test_left_to_right_is_print_order(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="P", profile_dir=pdir,
        )
        assert slot_positions(result.manifest) == dict(zip(PRINT_ORDER, SLOT_POSITIONS))

    def test_the_wall_sheet_says_which_square_to_scan(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="W", profile_dir=pdir,
        )
        sheet = (tmp_path / "out" / "W_tricolour.md").read_text(encoding="utf-8")
        assert "the **left** one of the three" in sheet
        assert "IDENTICAL" in sheet
        assert "Left to right on the print: **magenta**, **yellow**, **cyan**" in sheet

    def test_a_run_without_wedges_claims_no_positions(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="N", profile_dir=pdir, wedges=False,
        )
        assert slot_positions(result.manifest) == {}
        sheet = (tmp_path / "out" / "N_tricolour.md").read_text(encoding="utf-8")
        assert "IDENTICAL" not in sheet


class TestScanBack:
    def _run(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        result = make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="B", profile_dir=pdir,
        )
        return result, tmp_path / "out"

    def test_each_layer_is_read_through_its_own_channel(self, tmp_path):
        """The quantity comes from the manifest, not from the caller, so a layer cannot
        be read through the wrong channel by someone who has not thought about it."""
        import simulate
        from cyanoneg.imageio import save_tiff
        from cyanoneg.targets import step_wedge as make_wedge

        result, out = self._run(tmp_path)
        wedge = make_wedge((255, 64, 0), saturation=1.0, levels=16, redundancy=4)
        scan = simulate.scan_of(
            simulate.render_print(wedge, lambda x: np.clip(x, 0, 1)),
            colour=True, scale=1.0, rotate_deg=0.0, perspective=0.0, noise=0.0, blur_px=0.4,
        )
        path = save_tiff(out / "slot.tif", scan)

        for role in PRINT_ORDER:
            got = analyze_layer_wedge(out / "B_tricolour.json", role, path)
            assert got.quantity == RESPONSE_QUANTITY[role]

    def test_a_missing_sidecar_says_how_to_get_it_back(self, tmp_path):
        result, out = self._run(tmp_path)
        (out / "wedge_magenta.json").unlink()
        with pytest.raises(TricolourSetError, match="step_wedge"):
            analyze_layer_wedge(out / "B_tricolour.json", "magenta", out / "B_1M.tif")

    def test_a_run_without_wedges_is_refused_clearly(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="X", profile_dir=pdir, wedges=False,
        )
        out = tmp_path / "out"
        with pytest.raises(TricolourSetError, match="nothing to read back"):
            analyze_layer_wedge(out / "X_tricolour.json", "cyan", out / "X_3C.tif")

    def test_unknown_layer_refused(self, tmp_path):
        _, out = self._run(tmp_path)
        with pytest.raises(ValueError, match="unknown layer"):
            analyze_layer_wedge(out / "B_tricolour.json", "green", out / "B_1M.tif")

    def test_a_dict_manifest_needs_its_directory(self, tmp_path):
        result, _ = self._run(tmp_path)
        with pytest.raises(ValueError, match="manifest_dir is required"):
            analyze_layer_wedge(result.manifest, "magenta", "unused.tif")


class TestDiagnoseLayer:
    """A failure that reports nothing cannot be compared against the next failure.

    ``analyze_layer_wedge`` either returns a curve or raises, which is right for deriving
    a profile and wrong for someone changing bleach and toning between attempts. The
    first real magenta print separated by -2.5 L* on a sheet whose bare paper read 55
    instead of 95; that is a diagnosis, and it has to survive being reported."""

    def _run(self, tmp_path):
        tset, pdir = _seeded(tmp_path)
        make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "out",
            stem="D", profile_dir=pdir,
        )
        return tmp_path / "out" / "D_tricolour.json"

    def _wedge_crop(self, response):
        import simulate
        from cyanoneg.targets import step_wedge as mk
        w = mk((255, 64, 0), saturation=1.0, levels=16, redundancy=4)
        return simulate.scan_of(
            simulate.render_print(w, response), colour=True, scale=1.0, rotate_deg=0.0,
            perspective=0.0, noise=0.0, blur_px=0.4,
        )

    #: The simulator renders paper white at this L*. The sheet stand-in has to agree with
    #: it, or every comparison between "bare paper" and "the wedge's white" is measuring
    #: the gap between two different notions of paper rather than anything about the print.
    SIM_PAPER_LSTAR = 93.0

    def _sheet(self, lstar: float):
        """A stand-in sheet at a given lightness: only its corners are ever read."""
        from cyanoneg.imageio import from_linear

        y = ((lstar + 16.0) / 116.0) ** 3
        srgb = float(from_linear(np.full((1, 1, 3), y, dtype=np.float32), "srgb")[0, 0, 0])
        return Image(np.full((240, 200, 3), srgb, dtype=np.float32), "srgb", ppi=300)

    def test_a_working_layer_separates_and_yields_a_curve(self, tmp_path):
        manifest = self._run(tmp_path)
        got = diagnose_layer(
            manifest, "magenta", self._sheet(self.SIM_PAPER_LSTAR), self._wedge_crop(lambda x: np.clip(x, 0, 1))
        )
        assert got.separated
        assert got.separation >= MIN_SEPARATION
        assert got.analysis is not None
        assert got.quantity == "lstar_g"
        assert "SEPARATED" in got.summary()

    def test_a_stained_sheet_is_called_out(self, tmp_path):
        """The finding that mattered: paper white is not 100 when a dye bath has run."""
        manifest = self._run(tmp_path)
        got = diagnose_layer(
            manifest, "magenta", self._sheet(55.0), self._wedge_crop(lambda x: np.clip(x, 0, 1))
        )
        assert got.stain_floor < 90.0
        assert any("the sheet itself is stained" in n for n in got.notes)

    def test_a_flat_wedge_reports_how_far_off_it_was(self, tmp_path):
        """Not "the print failed" — the number, so two attempts can be compared."""
        manifest = self._run(tmp_path)
        got = diagnose_layer(
            manifest, "magenta", self._sheet(55.0), self._wedge_crop(lambda x: 0.5 + 0.0 * x)
        )
        assert not got.separated
        assert got.analysis is None
        assert abs(got.separation) < MIN_SEPARATION
        assert any("did not separate" in n for n in got.notes)
        assert f"{got.separation:5.1f}" in got.summary()

    def test_fog_is_distinguished_from_underexposure(self, tmp_path):
        """Two faults that look alike on the sheet and want opposite corrections.

        A dark print whose *blocked* patches are still clean is underexposed and wants
        more light. Blocked patches darker than bare paper mean the darkening came with
        the coating or the chemistry, and more light cannot fix it. The first magenta
        print was the second kind: blocked patches 14 L* below an uncoated corner.
        """
        manifest = self._run(tmp_path)
        dark = diagnose_layer(
            manifest, "magenta", self._sheet(self.SIM_PAPER_LSTAR),
            # weak shadows, but full ink still reaches paper white — underexposed
            self._wedge_crop(lambda x: 0.60 + 0.40 * np.clip(x, 0, 1)),
        )
        assert not any("below bare paper" in n for n in dark.notes), (
            "an underexposed print with clean blocked patches must not be called fogged"
        )

        fogged = diagnose_layer(
            manifest, "magenta", self._sheet(self.SIM_PAPER_LSTAR),
            self._wedge_crop(lambda x: 0.05 + 0.25 * np.clip(x, 0, 1)),
        )
        assert any("below bare paper" in n for n in fogged.notes)

    def test_unknown_layer_and_missing_wedges_refused(self, tmp_path):
        manifest = self._run(tmp_path)
        with pytest.raises(ValueError, match="unknown layer"):
            diagnose_layer(manifest, "green", self._sheet(self.SIM_PAPER_LSTAR), self._wedge_crop(lambda x: x))

        tset, pdir = _seeded(tmp_path)
        make_tricolour(
            _independent_rgb(), tset, PrintSize(130, 100), output_dir=tmp_path / "nw",
            stem="N", profile_dir=pdir, wedges=False,
        )
        with pytest.raises(TricolourSetError, match="no wedges"):
            diagnose_layer(
                tmp_path / "nw" / "N_tricolour.json", "magenta",
                self._sheet(self.SIM_PAPER_LSTAR), self._wedge_crop(lambda x: x),
            )
