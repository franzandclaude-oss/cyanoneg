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
from cyanoneg.profiles import Profile
from cyanoneg.targets import PPI, _mm, step_wedge
from cyanoneg.tricolour import (
    ALLOWED_TO_DIFFER,
    CHANNEL_INDEX,
    CONTROL_BORDER_MM,
    CONTROL_MM,
    MUST_AGREE,
    RESPONSE_QUANTITY,
    _glyph_stamp,
    control_region,
    format_seconds,
    full_blocker_value,
    layer_exposure,
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
        """picture 120 + gap 20 + wedges 60 + gap 5 + control 40 = 245 of 297."""
        _, place = tricolour_page(_picture(), _wedges(), "magenta", BLOCKER_RGB, SATURATION)
        top = place["picture"]["y_px"]
        bottom = place["control"]["y_px"] + place["control"]["h_px"]
        assert round((bottom - top) / PPI * 25.4) == 245
        assert top / PPI * 25.4 == pytest.approx(26, abs=1)

    def test_oversized_picture_raises_rather_than_crops(self):
        with pytest.raises(ValueError, match="but the page is"):
            tricolour_page(_picture(200, 260), _wedges(), "magenta", BLOCKER_RGB, SATURATION)

    def test_unknown_owner_is_refused(self):
        with pytest.raises(ValueError, match="unknown owner"):
            tricolour_page(_picture(), _wedges(), "green", BLOCKER_RGB, SATURATION)
