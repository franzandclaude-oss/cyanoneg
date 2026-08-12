import json

import numpy as np
import pytest

from cyanoneg.lut import Lut
from cyanoneg.profiles import PROFILE_DIR, Profile, ProfileError


def _valid_profile(**overrides) -> Profile:
    defaults = dict(
        name="Test",
        provisional=True,
        blocker={"model": "fixed_hue", "rgb": [64, 128, 0], "saturation": 0.8},
    )
    defaults.update(overrides)
    return Profile(**defaults)


class TestValidation:
    def test_valid_profile_passes(self):
        assert _valid_profile().validate() == []

    def test_bad_working_space(self):
        problems = _valid_profile(working_space="adobe98").validate()
        assert any("working_space" in p for p in problems)

    def test_bad_blocker_rgb(self):
        problems = _valid_profile(
            blocker={"model": "fixed_hue", "rgb": [300, 0, 0], "saturation": 1.0}
        ).validate()
        assert any("blocker rgb" in p for p in problems)

    def test_bad_saturation(self):
        problems = _valid_profile(
            blocker={"model": "fixed_hue", "rgb": [255, 0, 0], "saturation": 1.5}
        ).validate()
        assert any("saturation" in p for p in problems)

    def test_non_provisional_requires_measured_blocker(self):
        problems = _valid_profile(
            provisional=False,
            blocker={"model": "fixed_hue", "rgb": None, "saturation": None},
        ).validate()
        assert any("non-provisional" in p for p in problems)

    def test_non_monotonic_lut_flagged(self):
        values = np.linspace(0, 1, 256)
        values[100] = 0.9  # spike
        problems = _valid_profile(lut=Lut(values)).validate()
        assert any("monotonic" in p for p in problems)

    def test_ready_to_print(self):
        assert _valid_profile().is_ready_to_print
        assert not _valid_profile(
            blocker={"model": "fixed_hue", "rgb": None, "saturation": None}
        ).is_ready_to_print


class TestProvisionalStrictness:
    """``provisional`` must be a genuine JSON boolean, not something coerced into one.

    ``bool("false")`` is ``True`` in Python, so a naive ``bool(d.get(...))`` conversion in
    ``from_dict`` would silently accept a malformed profile — a hand-edited or corrupted
    JSON file with ``"provisional": "false"`` would load as calibrated. ``from_dict`` must
    pass the raw value through untouched and let ``validate`` reject anything that is not
    actually a bool.
    """

    @pytest.mark.parametrize("bad_value", ["false", "true", 0, 1, None, "", []])
    def test_non_bool_provisional_rejected(self, bad_value):
        profile = Profile.from_dict(
            {
                "name": "Test",
                "provisional": bad_value,
                "blocker": {"model": "fixed_hue", "rgb": [64, 128, 0], "saturation": 0.8},
            }
        )
        problems = profile.validate()
        assert any("provisional" in p for p in problems), (
            f"{bad_value!r} should have been rejected, got problems={problems!r}"
        )

    def test_missing_provisional_defaults_to_true(self):
        profile = Profile.from_dict(
            {
                "name": "Test",
                "blocker": {"model": "fixed_hue", "rgb": [64, 128, 0], "saturation": 0.8},
            }
        )
        assert profile.provisional is True
        assert profile.validate() == []

    def test_string_false_provisional_raises_on_load(self, tmp_path):
        """The exact failure mode: a malformed value must not survive a load/save round trip."""
        path = tmp_path / "malformed.json"
        d = _valid_profile(
            blocker={"model": "fixed_hue", "rgb": [64, 128, 0], "saturation": 0.8}
        ).to_dict()
        d["provisional"] = "false"
        path.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(ProfileError, match="provisional"):
            Profile.load(path)


class TestRoundTrip:
    def test_save_load_preserves_everything(self, tmp_path):
        profile = _valid_profile(
            paper="CassArt 300 Sm",
            chemistry="Chemistry",
            film="Film Generic",
            film_batch="Batch One",
            driver_settings={"color_correction": "No Color Adjustment"},
            lut=Lut(np.linspace(0, 1, 256) ** 0.85),
            measurements={"raw_patches": [[0, 0.01], [128, 0.5]], "scan_date": "2026-07-25"},
        )
        path = profile.save(tmp_path / "p.json")
        back = Profile.load(path)
        assert back.name == profile.name
        assert back.film == "Film Generic"
        assert back.film_batch == "Batch One"
        assert back.working_space == "srgb"
        assert back.driver_settings == profile.driver_settings
        assert back.measurements == profile.measurements
        assert np.abs(back.lut.values - profile.lut.values).max() < 1e-8

    def test_saving_invalid_profile_refused(self, tmp_path):
        bad = _valid_profile(working_space="wrong")
        with pytest.raises(ProfileError, match="refusing to save"):
            bad.save(tmp_path / "bad.json")

    def test_loading_garbage_raises(self, tmp_path):
        path = tmp_path / "garbage.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ProfileError, match="not valid JSON"):
            Profile.load(path)


class TestShippedProfiles:
    """The two profiles in profiles/ must always load, and mean what they claim."""

    def test_linear_baseline(self):
        profile = Profile.load(PROFILE_DIR / "linear.json")
        assert not profile.provisional
        assert profile.lut.is_identity()
        assert profile.is_ready_to_print

    def test_cassart_names_real_materials(self):
        """"Paper 1" and "Film 1" were stand-ins until the actual materials were identified.

        A profile is only reproducible if someone can go and buy what it was measured on,
        so the placeholders must not survive. The batch is a field of its own rather than
        part of the film's name: when prints drift for no visible reason a batch change is
        the first suspect, and a fact buried in free text cannot be compared across
        profiles.
        """
        profile = Profile.load(PROFILE_DIR / "CassArt 300 Sm.json")
        for value in (profile.paper, profile.film, profile.film_batch):
            assert value.strip(), "materials must be named"
        assert profile.paper == "CassArt 300 Sm"
        assert profile.film == "Film Generic"
        assert profile.film_batch == "Batch One"
        assert "Paper 1" not in profile.name, "the placeholder name must not survive"

    def test_cassart_is_measured_not_provisional(self):
        """The paper is calibrated: measured blocker, measured curve, provisional cleared.

        This assertion has now inverted twice as the calibration progressed — first the
        profile could not print at all, then it could print but was provisional pending a
        curve. Both earlier states are gone, and so is the filename that recorded them.
        """
        profile = Profile.load(PROFILE_DIR / "CassArt 300 Sm.json")
        assert profile.is_ready_to_print
        assert not profile.provisional
        assert not profile.lut.is_identity(), "a measured paper cannot have an identity curve"

    def test_cassart_curve_is_usable(self):
        """The curve must be monotonic and span the full range, or it is not a tone curve.

        A non-monotonic curve would reverse tones somewhere in the scale; one that does not
        reach both ends would clip. Neither is visible by looking at a profile.
        """
        lut = Profile.load(PROFILE_DIR / "CassArt 300 Sm.json").lut
        values = np.asarray(lut.values)
        assert np.all(np.diff(values) >= -1e-9)
        assert values[0] == pytest.approx(0.0, abs=1e-6)
        assert values[-1] == pytest.approx(1.0, abs=1e-6)

    def test_cassart_records_what_it_was_measured_from(self):
        """A calibration nobody can trace is a calibration nobody can check."""
        d = json.loads((PROFILE_DIR / "CassArt 300 Sm.json").read_text(encoding="utf-8"))
        m = d["measurements"]
        assert m["raw_patches"], "patch readings must be kept so the curve can be recomputed"
        assert m["scan_date"] and m["wedge_scan"] and m["blocker_scan"]
        assert 0.5 < m["density_range"] < 2.0

    def test_shipped_blockers_are_no_longer_the_placeholder(self):
        """(255, 0, 0) was a stand-in until the grid was measured; nothing should still use it.

        A profile silently carrying the placeholder would print, and print plausibly, while
        blocking UV worse than the measured hue — the kind of wrong that looks fine.
        """
        for name in ("linear", "CassArt 300 Sm"):
            blocker = Profile.load(PROFILE_DIR / f"{name}.json").blocker
            assert tuple(blocker["rgb"]) != (255, 0, 0), f"{name} still holds the placeholder"
            assert blocker["model"] == "fixed_hue"

    def test_roundtrip_preserves_every_key(self, tmp_path):
        """Load then save must not quietly drop a key the JSON already carries.

        ``to_dict`` and ``from_dict`` both work from fixed key lists, so a field present
        in the file but absent from the dataclass survives reading and vanishes on write.
        ``scan_settings`` did exactly that. It matters beyond one field: seeding a
        tricolour set clones a measured profile three times, so anything lost here is lost
        from all three layer profiles at once — and the agreement check in
        ``tricolour.MUST_AGREE`` is built from the dataclass, so it cannot flag what the
        dataclass never admitted existed.

        Written against the shipped profile rather than a constructed one so that the next
        undeclared field is caught too.
        """
        source = json.loads((PROFILE_DIR / "CassArt 300 Sm.json").read_text(encoding="utf-8"))
        out = tmp_path / "roundtrip.json"
        Profile.load(PROFILE_DIR / "CassArt 300 Sm.json").save(out)
        written = json.loads(out.read_text(encoding="utf-8"))

        missing = sorted(set(source) - set(written))
        assert not missing, f"round-trip dropped {missing}"

    def test_scan_settings_survives_and_is_checkable(self, tmp_path):
        """The scan path travels with the profile, and the set agreement check can see it."""
        from cyanoneg.tricolour import MUST_AGREE

        assert "scan_settings" in MUST_AGREE, "layers could disagree on how they were scanned"
        loaded = Profile.load(PROFILE_DIR / "CassArt 300 Sm.json")
        assert loaded.scan_settings["device"] == "Epson Perfection V600"
        out = tmp_path / "rt.json"
        loaded.save(out)
        assert Profile.load(out).scan_settings == loaded.scan_settings
