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


class TestRoundTrip:
    def test_save_load_preserves_everything(self, tmp_path):
        profile = _valid_profile(
            paper="Paper 1",
            chemistry="Chemistry",
            film="Film 1",
            driver_settings={"color_correction": "No Color Adjustment"},
            lut=Lut(np.linspace(0, 1, 256) ** 0.85),
            measurements={"raw_patches": [[0, 0.01], [128, 0.5]], "scan_date": "2026-07-25"},
        )
        path = profile.save(tmp_path / "p.json")
        back = Profile.load(path)
        assert back.name == profile.name
        assert back.film == "Film 1"
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

    def test_paper1_provisional(self):
        profile = Profile.load(PROFILE_DIR / "paper1-provisional.json")
        assert profile.provisional
        assert profile.lut.is_identity()
        assert not profile.is_ready_to_print  # blocker awaits the HSB grid reading
        assert profile.film == "Film 1"
        assert profile.paper == "Paper 1"
