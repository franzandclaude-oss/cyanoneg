"""Pipeline ordering — both failure modes here (curve after inversion, missing mirror)
produce silently wrong output, so the order is asserted two independent ways."""

import numpy as np
import pytest

from cyanoneg import pipeline
from cyanoneg.imageio import Image
from cyanoneg.lut import Lut
from cyanoneg.pipeline import PrintSize, make_negative
from cyanoneg.profiles import Profile


def _profile(**overrides) -> Profile:
    defaults = dict(
        name="test",
        provisional=True,
        blocker={"model": "fixed_hue", "rgb": [255, 0, 0], "saturation": 1.0},
    )
    defaults.update(overrides)
    return Profile(**defaults)


@pytest.fixture
def ramp() -> Image:
    data = np.tile(np.linspace(0.0, 1.0, 128, dtype=np.float32), (64, 1))
    return Image(np.stack([data] * 3, axis=-1), "srgb", ppi=300)


class TestStepOrder:
    def test_steps_run_in_mandated_order(self, ramp, monkeypatch):
        calls: list[str] = []

        def record(name, fn):
            def wrapper(*args, **kwargs):
                calls.append(name)
                return fn(*args, **kwargs)

            return wrapper

        for step in ("step_mono", "step_apply_lut", "step_resize", "step_invert", "step_blocker", "step_flip"):
            monkeypatch.setattr(pipeline, step, record(step, getattr(pipeline, step)))

        make_negative(ramp, _profile(), PrintSize(30, 20))
        assert calls == [
            "step_mono",
            "step_apply_lut",  # on the positive —
            "step_resize",
            "step_invert",  # — before inversion
            "step_blocker",
            "step_flip",  # mirror is always last
        ]

    def test_raw_scan_inverts_first(self, ramp, monkeypatch):
        calls: list[str] = []
        original = pipeline.step_invert

        def counting_invert(image):
            calls.append("invert")
            return original(image)

        monkeypatch.setattr(pipeline, "step_invert", counting_invert)
        make_negative(ramp, _profile(), PrintSize(30, 20), raw_scan=True)
        assert calls.count("invert") == 2  # once to make a positive, once to make the negative

    def test_curve_applies_to_positive_not_negative(self):
        """Functional proof of curve-before-invert, independent of call tracing.

        With LUT(x) = x², the correct order gives a negative value of 1 - v²; applying the
        curve after inversion would give (1 - v)². Checked on a flat patch so the resize
        cannot blur the distinction.
        """
        value = 0.6
        flat = Image(np.full((40, 40, 3), value, dtype=np.float32), "srgb", ppi=300)
        profile = _profile(lut=Lut(np.linspace(0, 1, 256) ** 2))
        negative = make_negative(flat, profile, PrintSize(10, 10))
        # Red blocker: out = (1, v, v), so the green channel is the negative value itself.
        green = float(np.median(negative.data[..., 1]))
        assert green == pytest.approx(1 - value**2, abs=0.01)
        assert abs(green - (1 - value) ** 2) > 0.15  # rules out the wrong order


class TestGeometryAndOutput:
    def test_fits_within_print_size_and_mirrored(self, ramp):
        negative = make_negative(ramp, _profile(), PrintSize(100, 50), output_ppi=360)
        h, w, _ = negative.data.shape
        assert w <= round(100 / 25.4 * 360) + 1
        assert h <= round(50 / 25.4 * 360) + 1
        # Source ramp is dark-left. Dark must print dark, so the left needs UV through it —
        # clear film, i.e. high negative value. The mirror puts that clear end on the right.
        green = negative.data[..., 1]
        assert green[:, -1].mean() > green[:, 0].mean()

    def test_output_is_coloured_not_grey(self, ramp):
        negative = make_negative(ramp, _profile(), PrintSize(50, 25))
        assert not np.allclose(negative.data[..., 0], negative.data[..., 1])

    def test_writes_16bit_tiff_with_ppi(self, ramp, tmp_path):
        import tifffile

        out = tmp_path / "neg.tif"
        make_negative(ramp, _profile(), PrintSize(50, 25), output_path=out, output_ppi=360)
        with tifffile.TiffFile(out) as tf:
            page = tf.pages[0]
            assert page.dtype == np.uint16
            num, den = page.tags["XResolution"].value
            assert num / den == 360

    def test_profile_without_blocker_refuses(self, ramp):
        profile = _profile(blocker={"model": "fixed_hue", "rgb": None, "saturation": None})
        with pytest.raises(ValueError, match="no blocker colour"):
            make_negative(ramp, profile, PrintSize(50, 25))

    def test_lut_applied_in_profile_space(self):
        """A gamma22 profile must see gamma22-encoded values, not sRGB ones."""
        value = 0.5
        flat = Image(np.full((40, 40, 3), value, dtype=np.float32), "srgb", ppi=300)
        profile = _profile(working_space="gamma22", lut=Lut(np.linspace(0, 1, 256) ** 2))
        negative = make_negative(flat, profile, PrintSize(10, 10))
        from cyanoneg.imageio import convert_space

        corrected = float(convert_space(np.array([value], dtype=np.float32), "srgb", "gamma22")[0]) ** 2
        assert float(np.median(negative.data[..., 1])) == pytest.approx(1 - corrected, abs=0.01)
