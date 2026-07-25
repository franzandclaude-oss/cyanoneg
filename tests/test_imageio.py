"""Colour-space handling: the tests guarding against silent tonal errors."""

import numpy as np
import pytest

from cyanoneg import imageio as cio

SIXTEEN_BIT_STEP = 1.0 / 65535.0


class TestTransferCurves:
    @pytest.mark.parametrize("space", cio.SPACES)
    def test_round_trip_well_below_16bit_precision(self, space):
        x = np.linspace(0.0, 1.0, 100_001, dtype=np.float32)
        back = cio.from_linear(cio.to_linear(x, space), space)
        assert np.abs(back - x).max() < SIXTEEN_BIT_STEP / 10

    def test_srgb_is_not_gamma22(self):
        """The two curves differ most in the shadows — conflating them is the classic error."""
        x = np.linspace(0.0, 1.0, 4097, dtype=np.float32)
        divergence = np.abs(cio.to_linear(x, "srgb") - cio.to_linear(x, "gamma22")).max()
        assert divergence > 0.005

    def test_srgb_linear_segment(self):
        """Below the breakpoint sRGB is linear (slope 1/12.92), not a power curve."""
        x = np.array([0.001, 0.02, 0.04], dtype=np.float32)
        assert np.allclose(cio.to_linear(x, "srgb"), x / 12.92, atol=1e-7)

    def test_unknown_space_rejected(self):
        with pytest.raises(ValueError, match="unknown colour space"):
            cio.to_linear(np.zeros(4), "adobe98")

    def test_convert_space_identity_when_same(self):
        x = np.random.default_rng(0).random(100).astype(np.float32)
        assert np.array_equal(cio.convert_space(x, "srgb", "srgb"), x)


class TestLoadSave:
    def test_chart_loads_as_srgb_300ppi_16bit(self, chart_path):
        img = cio.load_image(chart_path)
        assert img.space == "srgb"
        assert img.ppi == 300.0
        assert img.bit_depth == 16
        assert img.data.dtype == np.float32
        assert img.data.shape == (1507, 1507, 3)

    def test_untagged_file_raises_without_explicit_space(self, tmp_path):
        """Guessing a colour space is forbidden — the file must be tagged or told."""
        import tifffile

        path = tmp_path / "untagged.tif"
        tifffile.imwrite(path, np.zeros((8, 8, 3), dtype=np.uint16))
        with pytest.raises(cio.ColourSpaceError, match="no identifiable ICC profile"):
            cio.load_image(path)
        # ...but an explicit override is honoured.
        assert cio.load_image(path, space="gamma22").space == "gamma22"

    def test_save_reload_round_trip(self, tmp_path, ramp_image):
        path = cio.save_tiff(tmp_path / "rt.tif", ramp_image)
        back = cio.load_image(path)
        assert back.space == "srgb"
        assert back.ppi == 300.0
        # Arbitrary float data survives to within half a 16-bit quantisation step…
        assert np.abs(back.data - ramp_image.data).max() <= 0.5 / 65535.0
        # …and already-quantised data round-trips bit-exact.
        again = cio.load_image(cio.save_tiff(tmp_path / "rt2.tif", back))
        assert np.array_equal(again.data, back.data)

    def test_save_bare_array_requires_space(self, tmp_path):
        with pytest.raises(cio.ColourSpaceError):
            cio.save_tiff(tmp_path / "x.tif", np.zeros((4, 4, 3), dtype=np.float32))
