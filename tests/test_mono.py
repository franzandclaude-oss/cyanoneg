import numpy as np
import pytest

from cyanoneg.imageio import Image
from cyanoneg.mono import channel_noise, suggest_weights, to_mono


def _smooth_base(shape=(256, 256)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return ((np.sin(x / 40) + np.cos(y / 30)) / 4 + 0.5).astype(np.float32)


class TestToMono:
    @pytest.mark.parametrize("mix_in", ["linear", "encoded"])
    def test_neutral_grey_is_fixed_point(self, mix_in):
        """(g, g, g) must map to g regardless of weights or mixing space."""
        base = _smooth_base((64, 64))
        img = Image(np.stack([base] * 3, axis=-1), "srgb")
        out = to_mono(img, weights=(0.7, 0.2, 0.1), mix_in=mix_in)
        assert np.abs(out.data - base).max() < 1e-5

    def test_weights_normalised(self):
        base = _smooth_base((32, 32))
        img = Image(np.stack([base] * 3, axis=-1), "srgb")
        a = to_mono(img, weights=(1, 1, 0))
        b = to_mono(img, weights=(50, 50, 0))
        assert np.abs(a.data - b.data).max() < 1e-6

    def test_zero_weights_rejected(self):
        img = Image(np.zeros((4, 4, 3), dtype=np.float32), "srgb")
        with pytest.raises(ValueError, match="sum to a positive value"):
            to_mono(img, weights=(0, 0, 0))

    def test_mono_passthrough(self):
        img = Image(np.zeros((4, 4), dtype=np.float32), "srgb")
        assert to_mono(img) is img

    def test_output_is_2d_same_space(self, ramp_image):
        out = to_mono(ramp_image)
        assert out.data.ndim == 2
        assert out.space == ramp_image.space
        assert out.ppi == ramp_image.ppi


class TestChannelNoise:
    def test_identifies_grainy_channel(self):
        rng = np.random.default_rng(1)
        base = _smooth_base()
        rgb = np.stack(
            [
                base + rng.normal(0, 0.005, base.shape),
                base + rng.normal(0, 0.004, base.shape),
                base + rng.normal(0, 0.03, base.shape),  # grainy blue, as in real scans
            ],
            axis=-1,
        )
        noise = channel_noise(Image(np.clip(rgb, 0, 1).astype(np.float32), "srgb"))
        assert noise["blue"] > 3 * noise["green"]
        assert suggest_weights(noise) == (0.30, 0.59, 0.0)

    def test_quiet_channels_keep_defaults(self):
        noise = {"red": 0.005, "green": 0.004, "blue": 0.006}
        assert suggest_weights(noise) == (0.30, 0.59, 0.11)

    def test_mono_rejected(self):
        with pytest.raises(ValueError):
            channel_noise(Image(np.zeros((8, 8), dtype=np.float32), "srgb"))
