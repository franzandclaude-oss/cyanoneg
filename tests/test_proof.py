"""Soft proof: it must predict from measurement, and refuse without it."""

import numpy as np
import pytest

from cyanoneg.imageio import Image
from cyanoneg.lut import Lut
from cyanoneg.pipeline import PrintSize, make_negative
from cyanoneg.profiles import PROFILE_DIR, Profile
from cyanoneg.proof import ProofUnavailable, can_proof, measured_response, soft_proof

PAPER_LSTAR, BLACK_LSTAR = 93.0, 22.0


def response(x):
    raw = 0.5 * (1 + np.tanh(3.5 * (np.asarray(x) - 0.45)))
    lo, hi = 0.5 * (1 + np.tanh(3.5 * -0.45)), 0.5 * (1 + np.tanh(3.5 * 0.55))
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def measured_profile(**overrides) -> Profile:
    patches = [
        {"value": v, "lstar": float(BLACK_LSTAR + response(v / 255) * (PAPER_LSTAR - BLACK_LSTAR))}
        for v in range(256)
    ]
    defaults = dict(
        name="measured",
        provisional=False,
        blocker={"model": "fixed_hue", "rgb": [255, 0, 0], "saturation": 1.0},
        lut=Lut.identity(),
        measurements={"raw_patches": patches, "scan_date": "2026-07-25"},
    )
    defaults.update(overrides)
    return Profile(**defaults)


class TestRefusal:
    """A proof invented from no data would look authoritative and be fiction."""

    def test_profile_without_measurements_cannot_proof(self):
        """The refusal is what keeps a proof honest — it must survive Paper 1 being measured.

        This used to assert against the shipped Paper 1 profile, which carried no patches.
        It now does, so the test builds its own empty profile rather than quietly becoming
        a test of nothing the day the calibration landed.
        """
        profile = Profile.load(PROFILE_DIR / "linear.json")
        assert not profile.measurements.get("raw_patches")
        assert not can_proof(profile)
        with pytest.raises(ProofUnavailable, match="no measured patches"):
            measured_response(profile)

    def test_measured_paper1_can_proof(self):
        """The other half of the same rule: a measured profile must actually proof."""
        profile = Profile.load(PROFILE_DIR / "paper1-provisional.json")
        assert can_proof(profile)
        response = measured_response(profile)
        assert response is not None

    def test_too_few_patches(self):
        profile = measured_profile(measurements={"raw_patches": [{"value": 0, "lstar": 90.0}]})
        with pytest.raises(ProofUnavailable, match="too few"):
            measured_response(profile)

    def test_flat_measurements(self):
        patches = [{"value": v, "lstar": 60.0} for v in range(0, 256, 8)]
        profile = measured_profile(measurements={"raw_patches": patches})
        with pytest.raises(ProofUnavailable, match="no measured tonal range"):
            measured_response(profile)

    def test_malformed_patches(self):
        profile = measured_profile(measurements={"raw_patches": [{"nope": 1}]})
        with pytest.raises(ProofUnavailable, match="malformed"):
            measured_response(profile)


class TestResponseRecovery:
    def test_recovers_the_known_response(self):
        recovered = measured_response(measured_profile())
        x = np.linspace(0, 1, 501)
        assert np.abs(recovered.apply(x) - response(x)).max() < 0.01

    def test_response_is_monotone(self):
        assert np.all(np.diff(measured_response(measured_profile()).values) >= 0)


class TestProofRendering:
    @pytest.fixture
    def ramp_negative(self):
        profile = measured_profile()
        ramp = np.tile(np.linspace(0.0, 1.0, 256, dtype=np.float32), (48, 1))
        source = Image(np.stack([ramp] * 3, axis=-1), "srgb", ppi=300)
        return make_negative(source, profile, PrintSize(60, 12)), profile

    def test_tonality_matches_the_positive(self, ramp_negative):
        """Dark in, dark out. The proof must not be inverted, and must un-mirror."""
        negative, profile = ramp_negative
        proof = soft_proof(negative, profile)
        row = proof.data[proof.data.shape[0] // 2]
        assert row[0].mean() < row[-1].mean()  # source ramp was dark-left

    def test_monotone_across_the_ramp(self, ramp_negative):
        negative, profile = ramp_negative
        proof = soft_proof(negative, profile)
        row = proof.data[proof.data.shape[0] // 2, :, 2]
        assert np.all(np.diff(row) >= -0.02)

    def test_shadows_are_blue_highlights_are_paper(self, ramp_negative):
        negative, profile = ramp_negative
        proof = soft_proof(negative, profile)
        row = proof.data[proof.data.shape[0] // 2]
        shadow, highlight = row[0], row[-1]
        assert shadow[2] > shadow[0] + 0.15  # blue channel dominates in the shadows
        assert highlight.min() > 0.9  # highlights are bare paper
        assert highlight.max() - highlight.min() < 0.1  # and near-neutral

    def test_preserves_geometry_and_space(self, ramp_negative):
        negative, profile = ramp_negative
        proof = soft_proof(negative, profile)
        assert proof.data.shape == negative.data.shape
        assert proof.space == negative.space
        assert proof.ppi == negative.ppi

    def test_rejects_mono_input(self):
        profile = measured_profile()
        mono = Image(np.zeros((8, 8), dtype=np.float32), "srgb", ppi=300)
        with pytest.raises(ValueError, match="colour-blocked"):
            soft_proof(mono, profile)

    def test_correction_curve_flattens_the_proof(self):
        """The point of calibration: with the derived correction applied, the proofed
        print's lightness should track the positive far more linearly than without it.

        Measured on print lightness recovered from the proof, not on its encoded pixels —
        reflectance is linear in lightness, but the sRGB encoding of it is not, so a
        perfectly linearised print still has a curved code-value ramp.
        """
        from cyanoneg.imageio import to_linear
        from cyanoneg.lut import derive_correction
        from cyanoneg.proof import BLUE_RGB, PAPER_RGB

        levels = np.linspace(0, 1, 256)
        correction = derive_correction(levels, response(levels))
        ramp = np.tile(np.linspace(0.0, 1.0, 256, dtype=np.float32), (48, 1))
        source = Image(np.stack([ramp] * 3, axis=-1), "srgb", ppi=300)

        paper = np.asarray(PAPER_RGB, dtype=np.float32)
        blue = np.asarray(BLUE_RGB, dtype=np.float32)

        def linearity_error(profile):
            negative = make_negative(source, profile, PrintSize(60, 12))
            proof = soft_proof(negative, profile)
            row = to_linear(proof.data[proof.data.shape[0] // 2], proof.space)
            lightness = ((row - blue) / (paper - blue)).mean(axis=-1)
            target = np.linspace(lightness[0], lightness[-1], len(lightness))
            return float(np.abs(lightness - target).max())

        assert linearity_error(measured_profile(lut=correction)) < linearity_error(measured_profile()) / 2
