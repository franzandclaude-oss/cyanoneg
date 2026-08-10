"""Soft proof: it must predict from measurement, and refuse without it."""

import numpy as np
import pytest

from cyanoneg.imageio import Image
from cyanoneg.lut import Lut
from cyanoneg.pipeline import PrintSize, make_negative
from cyanoneg.profiles import PROFILE_DIR, Profile
from cyanoneg.proof import (
    ProofUnavailable,
    can_proof,
    measured_endpoints,
    measured_response,
    soft_proof,
)

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
        profile = Profile.load(PROFILE_DIR / "CassArt 300 Sm.json")
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


class TestPredictedLightness:
    """The proof must land tones where the measurement says, not merely in the right order.

    Every other test in this file compares the proof to itself — is it monotonic, are
    shadows bluer than highlights, does the correction flatten it. All of those passed
    while the proof was blending linear reflectance with an L* fraction, which put mid
    tones about 14 L* too light. It took two prints on paper to notice. These tests anchor
    the proof to an absolute quantity instead, which is what makes the error visible
    without spending film.
    """

    def ramp_negative(self, profile):
        """A negative whose ink coverage sweeps 0 → 1, so the proof's whole range is seen."""
        ink = np.tile(np.linspace(0.0, 1.0, 256, dtype=np.float32), (8, 1))
        return Image(np.stack([1.0 - ink] * 3, axis=-1), "srgb", ppi=300)

    def proofed_lstar(self, profile):
        from cyanoneg.imageio import to_linear

        proof = soft_proof(self.ramp_negative(profile), profile)
        row = to_linear(proof.data[proof.data.shape[0] // 2], proof.space)
        y = np.clip(row @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), 0, 1)
        return np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)[::-1]  # un-mirror

    def linear_profile(self):
        """Patches whose lightness rises linearly with value, so the response is identity."""
        patches = [
            {"value": v, "lstar": BLACK_LSTAR + (v / 255) * (PAPER_LSTAR - BLACK_LSTAR)}
            for v in range(256)
        ]
        return measured_profile(measurements={"raw_patches": patches})

    def test_lightness_is_linear_in_the_measured_response(self):
        """With an identity response, predicted L* must be a straight line.

        This is the assertion the old blend fails outright: interpolating reflectance by an
        L* fraction bows the line by ~14 L* in the middle while still hitting both ends,
        which is exactly why the endpoints looked right on paper and the midtones did not.
        """
        lstar = self.proofed_lstar(self.linear_profile())
        straight = np.linspace(lstar[0], lstar[-1], len(lstar))
        assert np.abs(lstar - straight).max() < 1.5

    def test_the_ends_are_the_papers_measured_ends(self):
        """Dmax and paper white come from the patches, not from the Prussian blue constant.

        The constant is L* 27; hand-coated paper measured L* 32-38. Proofing against the
        constant makes every shadow look deeper than the paper can actually print.
        """
        profile = self.linear_profile()
        dmax, paper = measured_endpoints(profile)
        assert (dmax, paper) == pytest.approx((BLACK_LSTAR, PAPER_LSTAR), abs=0.1)
        lstar = self.proofed_lstar(profile)
        assert lstar[0] == pytest.approx(BLACK_LSTAR, abs=1.0)
        assert lstar[-1] == pytest.approx(PAPER_LSTAR, abs=1.0)

    def test_a_darker_paper_proofs_darker(self):
        """Two papers differing only in Dmax must proof differently all the way up."""
        deep = [{"value": v, "lstar": 15.0 + (v / 255) * (PAPER_LSTAR - 15.0)} for v in range(256)]
        shallow = [{"value": v, "lstar": 45.0 + (v / 255) * (PAPER_LSTAR - 45.0)} for v in range(256)]
        a = self.proofed_lstar(measured_profile(measurements={"raw_patches": deep}))
        b = self.proofed_lstar(measured_profile(measurements={"raw_patches": shallow}))
        assert (b[:-8] > a[:-8]).all(), "the shallower paper must proof lighter throughout"

    def test_endpoints_need_measurements_too(self):
        with pytest.raises(ProofUnavailable):
            measured_endpoints(Profile.load(PROFILE_DIR / "linear.json"))


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

        Measured as L*, which is what "linear" means for a print someone looks at, and how
        the real prints were checked against this prediction. This test used to recover
        lightness as a fraction of the way from Prussian blue to paper *in reflectance* —
        which was the proof's own broken assumption, so the test could never have
        disagreed with it.
        """
        from cyanoneg.imageio import to_linear
        from cyanoneg.lut import derive_correction

        levels = np.linspace(0, 1, 256)
        correction = derive_correction(levels, response(levels))
        ramp = np.tile(np.linspace(0.0, 1.0, 256, dtype=np.float32), (48, 1))
        source = Image(np.stack([ramp] * 3, axis=-1), "srgb", ppi=300)
        luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

        def linearity_error(profile):
            negative = make_negative(source, profile, PrintSize(60, 12))
            proof = soft_proof(negative, profile)
            row = to_linear(proof.data[proof.data.shape[0] // 2], proof.space)
            y = np.clip(row @ luma, 0.0, 1.0)
            lstar = np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)
            target = np.linspace(lstar[0], lstar[-1], len(lstar))
            return float(np.abs(lstar - target).max())

        assert linearity_error(measured_profile(lut=correction)) < linearity_error(measured_profile()) / 2
