"""Blocker models — above all, the tonal polarity.

A polarity error here inverts every print the tool makes while every other test still
passes, because the rest of the system can be self-consistently wrong. These tests are
therefore written against the *physics*, not against the implementation: they check what
happens to UV and to the paper, so an inverted pipeline cannot satisfy them.
"""

import numpy as np
import pytest

from cyanoneg.blocker import (
    apply_blocker,
    apply_zone_blocker,
    effective_blocker,
    export_blocker_cube,
    fixed_hue_as_zones,
    hue_to_rgb,
    recover_coverage,
    zone_curves,
)
from cyanoneg.imageio import Image
from cyanoneg.pipeline import PrintSize, make_negative
from cyanoneg.profiles import Profile

RED = (255, 0, 0)


def ink_of(rgb: np.ndarray) -> float:
    """How much ink a film pixel carries: white film = 0, any saturated colour = high."""
    return float(1.0 - np.min(rgb))


class TestPolarity:
    """The chain: bright positive → dense negative → UV blocked → paper stays white."""

    def test_black_negative_value_means_maximum_ink(self):
        out = apply_blocker(np.zeros((2, 2), dtype=np.float32), RED, 1.0)
        assert ink_of(out[0, 0]) == pytest.approx(1.0)
        assert tuple(np.round(out[0, 0], 3)) == (1.0, 0.0, 0.0)

    def test_white_negative_value_means_clear_film(self):
        out = apply_blocker(np.ones((2, 2), dtype=np.float32), RED, 1.0)
        assert ink_of(out[0, 0]) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "positive, expect_ink, what",
        [
            (1.0, True, "bright positive must lay ink, block UV, and print white"),
            (0.0, False, "dark positive must leave film clear so UV prints it dark"),
        ],
    )
    def test_pipeline_end_to_end_polarity(self, positive, expect_ink, what):
        profile = Profile(
            name="polarity",
            provisional=True,
            blocker={"model": "fixed_hue", "rgb": list(RED), "saturation": 1.0},
        )
        flat = Image(np.full((24, 24, 3), positive, dtype=np.float32), "srgb", ppi=300)
        negative = make_negative(flat, profile, PrintSize(10, 10))
        ink = ink_of(negative.data[12, 12])
        assert (ink > 0.5) is expect_ink, what

    def test_zone_model_shares_the_convention(self):
        zones = [{"n": 0.0, "rgb": [255, 255, 255]}, {"n": 1.0, "rgb": [255, 140, 0]}]
        dense = apply_zone_blocker(np.zeros((2, 2), dtype=np.float32), zones)
        clear = apply_zone_blocker(np.ones((2, 2), dtype=np.float32), zones)
        assert ink_of(dense[0, 0]) > 0.5
        assert ink_of(clear[0, 0]) == pytest.approx(0.0)


class TestFixedHue:
    def test_saturation_scales_toward_white(self):
        assert effective_blocker(RED, 1.0) == pytest.approx([1.0, 0.0, 0.0])
        assert effective_blocker(RED, 0.5) == pytest.approx([1.0, 0.5, 0.5])
        assert effective_blocker(RED, 0.0) == pytest.approx([1.0, 1.0, 1.0])

    def test_lower_saturation_lays_less_ink(self):
        full = apply_blocker(np.zeros((2, 2), dtype=np.float32), RED, 1.0)
        half = apply_blocker(np.zeros((2, 2), dtype=np.float32), RED, 0.5)
        assert ink_of(full[0, 0]) > ink_of(half[0, 0])

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            apply_blocker(np.zeros((2, 2, 3), dtype=np.float32), RED, 1.0)
        with pytest.raises(ValueError):
            effective_blocker(RED, 1.5)
        with pytest.raises(ValueError):
            effective_blocker((300, 0, 0), 1.0)

    def test_hue_to_rgb(self):
        assert hue_to_rgb(0) == (255, 0, 0)
        assert hue_to_rgb(120) == (0, 255, 0)
        assert hue_to_rgb(360) == hue_to_rgb(0)


class TestRecoverCoverage:
    """Recovering ink coverage from blocked RGB must invert the model exactly at any
    saturation — not just the saturation-1.0 case where `1 - min(R, G, B)` happens to work.

    This is the fix for the soft-proof generalisation gap: the CassArt profile (saturation
    1.0, a channel that reaches zero) never exposed it, but any future profile measured at
    a lower saturation would have had its proof silently understate coverage.
    """

    @pytest.mark.parametrize("saturation", [1.0, 0.75, 0.5, 0.25])
    @pytest.mark.parametrize("coverage", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_round_trips_fixed_hue_at_any_saturation(self, saturation, coverage):
        v = np.full((2, 2), 1.0 - coverage, dtype=np.float32)  # apply_blocker's v = 1 - ink
        blocked = apply_blocker(v, RED, saturation)
        recovered = recover_coverage(
            blocked, {"model": "fixed_hue", "rgb": list(RED), "saturation": saturation}
        )
        assert recovered[0, 0] == pytest.approx(coverage, abs=1e-3)

    def test_naive_min_channel_shortcut_underestimates_below_full_saturation(self):
        """The exact bug the review flagged: `1 - min(R,G,B)` degrades quietly as
        saturation drops, while recover_coverage stays exact."""
        coverage, saturation = 0.6, 0.4
        v = np.full((1, 1), 1.0 - coverage, dtype=np.float32)
        blocked = apply_blocker(v, RED, saturation)
        naive = ink_of(blocked[0])
        exact = recover_coverage(
            blocked, {"model": "fixed_hue", "rgb": list(RED), "saturation": saturation}
        )[0, 0]
        assert naive < coverage - 0.1, "the naive shortcut should visibly understate coverage here"
        assert exact == pytest.approx(coverage, abs=1e-3)

    def test_round_trips_zone_hue(self):
        zones = [
            {"n": 0.0, "rgb": [255, 255, 255]},
            {"n": 0.4, "rgb": [255, 220, 120]},
            {"n": 1.0, "rgb": [200, 20, 0]},
        ]
        for coverage in (0.0, 0.25, 0.6, 1.0):
            v = np.full((2, 2), 1.0 - coverage, dtype=np.float32)
            blocked = apply_zone_blocker(v, zones)
            recovered = recover_coverage(blocked, {"model": "zone_hue", "zones": zones})
            assert recovered[0, 0] == pytest.approx(coverage, abs=1e-2)

    def test_unsaturated_blocker_recovers_zero_everywhere(self):
        """Saturation 0 lays no ink in any channel; there is nothing to recover, not a
        divide-by-a-near-zero-range guess."""
        v = np.full((2, 2), 0.3, dtype=np.float32)
        blocked = apply_blocker(v, RED, 0.0)
        recovered = recover_coverage(
            blocked, {"model": "fixed_hue", "rgb": list(RED), "saturation": 0.0}
        )
        assert np.all(recovered == 0.0)

    def test_missing_blocker_measurement_raises(self):
        with pytest.raises(ValueError, match="measured"):
            recover_coverage(
                np.zeros((1, 1, 3), dtype=np.float32),
                {"model": "fixed_hue", "rgb": None, "saturation": None},
            )


class TestZoneModel:
    def test_reproduces_fixed_hue_exactly(self):
        """The zone model with two endpoints must equal the fixed-hue model, or the
        Phase 3 upgrade would silently change every existing profile's output."""
        v = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
        for rgb, sat in [((255, 0, 0), 1.0), ((64, 128, 0), 0.7), ((255, 191, 0), 0.5)]:
            fixed = apply_blocker(v, rgb, sat)
            zoned = apply_zone_blocker(v, fixed_hue_as_zones(rgb, sat))
            assert np.abs(fixed - zoned).max() < 1e-6

    def test_curves_are_continuous_and_monotone(self):
        zones = [
            {"n": 0.0, "rgb": [255, 255, 255]},
            {"n": 0.4, "rgb": [255, 220, 120]},
            {"n": 0.75, "rgb": [255, 140, 0]},
            {"n": 1.0, "rgb": [200, 20, 0]},
        ]
        table = zone_curves(zones)
        assert np.abs(np.diff(table, axis=0)).max() < 0.02  # no visible banding
        for channel in range(3):
            assert np.all(np.diff(table[:, channel]) <= 1e-6)  # ink only increases

    def test_implied_endpoints(self):
        """A missing n=0 point means clear film; a missing n=1 holds the densest colour."""
        table = zone_curves([{"n": 0.5, "rgb": [255, 128, 0]}])
        assert table[0] == pytest.approx([1.0, 1.0, 1.0])
        assert table[-1] == pytest.approx(table[128], abs=1e-6)

    def test_rejects_bad_zones(self):
        with pytest.raises(ValueError, match="at least one control point"):
            zone_curves([])
        with pytest.raises(ValueError, match="distinct n"):
            zone_curves([{"n": 0.5, "rgb": [1, 2, 3]}, {"n": 0.5, "rgb": [4, 5, 6]}])
        with pytest.raises(ValueError, match="within \\[0, 1\\]"):
            zone_curves([{"n": 1.5, "rgb": [1, 2, 3]}])

    def test_cube_export_structure(self, tmp_path):
        zones = [{"n": 0.0, "rgb": [255, 255, 255]}, {"n": 1.0, "rgb": [255, 140, 0]}]
        path = export_blocker_cube(tmp_path / "b.cube", zones, size=9)
        lines = path.read_text().splitlines()
        assert lines[1] == "LUT_3D_SIZE 9"
        entries = [line for line in lines[5:] if line.strip()]
        assert len(entries) == 9**3
        assert all(len(line.split()) == 3 for line in entries)
