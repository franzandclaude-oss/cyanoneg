"""LUT mathematics — including the synthetic calibration round-trip, the key test of the
whole project: it proves derive_correction inverts a measured process response without
consuming film, paper or chemistry."""

import struct

import numpy as np
import pytest

from cyanoneg.lut import CUBE_SIZE, Lut, derive_correction, pchip_eval


class TestLutBasics:
    def test_identity_is_noop(self):
        a = np.random.default_rng(0).random(10_000).astype(np.float32)
        assert np.array_equal(Lut.identity().apply(a), a)

    def test_identity_detection(self):
        assert Lut.identity().is_identity()
        assert not Lut(np.linspace(0, 1, 256) ** 1.01).is_identity()

    def test_apply_clips_input(self):
        lut = Lut.identity()
        assert lut.apply(np.array([-0.5, 1.5], dtype=np.float32)) == pytest.approx([0.0, 1.0])

    def test_enforce_monotonic(self):
        noisy = Lut(np.linspace(0, 1, 256) + np.random.default_rng(1).normal(0, 0.05, 256))
        fixed = noisy.enforce_monotonic()
        assert np.all(np.diff(fixed.values) >= 0)
        assert fixed.values.min() >= 0 and fixed.values.max() <= 1

    def test_rejects_scalar_and_short_tables(self):
        with pytest.raises(ValueError):
            Lut(np.array([0.5]))


class TestPchip:
    def test_interpolates_knots_exactly(self):
        x = np.array([0.0, 0.3, 0.7, 1.0])
        y = np.array([0.0, 0.5, 0.8, 1.0])
        assert pchip_eval(x, y, x) == pytest.approx(y)

    def test_monotone_data_gives_monotone_curve(self):
        """The reason for PCHIP over a natural cubic: no overshoot between knots."""
        x = np.array([0.0, 0.1, 0.15, 0.9, 1.0])
        y = np.array([0.0, 0.05, 0.85, 0.9, 1.0])  # steep step that would ring a cubic
        q = pchip_eval(x, y, np.linspace(0, 1, 1000))
        assert np.all(np.diff(q) >= -1e-12)
        assert q.min() >= 0 and q.max() <= 1

    def test_clamps_rather_than_extrapolates(self):
        x = np.array([0.2, 0.8])
        y = np.array([0.3, 0.7])
        assert pchip_eval(x, y, np.array([0.0, 1.0])) == pytest.approx([0.3, 0.7])

    def test_rejects_unsorted_knots(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            pchip_eval(np.array([0.0, 0.5, 0.4]), np.zeros(3), np.array([0.1]))


def _process_response(x: np.ndarray) -> np.ndarray:
    """A plausible nasty print response: S-curve with a long toe and shoulder."""
    raw = 0.5 * (1 + np.tanh(3.5 * (x - 0.45)))
    lo, hi = 0.5 * (1 + np.tanh(3.5 * (0 - 0.45))), 0.5 * (1 + np.tanh(3.5 * (1 - 0.45)))
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


class TestSyntheticRoundTrip:
    """Generate a wedge, push it through a known non-linear response, derive the
    correction, and assert (correction ∘ process) ≈ identity."""

    def test_clean_measurements(self):
        patches = np.linspace(0, 1, 256)
        corr = derive_correction(patches, _process_response(patches))
        x = np.linspace(0, 1, 1001)
        recovered = _process_response(corr.apply(x))
        interior = slice(50, 951)  # extremes are where the response itself clips
        assert np.abs(recovered[interior] - x[interior]).max() < 0.01
        assert np.abs(recovered - x).mean() < 0.005

    def test_noisy_measurements(self):
        rng = np.random.default_rng(42)
        patches = np.linspace(0, 1, 256)
        measured = np.clip(_process_response(patches) + rng.normal(0, 0.01, 256), 0, 1)
        corr = derive_correction(patches, measured)
        x = np.linspace(0, 1, 1001)
        recovered = _process_response(corr.apply(x))
        assert np.abs(recovered[50:951] - x[50:951]).max() < 0.02

    def test_correction_is_monotone(self):
        patches = np.linspace(0, 1, 256)
        corr = derive_correction(patches, _process_response(patches))
        assert np.all(np.diff(corr.values) >= 0)

    def test_linear_process_needs_no_correction(self):
        patches = np.linspace(0, 1, 256)
        corr = derive_correction(patches, patches)
        assert np.abs(corr.values - np.linspace(0, 1, 256)).max() < 0.005

    def test_constant_response_rejected(self):
        with pytest.raises(ValueError, match="carries no information"):
            derive_correction(np.linspace(0, 1, 10), np.full(10, 0.5))

    def test_randomised_patch_order_is_equivalent(self):
        """The wedge is randomised on film; derivation must not depend on patch order."""
        rng = np.random.default_rng(7)
        patches = np.linspace(0, 1, 256)
        measured = _process_response(patches)
        order = rng.permutation(256)
        a = derive_correction(patches, measured)
        b = derive_correction(patches[order], measured[order])
        assert np.abs(a.values - b.values).max() < 1e-9


class TestExport:
    def test_acv_structure(self, tmp_path):
        lut = Lut(np.linspace(0, 1, 256) ** 0.8)
        path = lut.export_acv(tmp_path / "c.acv")
        raw = path.read_bytes()
        version, curve_count = struct.unpack(">hh", raw[:4])
        (points,) = struct.unpack(">h", raw[4:6])
        assert (version, curve_count, points) == (4, 1, 16)
        assert len(raw) == 6 + points * 4
        pairs = struct.unpack(f">{points * 2}h", raw[6:])
        outputs, inputs = pairs[0::2], pairs[1::2]
        assert all(0 <= v <= 255 for v in pairs)
        assert list(inputs) == sorted(inputs)  # Photoshop requires ascending inputs
        assert inputs[0] == 0 and inputs[-1] == 255

    def test_acv_point_count_limits(self, tmp_path):
        with pytest.raises(ValueError):
            Lut.identity().export_acv(tmp_path / "x.acv", points=20)

    def cube_rows(self, path):
        lines = path.read_text().splitlines()
        header = {line.split()[0]: line for line in lines if line and not line[0].isdigit()}
        rows = np.array([[float(v) for v in line.split()] for line in lines if line and line[0].isdigit()])
        return header, rows

    def test_cube_is_three_dimensional(self, tmp_path):
        """Photoshop's Color Lookup reads 3D LUTs only.

        A tone curve is one-dimensional and the .cube format does define LUT_1D_SIZE for
        it — this wrote that for months, and Photoshop rejected the file outright. Steven
        found it by trying to open one.
        """
        header, rows = self.cube_rows(Lut.identity(256).export_cube(tmp_path / "c.cube", size=8))
        assert header["LUT_3D_SIZE"] == "LUT_3D_SIZE 8"
        assert "LUT_1D_SIZE" not in header
        assert len(rows) == 8**3

    def test_cube_entries_are_ordered_red_fastest(self, tmp_path):
        """Get the axis order wrong and the file loads, then maps colours to nonsense."""
        _, rows = self.cube_rows(Lut.identity(256).export_cube(tmp_path / "c.cube", size=4))
        step = 1 / 3
        assert rows[1] == pytest.approx([step, 0, 0], abs=1e-6), "red must vary fastest"
        assert rows[4] == pytest.approx([0, step, 0], abs=1e-6), "green next"
        assert rows[16] == pytest.approx([0, 0, step], abs=1e-6), "blue slowest"

    def test_cube_carries_the_curve_on_every_axis(self, tmp_path):
        curve = Lut(np.linspace(0, 1, 256) ** 2.0)
        _, rows = self.cube_rows(curve.export_cube(tmp_path / "c.cube", size=16))
        axis = np.linspace(0, 1, 16)
        expected = curve.apply(axis)
        assert rows[:16, 0] == pytest.approx(expected, abs=1e-5)  # red sweep
        assert rows[::16 * 16, 2] == pytest.approx(expected, abs=1e-5)  # blue sweep

    def test_cube_grid_resolves_the_shipped_curve(self, tmp_path):
        """The default size has to survive the shape a real correction actually takes.

        The measured cyanotype curve lifts input 17 to output 81 — nearly vertical at the
        foot, which is what makes the grid size matter. A 16-point cube misses it by 8
        code values and 32 by 2; the default must stay under one. Tested against the
        shipped profile rather than an invented curve, so it tracks whatever is really
        being exported.
        """
        from cyanoneg.profiles import PROFILE_DIR, Profile

        curve = Profile.load(PROFILE_DIR / "CassArt 300 Sm.json").lut
        _, rows = self.cube_rows(curve.export_cube(tmp_path / "c.cube"))
        size = round(len(rows) ** (1 / 3))
        assert size == CUBE_SIZE
        fine = np.linspace(0, 1, 2001)
        grid = np.linspace(0, 1, size)
        error = np.abs(np.interp(fine, grid, curve.apply(grid)) - curve.apply(fine)).max()
        assert error * 255 < 1.0, f"{error * 255:.2f} code values at size {size}"

    def test_cube_size_limits(self, tmp_path):
        for bad in (1, 257):
            with pytest.raises(ValueError, match="between 2 and 256"):
                Lut.identity().export_cube(tmp_path / "x.cube", size=bad)
