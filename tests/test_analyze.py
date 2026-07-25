"""Phase 2: the scanned-print measurement chain, proven on simulated scans.

The headline test renders a wedge print through a known process response, degrades it the
way a real flatbed scan would (scale, rotation, perspective, blur, noise), runs the full
analysis, and asserts the derived correction inverts the response. That validates
fiducial detection, orientation recovery, the homography, sampling, normalisation and
derivation together — before any film, paper or chemistry is spent.
"""

import json

import numpy as np
import pytest

from cyanoneg.analyze import (
    AnalysisError,
    analyze_grid,
    analyze_wedge,
    apply_homography,
    detect_fiducials,
    homography,
    lightness,
)
from cyanoneg.imageio import Image, save_tiff
from cyanoneg.targets import blocker_grid, step_wedge
from simulate import _lstar_to_y, render_print, scan_of


def process_response(x):
    """The same nasty S-curve used in the Phase 1 synthetic test."""
    raw = 0.5 * (1 + np.tanh(3.5 * (np.asarray(x) - 0.45)))
    lo = 0.5 * (1 + np.tanh(3.5 * (0 - 0.45)))
    hi = 0.5 * (1 + np.tanh(3.5 * (1 - 0.45)))
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


@pytest.fixture(scope="module")
def wedge():
    return step_wedge((255, 0, 0), seed=123)


@pytest.fixture(scope="module")
def wedge_print(wedge):
    return render_print(wedge, process_response)


@pytest.fixture(scope="module")
def wedge_files(tmp_path_factory, wedge, wedge_print):
    """Default simulated scan + sidecar, on disk, shared across tests."""
    d = tmp_path_factory.mktemp("wedge")
    sidecar = d / "step_wedge.json"
    sidecar.write_text(json.dumps(wedge.sidecar))
    scan = d / "scan.tif"
    save_tiff(scan, scan_of(wedge_print))
    return scan, sidecar


class TestHomography:
    def test_recovers_known_mapping(self):
        src = np.array([[0, 0], [100, 0], [0, 60], [100, 60]], dtype=float)
        dst = np.array([[10, 5], [205, 12], [4, 130], [198, 141]], dtype=float)
        h = homography(src, dst)
        assert np.abs(apply_homography(h, src) - dst).max() < 1e-9

    def test_interior_points_follow(self):
        src = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float)
        dst = src * 2 + 7  # pure scale + translate
        h = homography(src, dst)
        mid = np.array([[50.0, 50.0], [25.0, 75.0]])
        assert np.abs(apply_homography(h, mid) - (mid * 2 + 7)).max() < 1e-9

    def test_lightness_inverse(self):
        y = np.linspace(0.001, 1.0, 500)
        assert np.abs(_lstar_to_y(lightness(y)) - y).max() < 1e-9


class TestFiducialDetection:
    def test_finds_all_four_corners(self, wedge, wedge_print):
        scan = scan_of(wedge_print)
        frame = detect_fiducials(scan, wedge.sidecar)
        assert set(frame.corners) == {"top_left", "top_right", "bottom_left", "bottom_right"}
        # Homography must map the print frame onto the scan consistently: reprojecting
        # the sidecar fiducial centres lands on the detected ones.
        for name, (x, y) in frame.corners.items():
            f = wedge.sidecar["fiducials"][name]
            src = np.array([[f["x_px"] + f["size_px"] / 2, f["y_px"] + f["size_px"] / 2]])
            proj = apply_homography(frame.h_print_to_scan, src)[0]
            assert abs(proj[0] - x) < 3 and abs(proj[1] - y) < 3

    def test_blank_scan_rejected(self, wedge, tmp_path):
        blank = Image(np.full((600, 800), 0.9, dtype=np.float32), "srgb", ppi=300)
        with pytest.raises(AnalysisError, match="tonal separation"):
            detect_fiducials(blank, wedge.sidecar)


class TestWedgeAnalysis:
    def test_round_trip_default_scan(self, wedge_files):
        scan, sidecar = wedge_files
        result = analyze_wedge(scan, sidecar)
        x = np.linspace(0, 1, 1001)
        recovered = process_response(result.lut.apply(x))
        assert np.abs(recovered[50:951] - x[50:951]).max() < 0.02
        assert not result.spikes
        # DR of the simulated print (L* 93 → 22) is ~1.37 and must be reported as such.
        assert 1.2 <= result.density_range <= 1.45
        assert not result.warnings

    @pytest.mark.parametrize("orientation", ["rot90", "rot180", "rot270", "mirror"])
    def test_any_scan_orientation_recovered(self, wedge, wedge_print, tmp_path, orientation):
        sidecar = tmp_path / "sc.json"
        sidecar.write_text(json.dumps(wedge.sidecar))
        scan = tmp_path / f"{orientation}.tif"
        save_tiff(scan, scan_of(wedge_print, orientation=orientation))
        result = analyze_wedge(scan, sidecar)
        x = np.linspace(0, 1, 1001)
        recovered = process_response(result.lut.apply(x))
        assert np.abs(recovered[50:951] - x[50:951]).max() < 0.02

    def test_ink_spike_flagged_and_suppressed(self, wedge, tmp_path):
        """Corrupt one copy of one level; the flag must fire and the redundant copy
        must keep the curve on track."""
        corrupted_level = 128
        printed = render_print(wedge, process_response)
        victim = next(c for c in wedge.sidecar["cells"] if c["value"] == corrupted_level)
        printed[
            victim["y_px"] : victim["y_px"] + victim["h_px"],
            victim["x_px"] : victim["x_px"] + victim["w_px"],
        ] = _lstar_to_y(np.array([30.0]))[0]  # way too dark for its level

        sidecar = tmp_path / "sc.json"
        sidecar.write_text(json.dumps(wedge.sidecar))
        scan = tmp_path / "spiked.tif"
        save_tiff(scan, scan_of(printed))
        result = analyze_wedge(scan, sidecar)
        assert any(s["value"] == corrupted_level for s in result.spikes)
        # The spike averaged with its clean copy still cannot wreck the curve region.
        x = np.linspace(0, 1, 1001)
        recovered = process_response(result.lut.apply(x))
        assert np.abs(recovered[50:951] - x[50:951]).max() < 0.06

    def test_low_dr_print_warns(self, wedge, tmp_path):
        """A weak print (short tonal range) must warn, not silently calibrate."""
        import simulate

        printed = render_print(wedge, lambda v: process_response(v) * 0.55)  # feeble max black
        sidecar = tmp_path / "sc.json"
        sidecar.write_text(json.dumps(wedge.sidecar))
        scan = tmp_path / "weak.tif"
        save_tiff(scan, scan_of(printed))
        result = analyze_wedge(scan, sidecar)
        assert result.density_range < 1.2
        assert any("below" in w for w in result.warnings)

    def test_wrong_sidecar_rejected(self, wedge_files, tmp_path):
        scan, _ = wedge_files
        grid_sidecar = tmp_path / "grid.json"
        grid_sidecar.write_text(json.dumps(blocker_grid().sidecar))
        with pytest.raises(AnalysisError, match="not a step-wedge sidecar"):
            analyze_wedge(scan, grid_sidecar)


BEST_HUE = 45


def _grid_cell_response(cell):
    if "ref" in cell:
        return 1.0 if cell["ref"] == "clear" else 0.35
    gap = abs(cell["hue_deg"] - BEST_HUE) / 150
    return float(np.clip(1 - (1 - 0.8 * gap) * cell["saturation"] ** 1.5, 0, 1)) * 0.9


@pytest.fixture(scope="module")
def grid_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("grid")
    grid = blocker_grid()
    sidecar = d / "blocker_grid.json"
    sidecar.write_text(json.dumps(grid.sidecar))
    scan = d / "scan.tif"
    save_tiff(scan, scan_of(render_print(grid, cell_response=_grid_cell_response), rotate_deg=-1.0))
    return scan, sidecar


class TestGridAnalysis:
    def test_identifies_best_hue(self, grid_files):
        scan, sidecar = grid_files
        result = analyze_grid(scan, sidecar)
        assert result.best["hue_deg"] == BEST_HUE
        assert result.best["saturation"] == 1.0
        assert result.recommended_saturation is not None

    def test_ranking_is_sorted_and_references_read(self, grid_files):
        scan, sidecar = grid_files
        result = analyze_grid(scan, sidecar)
        lstars = [c["lstar"] for c in result.ranking]
        assert lstars == sorted(lstars, reverse=True)
        refs = {r["ref"] for r in result.references}
        assert refs == {"clear", "black"}
