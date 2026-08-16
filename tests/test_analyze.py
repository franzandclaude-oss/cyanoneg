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
    QUANTITIES,
    RESPONSE_CHANNEL,
    analyze_grid,
    analyze_wedge,
    analyze_zone_grid,
    apply_homography,
    detect_fiducials,
    homography,
    lightness,
    sample_cells,
)
from cyanoneg.imageio import Image, save_tiff
from cyanoneg.targets import blocker_grid, step_wedge, zone_blocker_grid
from simulate import _lstar_to_y, render_print, scan_of


def process_response(x):
    """Input level → normalised print lightness. The nasty S-curve from Phase 1.

    Increasing: input 0 is clear film (max black), input 1 lays maximum ink (paper white).
    """
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

    def test_patches_record_their_colour(self, wedge, wedge_print, tmp_path):
        """Every patch keeps its own linear RGB, not just its lightness.

        The tonal analysis never looks at it — it works in luminance throughout — but the
        soft proof does. Prussian blue fades towards paper along a curve that no blend of
        two endpoints reproduces, and modelling it produced a proof that was tonally
        correct and looked grey. Measuring it here is what removed the model.
        """
        sidecar = tmp_path / "sc.json"
        sidecar.write_text(json.dumps(wedge.sidecar))
        scan = tmp_path / "colour.tif"
        save_tiff(scan, scan_of(wedge_print, colour=True))
        result = analyze_wedge(scan, sidecar)

        assert all("rgb" in p for p in result.raw_patches), "colour must be recorded per patch"
        assert all(len(p["rgb"]) == 3 for p in result.raw_patches)
        by_value = sorted(result.raw_patches, key=lambda p: p["value"])
        dark, light = np.array(by_value[0]["rgb"]), np.array(by_value[-1]["rgb"])
        assert dark[2] - dark[0] > 0.05, "the dark end must read blue, not neutral"
        assert light[2] - light[0] < dark[2] - dark[0], "paper must be less blue than Dmax"

    def test_a_colour_scan_measures_the_same_tones_as_a_mono_one(self, wedge, wedge_print, tmp_path):
        """Recording colour must not disturb the tonal result, or it buys one thing by
        quietly breaking another. The tint is luminance-preserving by construction; this
        checks that end to end rather than trusting it."""
        sidecar = tmp_path / "sc.json"
        sidecar.write_text(json.dumps(wedge.sidecar))
        results = []
        for name, colour in (("mono.tif", False), ("colour.tif", True)):
            path = tmp_path / name
            save_tiff(path, scan_of(wedge_print, colour=colour))
            results.append(analyze_wedge(path, sidecar))
        mono, tinted = results
        assert np.abs(mono.lut.values - tinted.lut.values).max() < 0.01
        assert abs(mono.density_range - tinted.density_range) < 0.02

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
        corrupted_level = wedge.sidecar["levels"] // 2  # a midtone, whatever the design
        printed = render_print(wedge, process_response)
        victim = next(c for c in wedge.sidecar["cells"] if c["value"] == corrupted_level)
        printed[
            victim["y_px"] : victim["y_px"] + victim["h_px"],
            victim["x_px"] : victim["x_px"] + victim["w_px"],
        ] = _lstar_to_y(np.array([88.0]))[0]  # way too light for its level

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
        # A feeble print: compressed toward mid-grey at both ends.
        printed = render_print(wedge, lambda v: 0.25 + process_response(v) * 0.5)
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
    """Simulated print lightness: better UV blocking → whiter paper → higher value."""
    if "ref" in cell:
        return 0.0 if cell["ref"] == "clear" else 0.65  # clear film prints max black
    gap = abs(cell["hue_deg"] - BEST_HUE) / 150
    blocking = (1 - 0.8 * gap) * cell["saturation"] ** 1.5
    return float(np.clip(blocking, 0, 1)) * 0.95


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


# --------------------------------------------------------------------------- zone grid


def _drifting_hue(n):
    """A process whose best blocking hue moves with density: 90° at the top, 30° at max."""
    return 90 - 60 * n


def _zone_cell_response(cell):
    """Simulated print lightness — better blocking → whiter paper."""
    n = cell["zone_n"]
    gap = abs(cell["hue_deg"] - _drifting_hue(n)) / 150
    return float(np.clip((1 - 0.75 * gap) * n**0.7, 0, 1)) * 0.95


@pytest.fixture(scope="module")
def zone_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("zone")
    grid = zone_blocker_grid()
    sidecar = d / "zone_grid.json"
    sidecar.write_text(json.dumps(grid.sidecar))
    scan = d / "scan.tif"
    save_tiff(scan, scan_of(render_print(grid, cell_response=_zone_cell_response), rotate_deg=2.0))
    return scan, sidecar, grid


class TestZoneGridAnalysis:
    def test_recovers_the_hue_drift(self, zone_files):
        scan, sidecar, grid = zone_files
        result = analyze_zone_grid(scan, sidecar)
        assert len(result.zones) == len(grid.sidecar["zones"])
        step = 15  # the sweep's hue resolution — recovery cannot beat it
        for zone in result.zones:
            assert abs(zone["hue_deg"] - _drifting_hue(zone["n"])) <= step

    def test_reports_that_the_zone_model_is_justified(self, zone_files):
        scan, sidecar, _ = zone_files
        result = analyze_zone_grid(scan, sidecar)
        assert result.hue_varies
        assert "justified" in result.summary()

    def test_control_points_are_usable_as_a_profile_blocker(self, zone_files):
        """The analysis output must drop straight into a profile and validate."""
        from cyanoneg.profiles import Profile

        scan, sidecar, _ = zone_files
        points = analyze_zone_grid(scan, sidecar).control_points()
        profile = Profile(
            name="zoned",
            provisional=False,
            blocker={"model": "zone_hue", "zones": points},
        )
        assert profile.validate() == []
        assert profile.is_ready_to_print
        assert points[0]["n"] == 0.0 and points[0]["rgb"] == [255, 255, 255]

    def test_single_best_hue_reports_no_justification(self, tmp_path):
        """If one hue wins everywhere, say so — PLAN.md only licenses the upgrade when
        measurements justify it."""
        grid = zone_blocker_grid()
        sidecar = tmp_path / "z.json"
        sidecar.write_text(json.dumps(grid.sidecar))
        scan = tmp_path / "flat.tif"

        def fixed_best(cell):
            gap = abs(cell["hue_deg"] - 45) / 150
            return float(np.clip((1 - 0.75 * gap) * cell["zone_n"] ** 0.7, 0, 1)) * 0.95

        save_tiff(scan, scan_of(render_print(grid, cell_response=fixed_best)))
        result = analyze_zone_grid(scan, sidecar)
        assert not result.hue_varies
        assert "fixed-hue" in result.summary()

    def test_wrong_sidecar_rejected(self, zone_files, tmp_path):
        scan, _, _ = zone_files
        wedge_sidecar = tmp_path / "w.json"
        wedge_sidecar.write_text(json.dumps(step_wedge((255, 0, 0)).sidecar))
        with pytest.raises(AnalysisError, match="not a zone-grid sidecar"):
            analyze_zone_grid(scan, wedge_sidecar)


class TestOutlierRejection:
    """Spike detection must not get stricter just because there are more copies.

    A max-minus-min spread test grows with sample count on perfectly clean data. Tuned at
    k=2 it fired on 31 of 32 levels once the wedge moved to k=16, on a good print — the
    warning became noise at exactly the moment the redundancy made real rejection possible.
    """

    def _sidecar(self, tmp_path, wedge):
        p = tmp_path / "sc.json"
        p.write_text(json.dumps(wedge.sidecar))
        return p

    def test_clean_print_at_high_redundancy_is_quiet(self, tmp_path):
        wedge = step_wedge((255, 0, 0), levels=32, redundancy=16, seed=7)
        printed = render_print(wedge, process_response)
        scan = tmp_path / "clean.tif"
        save_tiff(scan, scan_of(printed))
        result = analyze_wedge(scan, self._sidecar(tmp_path, wedge))
        assert len(result.spikes) <= 2, f"clean print flagged {len(result.spikes)} levels"

    def test_single_bad_patch_is_identified_and_dropped(self, tmp_path):
        """With 16 siblings the analysis can say which copy is wrong, not just that one is."""
        wedge = step_wedge((255, 0, 0), levels=32, redundancy=16, seed=7)
        printed = render_print(wedge, process_response)
        victim = next(c for c in wedge.sidecar["cells"] if c["value"] == 16)
        printed[
            victim["y_px"] : victim["y_px"] + victim["h_px"],
            victim["x_px"] : victim["x_px"] + victim["w_px"],
        ] = _lstar_to_y(np.array([88.0]))[0]

        scan = tmp_path / "spiked.tif"
        save_tiff(scan, scan_of(printed))
        result = analyze_wedge(scan, self._sidecar(tmp_path, wedge))

        flagged = [s for s in result.spikes if s["value"] == 16]
        assert flagged, "the corrupted patch was not flagged"
        assert flagged[0]["rejected"] >= 1
        assert (victim["row"], victim["col"]) in flagged[0]["positions"]

        # And the curve is unharmed, because the bad copy never entered the mean.
        x = np.linspace(0, 1, 1001)
        recovered = process_response(result.lut.apply(x))
        assert np.abs(recovered[50:951] - x[50:951]).max() < 0.06

    def test_two_copies_flag_without_rejecting(self, tmp_path):
        """At k=2 neither copy can be shown to be the bad one, so both must survive."""
        wedge = step_wedge((255, 0, 0), levels=64, redundancy=2, seed=7)
        printed = render_print(wedge, process_response)
        victim = next(c for c in wedge.sidecar["cells"] if c["value"] == 32)
        printed[
            victim["y_px"] : victim["y_px"] + victim["h_px"],
            victim["x_px"] : victim["x_px"] + victim["w_px"],
        ] = _lstar_to_y(np.array([88.0]))[0]

        scan = tmp_path / "spiked2.tif"
        save_tiff(scan, scan_of(printed))
        result = analyze_wedge(scan, self._sidecar(tmp_path, wedge))
        flagged = [s for s in result.spikes if s["value"] == 32]
        assert flagged and flagged[0]["rejected"] == 0


class TestFiducialSizeFilter:
    """Shape alone cannot tell a fiducial from a dark wedge patch — both are dark squares.

    Which blobs clear the dark threshold depends on how open the print's shadows are, so
    the same sheet scanned two ways gave 60 candidates once and 82 the other time; in the
    second the corner picks landed on patches and detection failed with "no hollow fiducial
    found" while the fiducial sat plainly in the scan. The sidecar has always recorded how
    big a fiducial should be; the detector just never used it.
    """

    def test_open_shadow_print_still_detects(self, wedge, tmp_path):
        """A print with lifted shadows puts many patches into the dark mask."""
        printed = render_print(wedge, lambda v: 0.30 + process_response(v) * 0.70)
        sidecar = tmp_path / "sc.json"
        sidecar.write_text(json.dumps(wedge.sidecar))
        scan = tmp_path / "open.tif"
        save_tiff(scan, scan_of(printed))
        frame = detect_fiducials(  # must not raise
            __import__("cyanoneg.imageio", fromlist=["load_image"]).load_image(scan),
            wedge.sidecar,
        )
        assert set(frame.corners) == {"top_left", "top_right", "bottom_left", "bottom_right"}

    def test_size_filter_rejects_patch_sized_blobs(self, wedge):
        """The band must sit below the patch-to-fiducial size ratio, or it filters nothing.

        This is the assertion that matters. The first band tried was 1.8x while patches are
        1.39x a fiducial, so it admitted precisely the blobs it existed to exclude.
        """
        from cyanoneg.analyze import _FID_SIZE_BAND, _expected_fiducial_size

        fid_px = wedge.sidecar["fiducials"]["top_left"]["size_px"]
        patch_px = wedge.sidecar["cells"][0]["w_px"]
        assert fid_px != patch_px, "the test is vacuous if they happen to match"
        assert _FID_SIZE_BAND[1] < patch_px / fid_px, "band admits patch-sized blobs"

        class FakeScan:
            ppi = 300.0

        expected = _expected_fiducial_size(FakeScan(), wedge.sidecar, fid_px, scale=2, candidates=[])
        assert expected == pytest.approx(fid_px * (300.0 / wedge.sidecar["ppi"]) / 2)
        patch_size = patch_px * (300.0 / wedge.sidecar["ppi"]) / 2
        assert not (_FID_SIZE_BAND[0] * expected <= patch_size <= _FID_SIZE_BAND[1] * expected)

    def test_falls_back_when_the_scan_has_no_ppi(self, wedge):
        """Untagged scans still work: the fiducials are the biggest same-sized cluster."""
        from cyanoneg.analyze import _expected_fiducial_size

        class Blobby:
            def __init__(self, size):
                self._s = size

            @property
            def bbox(self):
                return (0, 0, self._s - 1, self._s - 1)

        class NoPPI:
            ppi = None

        cands = [Blobby(24) for _ in range(4)] + [Blobby(9), Blobby(11), Blobby(35)]
        got = _expected_fiducial_size(NoPPI(), {"ppi": None}, 56, 2, cands)
        assert got == pytest.approx(24, abs=1.5)


class TestResponseQuantity:
    """``sample_cells`` accepted a ``quantity`` argument and ignored it for the whole of
    phase 2 — flagged as V1 in the tricolour second review. Tricolour needs it live: a
    madder-toned magenta layer read in luminance has its response diluted by two colorants
    it never laid."""

    def _wedge_scan(self, colour: bool):
        wedge = step_wedge((255, 64, 0), saturation=1.0, levels=8, redundancy=2)
        y = render_print(wedge, process_response)
        return wedge, scan_of(y, colour=colour, scale=1.0, rotate_deg=0.0, perspective=0.0,
                              noise=0.0, blur_px=0.4)

    def test_default_is_neutral_and_costs_nothing(self):
        """At the default, ``response`` *is* ``lstar`` — the mono path cannot have moved."""
        wedge, scan = self._wedge_scan(colour=False)
        frame = detect_fiducials(scan, wedge.sidecar)
        for s in sample_cells(scan, wedge.sidecar, frame):
            assert s["response"] == s["lstar"]
            assert s["response_linear"] == s["y_linear"]
            assert s["quantity"] == "lstar"

    def test_a_channel_reads_differently_from_luminance(self):
        """Liveness. If the argument were still ignored these would be equal."""
        wedge, scan = self._wedge_scan(colour=True)
        frame = detect_fiducials(scan, wedge.sidecar)
        neutral = sample_cells(scan, wedge.sidecar, frame)
        green = sample_cells(scan, wedge.sidecar, frame, quantity="lstar_g")
        assert all(g["lstar"] == n["lstar"] for g, n in zip(green, neutral)), (
            "the neutral pair must survive unchanged alongside the per-channel one"
        )
        assert any(abs(g["response"] - n["response"]) > 0.5 for g, n in zip(green, neutral))

    def test_every_named_quantity_resolves(self):
        wedge, scan = self._wedge_scan(colour=True)
        frame = detect_fiducials(scan, wedge.sidecar)
        for q in QUANTITIES:
            got = sample_cells(scan, wedge.sidecar, frame, quantity=q)
            assert all(c["quantity"] == q for c in got)
        assert set(RESPONSE_CHANNEL) < set(QUANTITIES)

    def test_unknown_quantity_is_refused(self):
        wedge, scan = self._wedge_scan(colour=True)
        frame = detect_fiducials(scan, wedge.sidecar)
        with pytest.raises(AnalysisError, match="unknown quantity"):
            sample_cells(scan, wedge.sidecar, frame, quantity="lstar_k")

    def test_a_channel_cannot_be_read_from_a_mono_scan(self):
        """The failure a tricolour user would otherwise hit as a silent neutral reading."""
        wedge, mono = self._wedge_scan(colour=False)
        assert mono.data.ndim == 2, "this test is only meaningful on a single-channel scan"
        frame = detect_fiducials(mono, wedge.sidecar)
        with pytest.raises(AnalysisError, match="needs a colour scan"):
            sample_cells(mono, wedge.sidecar, frame, quantity="lstar_g")

    def test_the_density_window_is_not_applied_to_a_bleached_layer(self, tmp_path):
        """DR_WINDOW is a Prussian-blue number, measured in luminance on an untoned print.

        Yellow is a pale Fe(OH)3 ochre read through blue; its range has no reason to land
        in 1.2-1.4, so grading it against that window fires on every yellow analysis. A
        warning that is always wrong is worse than none — it trains the reader to skip the
        ones that are right. The range is still computed and still reported; it is simply
        not graded against a window that does not describe it.
        """
        wedge, scan = self._wedge_scan(colour=True)
        side = tmp_path / "w.json"
        side.write_text(json.dumps(wedge.sidecar), encoding="utf-8")
        path = save_tiff(tmp_path / "scan.tif", scan)

        blue = analyze_wedge(path, side, quantity="lstar_b")
        assert not any("below the" in w or "above the" in w for w in blue.warnings), (
            "a bleached layer was graded against the Prussian-blue window"
        )
        assert any("not applied here" in w for w in blue.warnings)
        assert any(f"{blue.density_range:.2f}" in w for w in blue.warnings), (
            "the range must still be reported, not silently dropped"
        )

    def test_the_window_still_applies_to_neutral_readings(self):
        """The gate must not disarm the check for the mono path it was written for."""
        # A real print with real separation between its references, but shadows that never
        # reach black — a low density range, which is what the window is there to catch.
        wedge = step_wedge((255, 64, 0), saturation=1.0, levels=8, redundancy=2)
        weak = render_print(wedge, lambda x: 0.35 + 0.65 * np.asarray(x))
        scan = scan_of(weak, scale=1.0, rotate_deg=0.0, perspective=0.0, noise=0.0,
                       blur_px=0.4)
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "w.json").write_text(json.dumps(wedge.sidecar), encoding="utf-8")
            got = analyze_wedge(save_tiff(d / "s.tif", scan), d / "w.json")
        assert any("window" in w for w in got.warnings)
        assert not any("not applied here" in w for w in got.warnings)

    def test_analyze_wedge_records_what_it_measured(self, tmp_path):
        """A curve read through green is not comparable to one read in luminance."""
        wedge, scan = self._wedge_scan(colour=True)
        side = tmp_path / "w.json"
        side.write_text(json.dumps(wedge.sidecar), encoding="utf-8")
        path = save_tiff(tmp_path / "scan.tif", scan)

        neutral = analyze_wedge(path, side)
        green = analyze_wedge(path, side, quantity="lstar_g")
        assert neutral.quantity == "lstar"
        assert green.quantity == "lstar_g"
        assert "lstar_g" in green.summary()
        assert not np.allclose(neutral.measured_out, green.measured_out)
