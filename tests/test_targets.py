"""Target geometry and sidecar integrity, plus the real-chart parse."""

import json

import numpy as np
import pytest

from cyanoneg.blocker import apply_blocker
from cyanoneg.imageio import load_image
from cyanoneg.targets import PPI, blocker_grid, exposure_strip, step_wedge


@pytest.fixture(scope="module")
def wedge():
    return step_wedge((255, 0, 0), saturation=1.0, seed=123)


class TestStepWedge:

    def test_covers_all_levels_with_redundancy(self, wedge):
        values = [c["value"] for c in wedge.sidecar["cells"]]
        assert len(values) == 512
        assert sorted(set(values)) == list(range(256))
        assert all(values.count(v) == 2 for v in range(256))

    def test_randomised_not_sequential(self, wedge):
        values = [c["value"] for c in wedge.sidecar["cells"]]
        runs = sum(1 for a, b in zip(values, values[1:]) if b == a + 1)
        assert runs < 30  # sequential layout would have ~511

    def test_seed_reproducible(self):
        a = step_wedge((255, 0, 0), seed=99).sidecar["cells"]
        b = step_wedge((255, 0, 0), seed=99).sidecar["cells"]
        assert a == b
        c = step_wedge((255, 0, 0), seed=100).sidecar["cells"]
        assert [x["value"] for x in a] != [x["value"] for x in c]

    def test_every_cell_renders_its_sidecar_value(self, wedge):
        """The wedge TIFF and its sidecar must agree exactly — this is what analyze.py
        will rely on."""
        print_view = wedge.film[:, ::-1]  # back to print orientation
        rgb = tuple(wedge.sidecar["blocker_rgb"])
        for cell in wedge.sidecar["cells"][::17]:  # sample sparsely for speed
            y, x = cell["y_px"], cell["x_px"]
            h, w = cell["h_px"], cell["w_px"]
            sample = print_view[y + h // 2 - 3 : y + h // 2 + 4, x + w // 2 - 3 : x + w // 2 + 4]
            expected = apply_blocker(
                np.array([[1.0 - cell["value"] / 255.0]], dtype=np.float32), rgb, 1.0
            )[0, 0]
            assert np.abs(sample.mean(axis=(0, 1)) - expected).max() < 1 / 255

    def test_border_is_full_blocker(self, wedge):
        """Continuous dense border absorbs edge-etch; fiducial corners excepted."""
        border_px = wedge.sidecar["fiducials"]["top_left"]["size_px"]
        mid_x = wedge.film.shape[1] // 2
        top_strip = wedge.film[: border_px // 2, mid_x - 50 : mid_x + 50]
        # Zero = black in the negative = maximum ink.
        expected = apply_blocker(np.zeros((1, 1), dtype=np.float32), (255, 0, 0), 1.0)[0, 0]
        assert np.abs(top_strip - expected).max() < 1 / 255

    def test_fiducials_asymmetric(self, wedge):
        fids = wedge.sidecar["fiducials"]
        assert sum(f["hollow"] for f in fids.values()) == 1  # exactly one breaks symmetry
        print_view = wedge.film[:, ::-1]
        hollow = fids["top_left"]
        solid = fids["top_right"]
        for f, expect_hole in ((hollow, True), (solid, False)):
            square = print_view[
                f["y_px"] : f["y_px"] + f["size_px"], f["x_px"] : f["x_px"] + f["size_px"]
            ]
            centre = square[f["size_px"] // 2, f["size_px"] // 2]
            is_holed = centre[1] < 0.5  # blocker centre → green channel drops
            assert is_holed == expect_hole


class TestOtherTargets:
    def test_exposure_strip_geometry(self):
        strip = exposure_strip(zones=8, zone_mm=22.0, height_mm=30.0)
        h, w, _ = strip.film.shape
        assert w == round(22.0 / 25.4 * PPI) * 8
        assert h == round(30.0 / 25.4 * PPI)
        assert len(strip.sidecar["layout"]) == 8
        # Bottom half is full blocker (red), top half clear film (white).
        assert strip.film[-10, 30, 1] < 0.01  # green killed by red blocker
        assert strip.film[10, 30, :].min() > 0.9  # clear film, away from labels

    def test_blocker_grid_cells_match_sidecar(self):
        grid = blocker_grid()
        print_view = grid.film[:, ::-1]
        for cell in grid.sidecar["cells"][:: max(1, len(grid.sidecar["cells"]) // 12)]:
            y, x = cell["y_px"] + cell["h_px"] // 2, cell["x_px"] + cell["w_px"] // 2
            sampled = (print_view[y - 2 : y + 3, x - 2 : x + 3].mean(axis=(0, 1)) * 255).round()
            assert np.abs(sampled - np.array(cell["rgb"])).max() <= 2

    def test_grid_includes_references(self):
        refs = {r["ref"] for r in blocker_grid().sidecar["references"]}
        assert refs == {"clear", "black"}

    def test_save_writes_tiff_and_sidecar(self, tmp_path):
        tif, side = exposure_strip().save(tmp_path)
        assert tif.exists() and side.exists()
        meta = json.loads(side.read_text())
        assert meta["ppi"] == PPI
        assert meta["working_space"] == "srgb"
        img = load_image(tif)
        assert img.space == "srgb" and img.ppi == PPI


class TestRealChart:
    """The supplied EDN chart: all 256 patch levels must be recoverable, and the
    anti-aliased label pixels must be identifiable for rejection."""

    def test_chart_levels_and_label_rejection(self, chart_path):
        img = load_image(chart_path)
        grey = np.rint(img.data[..., 0] * 65535).astype(np.uint16)
        assert np.array_equal(img.data[..., 0], img.data[..., 1])  # neutral

        values, counts = np.unique(grey, return_counts=True)
        on_grid = np.isclose(values, np.rint(values / 257.0) * 257, atol=1)
        # All 256 patch levels present…
        eight_bit = set(np.rint(values[on_grid] / 257.0).astype(int))
        assert eight_bit == set(range(256))
        # …and off-grid (label) pixels are a rejectable sliver, not a real signal.
        off_grid_fraction = counts[~on_grid].sum() / grey.size
        assert 0 < off_grid_fraction < 0.01
