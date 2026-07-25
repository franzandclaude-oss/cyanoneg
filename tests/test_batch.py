"""Batch processing: one bad file must never cost you the rest of the run."""

import numpy as np
import pytest
import tifffile

from cyanoneg.imageio import Image, load_image, save_tiff
from cyanoneg.lut import Lut
from cyanoneg.pipeline import PrintSize, batch_negatives, main
from cyanoneg.profiles import Profile


def _profile(**overrides) -> Profile:
    defaults = dict(
        name="batch",
        provisional=True,
        blocker={"model": "fixed_hue", "rgb": [255, 0, 0], "saturation": 1.0},
        lut=Lut.identity(),
    )
    defaults.update(overrides)
    return Profile(**defaults)


@pytest.fixture
def sources(tmp_path):
    """Three good positives, one untagged file that cannot be read without a space."""
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        data = (rng.random((40, 60, 3)) * 0.6 + 0.2).astype(np.float32)
        save_tiff(src / f"shot{i}.tif", Image(data, "srgb", ppi=300))
    tifffile.imwrite(src / "untagged.tif", np.zeros((20, 20, 3), dtype=np.uint16))
    (src / "notes.txt").write_text("ignored", encoding="utf-8")
    return src


class TestBatch:
    def test_processes_folder_and_isolates_failures(self, sources, tmp_path):
        out = tmp_path / "out"
        result = batch_negatives(sources, _profile(), PrintSize(50, 40), out)
        assert len(result.written) == 3
        assert len(result.failed) == 1
        assert result.failed[0][0].name == "untagged.tif"
        assert "colour space" in result.failed[0][1]
        assert result.total == 4  # the .txt is not a source
        assert sorted(p.name for p in result.written) == [
            "shot0_negative.tif",
            "shot1_negative.tif",
            "shot2_negative.tif",
        ]

    def test_outputs_are_valid_negatives(self, sources, tmp_path):
        out = tmp_path / "out"
        result = batch_negatives(sources, _profile(), PrintSize(50, 40), out, output_ppi=360)
        image = load_image(result.written[0])
        assert image.ppi == 360
        assert image.space == "srgb"
        assert not np.allclose(image.data[..., 0], image.data[..., 1])  # colour-blocked

    def test_progress_callback_reports_every_file(self, sources, tmp_path):
        seen = []
        batch_negatives(
            sources,
            _profile(),
            PrintSize(50, 40),
            tmp_path / "out",
            progress=lambda i, total, path: seen.append((i, total, path.name)),
        )
        assert [s[0] for s in seen] == [0, 1, 2, 3]
        assert {s[1] for s in seen} == {4}

    def test_explicit_file_list(self, sources, tmp_path):
        chosen = [sources / "shot0.tif", sources / "shot2.tif"]
        result = batch_negatives(chosen, _profile(), PrintSize(50, 40), tmp_path / "out")
        assert len(result.written) == 2
        assert not result.failed

    def test_refuses_to_overwrite_its_own_source(self, sources):
        """Writing into the source folder with no suffix would eat the originals."""
        result = batch_negatives(sources, _profile(), PrintSize(50, 40), sources, suffix="")
        assert not result.written
        assert all("overwrite the source" in why for _, why in result.failed)

    def test_custom_suffix(self, sources, tmp_path):
        result = batch_negatives(sources, _profile(), PrintSize(50, 40), tmp_path / "o", suffix="_neg")
        assert all(p.name.endswith("_neg.tif") for p in result.written)

    def test_summary_names_failures(self, sources, tmp_path):
        summary = batch_negatives(sources, _profile(), PrintSize(50, 40), tmp_path / "out").summary()
        assert "3 of 4" in summary
        assert "untagged.tif" in summary

    def test_missing_blocker_fails_every_file_not_just_one(self, sources, tmp_path):
        naked = _profile(blocker={"model": "fixed_hue", "rgb": None, "saturation": None})
        result = batch_negatives(sources, naked, PrintSize(50, 40), tmp_path / "out")
        assert not result.written
        assert all("blocker colour" in why for _, why in result.failed if "colour space" not in why)


class TestCli:
    def test_cli_returns_nonzero_when_a_file_fails(self, sources, tmp_path, capsys, monkeypatch):
        profile_path = tmp_path / "p.json"
        _profile().save(profile_path)
        code = main(
            [
                str(sources),
                "--profile",
                str(profile_path),
                "--out",
                str(tmp_path / "out"),
                "--width",
                "50",
                "--height",
                "40",
            ]
        )
        assert code == 1
        printed = capsys.readouterr().out
        assert "3 of 4 negatives written" in printed
        assert "provisional" in printed  # the profile's status is surfaced, not hidden

    def test_cli_returns_zero_when_all_succeed(self, sources, tmp_path):
        profile_path = tmp_path / "p.json"
        _profile(provisional=False).save(profile_path)
        (sources / "untagged.tif").unlink()
        code = main(
            [
                str(sources),
                "--profile",
                str(profile_path),
                "--out",
                str(tmp_path / "out"),
                "--width",
                "50",
                "--height",
                "40",
            ]
        )
        assert code == 0
