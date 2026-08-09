"""Where the GUI puts a negative.

Importing the GUI module pulls in tkinter but constructs no window, so these run
headless. The rule under test is the one that stops a print session quietly
destroying its own output.
"""

from pathlib import Path

from cyanoneg.gui.app import output_for_source


class TestOutputPath:
    def test_name_comes_from_the_source(self):
        assert Path(output_for_source(r"C:\pics\barn.tif")).name == "barn_negative.tif"

    def test_sits_beside_the_source_by_default(self):
        got = Path(output_for_source(r"C:\pics\barn.tif"))
        assert got.parent == Path(r"C:\pics")

    def test_a_new_source_never_reuses_the_old_negatives_name(self):
        """The whole point: two images in a row must not land on one file.

        The old behaviour only filled the box when it was empty, so every image after
        the first overwrote the first one's negative — silently, since the field still
        showed a plausible path.
        """
        first = output_for_source(r"C:\pics\barn.tif")
        second = output_for_source(r"C:\pics\gate.tif", first)
        assert first != second
        assert Path(second).name == "gate_negative.tif"

    def test_a_chosen_folder_is_kept(self):
        """Steven picked that destination on purpose; only the filename should move."""
        chosen = r"D:\negatives\barn_negative.tif"
        got = Path(output_for_source(r"C:\pics\gate.tif", chosen))
        assert got.parent == Path(r"D:\negatives")
        assert got.name == "gate_negative.tif"

    def test_a_hand_typed_filename_does_not_follow_a_different_image(self):
        """A name like "for_the_show.tif" belongs to the image it was typed for."""
        got = Path(output_for_source(r"C:\pics\gate.tif", r"D:\negatives\for_the_show.tif"))
        assert got.name == "gate_negative.tif"

    def test_same_source_twice_is_stable(self):
        """Reprocessing one image with a different profile does overwrite, deliberately —
        otherwise the folder fills with barn_negative (3).tif and nobody knows which is live."""
        once = output_for_source(r"C:\pics\barn.tif")
        assert output_for_source(r"C:\pics\barn.tif", once) == once
