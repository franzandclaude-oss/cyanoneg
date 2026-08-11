"""Where the GUI puts a negative.

Two rules, and they pull against each other, which is why this file exists:

* the negative's *name* always follows the source image, so a print session cannot
  quietly overwrite its own earlier output;
* the negative's *folder* follows the source too, unless Steven chose one himself.

Getting the second rule wrong in either direction is invisible until a file goes
missing: keep the folder too eagerly and every image lands in the first one's
directory; keep it not at all and a deliberately chosen destination is ignored.
"""

import sys
from pathlib import Path

import pytest

from cyanoneg.gui.app import App, output_for_source, reveal_command

#: `output_for_source` splits and rebuilds paths with the ambient `pathlib.Path`, which is
#: the right choice in production — Windows is the only supported runtime, and `Path` there
#: *is* `WindowsPath`. On a POSIX host the same class has no idea `\` is a separator or that
#: `C:` is a drive, so a Windows-style source string like `C:\pics\barn.tif` comes back as
#: one opaque filename instead of a folder plus a name. That is a test-portability gap, not
#: a production defect: nothing here runs, or needs to run, against real Windows path
#: splitting anywhere but Windows itself. Per the technical review (2026-08-11, finding 7),
#: these are marked Windows-only rather than rewritten against `PureWindowsPath` — a
#: `PureWindowsPath` assertion could pass here while telling us nothing about whether the
#: function that actually ships still behaves the same way on its one real runtime.
windows_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason="depends on the host Path class parsing Windows-style paths; only meaningful on Windows",
)


class TestRevealCommand:
    """Opening a file manager is the one place this GUI touches the operating system.

    It fails silently when wrong — the button does nothing and nothing is logged — so the
    per-platform commands are checked here rather than discovered on the day someone opens
    the project on a different machine.
    """

    @pytest.mark.parametrize(
        "platform,expected",
        [
            ("win32", ["explorer", "/select,", r"C:\pics\barn_negative.tif"]),
            ("darwin", ["open", "-R", r"C:\pics\barn_negative.tif"]),
        ],
    )
    def test_each_platform_gets_its_own_command(self, monkeypatch, platform, expected):
        monkeypatch.setattr("cyanoneg.gui.app.sys.platform", platform)
        assert reveal_command(Path(r"C:\pics\barn_negative.tif")) == expected

    def test_linux_opens_the_folder_since_it_cannot_select(self, monkeypatch):
        monkeypatch.setattr("cyanoneg.gui.app.sys.platform", "linux")
        command = reveal_command(Path("/home/s/pics/barn_negative.tif"))
        assert command[0] == "xdg-open"
        assert command[1].endswith("pics"), "no standard select verb, so open the folder"

    def test_the_windows_flag_keeps_its_comma(self, monkeypatch):
        """`explorer /select, path` only works with the comma attached to the flag;
        `/select` alone opens the file instead of showing it in its folder."""
        monkeypatch.setattr("cyanoneg.gui.app.sys.platform", "win32")
        assert reveal_command(Path("x.tif"))[1] == "/select,"


class TestOutputPath:
    @windows_only
    def test_name_comes_from_the_source(self):
        assert Path(output_for_source(r"C:\pics\barn.tif")).name == "barn_negative.tif"

    @windows_only
    def test_sits_beside_its_own_source(self):
        got = Path(output_for_source(r"C:\pics\barn.tif"))
        assert got.parent == Path(r"C:\pics")

    @windows_only
    def test_a_new_source_never_reuses_the_old_negatives_name(self):
        """Two images in a row must not land on one file.

        The original behaviour filled the box only when it was empty, so every image
        after the first overwrote the first one's negative — silently, since the field
        still showed a plausible path.
        """
        first = output_for_source(r"C:\pics\barn.tif")
        second = output_for_source(r"C:\pics\gate.tif")
        assert first != second
        assert Path(second).name == "gate_negative.tif"

    @windows_only
    def test_images_from_different_folders_stay_in_their_own_folders(self):
        """The regression from the first attempt at the fix.

        That version kept whatever folder was already in the box, which could not tell a
        folder Steven picked from one merely derived from the previous source. Loading an
        image from a second folder wrote its negative back to the first folder, and no
        amount of choosing a new location while a source was selected would stick.
        """
        assert Path(output_for_source(r"D:\shoot2\gate.tif")).parent == Path(r"D:\shoot2")

    @windows_only
    def test_a_chosen_folder_is_kept(self):
        """Steven picked that destination on purpose; only the filename should move."""
        got = Path(output_for_source(r"C:\pics\gate.tif", chosen=r"D:\negatives\barn_negative.tif"))
        assert got.parent == Path(r"D:\negatives")
        assert got.name == "gate_negative.tif"

    @windows_only
    def test_a_hand_typed_filename_does_not_follow_a_different_image(self):
        """A name like "for_the_show.tif" belongs to the image it was typed for."""
        got = Path(output_for_source(r"C:\pics\gate.tif", chosen=r"D:\out\for_the_show.tif"))
        assert got.name == "gate_negative.tif"

    def test_same_source_twice_is_stable(self):
        """Reprocessing one image with a different profile does overwrite, deliberately —
        otherwise the folder fills with barn_negative (3).tif and nobody knows which is live."""
        once = output_for_source(r"C:\pics\barn.tif")
        assert output_for_source(r"C:\pics\barn.tif", chosen=once) == once


@pytest.fixture
def app(tk_root):
    """A real, hidden App. The rules above are only correct if the window wires them up."""
    import tkinter as tk

    window = tk.Toplevel(tk_root)
    window.withdraw()
    try:
        yield App(window)
    finally:
        window.destroy()


class TestTheBoxTracksTheSource:
    """Driving the widgets, because the bug was never in the function — it was in what
    the window passed to it."""

    def pick(self, app, source):
        """What _pick_source does once the file dialog has returned a path."""
        app.source_var.set(source)
        app._set_output(output_for_source(source, app._chosen_output))

    def test_following_the_source_across_folders(self, app):
        self.pick(app, r"C:\shoot1\barn.tif")
        assert app.output_var.get() == str(Path(r"C:\shoot1\barn_negative.tif"))
        self.pick(app, r"D:\shoot2\gate.tif")
        assert app.output_var.get() == str(Path(r"D:\shoot2\gate_negative.tif"))

    @windows_only
    def test_a_typed_destination_survives_the_next_image(self, app):
        self.pick(app, r"C:\shoot1\barn.tif")
        app.output_var.set(r"E:\final\barn_negative.tif")  # as if typed into the entry
        self.pick(app, r"C:\shoot1\gate.tif")
        assert app.output_var.get() == str(Path(r"E:\final\gate_negative.tif"))

    def test_the_app_writing_the_box_is_not_mistaken_for_a_choice(self, app):
        """If it were, the first auto-filled folder would pin every later negative."""
        self.pick(app, r"C:\shoot1\barn.tif")
        assert app._chosen_output == ""
