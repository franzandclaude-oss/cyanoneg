"""The Process tab's render-then-save state machine.

Making a negative and writing it to disk used to be one action. Splitting them buys a
look at the soft proof before committing film, but it introduces a failure that is silent
by construction: a negative rendered under one set of settings, saved after the panel has
been changed to describe a different one. The file would open fine and be wrong.

The defence is a fingerprint of every input, compared against the one taken when the
render started. These tests cover the comparison, the mid-render race, and — most
importantly — that no future input can escape the fingerprint unnoticed.
"""

import time

import numpy as np
import pytest

from cyanoneg import imageio as cio

tk = pytest.importorskip("tkinter")

from cyanoneg.gui import app as app_module  # noqa: E402
from cyanoneg.gui.app import App  # noqa: E402


@pytest.fixture
def app(tk_root, monkeypatch):
    # Message boxes are modal: one unexpected dialog and the run hangs until it is
    # dismissed by hand. Recording them instead keeps failures visible and non-blocking.
    dialogs: list[tuple[str, str]] = []
    for name in ("showinfo", "showerror", "showwarning"):
        monkeypatch.setattr(
            app_module.messagebox, name, lambda title, msg, n=name: dialogs.append((n, msg))
        )
    monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *a, **k: False)

    window = tk.Toplevel(tk_root)
    window.withdraw()
    try:
        instance = App(window)
        instance.dialogs = dialogs  # type: ignore[attr-defined]
        yield instance
    finally:
        window.destroy()


@pytest.fixture
def rendered(app, tmp_path):
    """A negative in hand, exactly as a finished render leaves the tab."""
    app._last_negative = cio.Image(np.zeros((4, 4, 3), dtype=np.float32), "srgb", ppi=360)
    app._rendered_fingerprint = app._settings_fingerprint()
    app._refresh_save_state()
    app.output_var.set(str(tmp_path / "out.tif"))
    return app


def widget_variables(widget) -> set[str]:
    """Every Tcl variable bound to a control anywhere under `widget`."""
    found = set()
    for child in widget.winfo_children():
        for option in ("textvariable", "variable"):
            try:
                name = str(child.cget(option))
            except tk.TclError:
                continue
            if name:
                found.add(name)
        found |= widget_variables(child)
    return found


class TestNothingEscapesTheFingerprint:
    def test_every_control_is_tracked_or_declared_harmless(self, app):
        """The test that keeps the rest of this file honest.

        Anyone adding a control to the Process tab must either register it with
        `_tracked` or add it below with a reason. Doing neither means a setting that
        changes the negative without invalidating the one already rendered, and that
        failure produces a plausible-looking wrong file rather than an error.
        """
        harmless = {
            str(app.preview_mode): "which view is on screen cannot change the pixels",
            str(app.output_var): "where the file goes does not change what is in it",
        }
        tracked = {str(var) for var in app._fingerprint_vars.values()}
        escaped = widget_variables(app.process_tab) - tracked - harmless.keys()
        assert not escaped, f"untracked Process tab controls: {escaped}"

    def test_the_walker_actually_finds_controls(self, app):
        """Guards the test above: a walker that returned nothing would always pass."""
        assert len(widget_variables(app.process_tab)) >= len(app._fingerprint_vars)

    @pytest.mark.parametrize(
        "name", ["source_var", "space_var", "raw_var", "weights_var", "profile_var",
                 "width_var", "height_var", "ppi_var", "auto_orient_var"]
    )
    def test_each_tracked_input_moves_the_fingerprint(self, app, name):
        var = app._fingerprint_vars[name]
        before = app._settings_fingerprint()
        var.set(var.get() + "x")
        assert app._settings_fingerprint() != before

    def test_the_parametrised_list_matches_what_is_registered(self, app):
        """So that adding a tracked input without a case here is not silently allowed."""
        listed = set(TestNothingEscapesTheFingerprint
                     .test_each_tracked_input_moves_the_fingerprint.pytestmark[0].args[1])
        assert listed == set(app._fingerprint_vars)


class TestStaleness:
    def test_a_fresh_negative_can_be_saved(self, rendered):
        assert rendered._is_current()
        assert str(rendered.save_button["state"]) == "normal"

    @pytest.mark.parametrize("name", ["profile_var", "width_var", "weights_var", "raw_var"])
    def test_changing_an_input_disables_saving(self, rendered, name):
        rendered._fingerprint_vars[name].set("99")
        assert not rendered._is_current()
        assert str(rendered.save_button["state"]) == "disabled"

    def test_changing_the_output_path_does_not_disable_saving(self, rendered, tmp_path):
        """The destination is not part of the negative; retargeting it must not block a save."""
        rendered.output_var.set(str(tmp_path / "elsewhere.tif"))
        assert rendered._is_current()

    def test_switching_preview_mode_does_not_disable_saving(self, rendered):
        rendered.preview_mode.set("film")
        assert rendered._is_current()

    def test_a_stale_negative_is_not_written_even_if_asked(self, rendered, tmp_path):
        """The button state is a hint; this check is the guarantee."""
        target = tmp_path / "must_not_exist.tif"
        rendered.output_var.set(str(target))
        rendered.width_var.set("300")  # now stale
        rendered.save_button.config(state="normal")  # pretend the button lied
        rendered._save_negative()
        assert not target.exists()

    def test_a_render_that_finishes_after_a_change_arrives_stale(self, app):
        """The race: change a setting while the worker thread is still going.

        The fingerprint is captured when the render starts, so the negative that comes
        back describes settings that no longer apply and must not be saveable.
        """
        at_start = app._settings_fingerprint()
        app.width_var.set("300")  # changed mid-render
        app._last_negative = cio.Image(np.zeros((4, 4, 3), dtype=np.float32), "srgb", ppi=360)
        app._rendered_fingerprint = at_start
        assert not app._is_current()


class TestRenderThenSave:
    @pytest.fixture
    def source(self, tmp_path):
        ramp = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
        image = cio.Image(np.stack([ramp] * 3, axis=-1), "srgb", ppi=300)
        return cio.save_tiff(tmp_path / "positive.tif", image)

    def settle(self, app, until, timeout=30.0):
        """Pump the event loop until the worker thread's result has been handled.

        The queue is drained by an `after(100, ...)` callback, so the loop has to actually
        let time pass — spinning `update()` alone never lets the timer come due.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.root.update()
            if until():
                return True
            time.sleep(0.01)
        return False

    def test_making_a_negative_writes_nothing(self, app, source, tmp_path):
        """The point of the split: film is expensive, disk clutter is not the issue —
        being able to look before committing is."""
        target = tmp_path / "negative.tif"
        app.source_var.set(str(source))
        app.output_var.set(str(target))
        app.profile_var.set("linear")
        app._process()

        assert self.settle(app, lambda: app._last_negative is not None), "render never finished"
        assert not target.exists(), "Make negative must not write a file"
        assert app._is_current()

        app._save_negative()
        assert target.exists()
        assert app._saved_path == target

    def test_the_saved_file_is_the_negative_that_was_previewed(self, app, source, tmp_path):
        target = tmp_path / "negative.tif"
        app.source_var.set(str(source))
        app.output_var.set(str(target))
        app.profile_var.set("linear")
        app._process()
        assert self.settle(app, lambda: app._last_negative is not None)
        app._save_negative()

        written = cio.load_image(target)
        # One 16-bit step: the file holds what was previewed, to the precision it can hold.
        assert np.abs(written.data - app._last_negative.data).max() <= 1.0 / 65535.0
