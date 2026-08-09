"""Nothing in a tab may be unreachable, at any window size.

The Process tab needed 632px of height and the window opened at 640 total, so the bottom
of the left column — "Make negative", "Save negative", "Show in Explorer" — was simply cut
off. Nothing about the window said the buttons were there; they appeared not to exist.

Two things went wrong and both are covered here: the window was sized from a guess at the
taskbar's height rather than from the actual work area, and content that outgrew its tab
was clipped rather than scrolled.
"""

import time

import pytest

tk = pytest.importorskip("tkinter")

from cyanoneg.gui import theme  # noqa: E402
from cyanoneg.gui.app import App  # noqa: E402
from cyanoneg.gui.scroll import ScrollableFrame  # noqa: E402


@pytest.fixture
def window(tk_root):
    top = tk.Toplevel(tk_root)
    top.geometry("400x300")
    try:
        yield top
    finally:
        top.destroy()


def settle(widget) -> None:
    widget.update_idletasks()
    widget.update()


def wait_until(widget, condition, timeout: float = 3.0) -> bool:
    """Pump the event loop until `condition` holds — the size watcher runs on a timer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        settle(widget)
        if condition():
            return True
        time.sleep(0.01)
    return False


def filled(parent, height: int) -> ScrollableFrame:
    frame = ScrollableFrame(parent)
    frame.pack(fill=tk.BOTH, expand=True)
    tk.Frame(frame.body, height=height, width=50).pack()
    settle(parent)
    return frame


class TestScrollableFrame:
    def test_short_content_stretches_to_fill(self, window):
        """With room to spare the body fills the viewport.

        This is what keeps panes that expand — the image preview — behaving as they did
        before scrolling existed. A body pinned to its natural height would leave the
        preview stranded at minimum size on a large display.
        """
        frame = filled(window, 50)
        assert not frame.scrollable
        assert frame.canvas.itemcget(frame._window, "height") == str(frame.canvas.winfo_height())

    def test_short_content_shows_no_scrollbar(self, window):
        assert not filled(window, 50).scrollbar.winfo_ismapped()

    def test_tall_content_scrolls_instead_of_being_clipped(self, window):
        frame = filled(window, 2000)
        assert frame.scrollable
        assert frame.scrollbar.winfo_ismapped()
        _, _, _, bottom = frame.canvas.cget("scrollregion").split()
        assert int(bottom) >= 2000, "the whole body must be inside the scrollable region"

    def test_content_that_changes_size_is_noticed(self, window):
        """Adding or removing content changes what the body *requests* without changing
        what it currently *is*, so Tk fires no event. The container has to notice anyway.

        The dangerous direction is growth: content appears below the fold, the scrollregion
        stays where it was, and the new content cannot be reached at all — which is the
        original bug, reintroduced by the fix for it.
        """
        frame = filled(window, 50)
        assert not frame.scrollable

        grown = tk.Frame(frame.body, height=2000, width=50)
        grown.pack()
        assert wait_until(window, lambda: frame.scrollbar.winfo_ismapped()), "growth unnoticed"
        assert int(frame.canvas.cget("scrollregion").split()[3]) >= 2000

        frame.canvas.yview_moveto(1.0)
        grown.destroy()
        assert wait_until(window, lambda: not frame.scrollbar.winfo_ismapped()), "shrink unnoticed"
        # ...and the view resets, or the body stays scrolled off the top with no way back.
        assert frame.canvas.yview()[0] == 0.0


class TestWindowSize:
    def test_sized_from_the_work_area_not_a_guess(self, tk_root):
        """wm_maxsize excludes the taskbar; the old fixed margin only pretended to.

        On the 1280x720 display this was found on, guessing cost 61px of window — more
        than the button row that went missing needed.
        """
        width, height, _, _ = theme.window_size(tk_root)
        max_w, max_h = tk_root.wm_maxsize()
        assert height <= max_h and width <= max_w
        assert height >= min(theme.PREFERRED_WINDOW[1], max_h - theme.EDGE)

    def test_the_window_fits_on_screen(self, tk_root):
        width, height, x, y = theme.window_size(tk_root)
        assert x >= 0 and y >= 0
        assert x + width <= tk_root.winfo_screenwidth() + theme.EDGE

    def test_minimum_is_never_larger_than_the_screen(self, tk_root):
        """A minsize bigger than the display cannot be satisfied, so the excess is clipped
        with no way to scroll or resize out of it."""
        top = tk.Toplevel(tk_root)
        try:
            theme.apply(top)
            min_w, min_h = top.minsize()
            assert min_w <= top.winfo_screenwidth()
            assert min_h <= top.winfo_screenheight()
        finally:
            top.destroy()


class TestEveryControlIsReachable:
    @pytest.fixture
    def cramped(self, tk_root):
        """The app in a window far too small for it — the situation that started this."""
        top = tk.Toplevel(tk_root)
        app = App(top)
        top.geometry("900x480")
        settle(top)
        try:
            yield app
        finally:
            top.destroy()

    @pytest.mark.parametrize("button", ["process_button", "save_button", "reveal_button"])
    def test_action_buttons_stay_inside_the_scrollable_region(self, cramped, button):
        """Reachable by scrolling is the guarantee; visible without scrolling is not.

        Measured against the scrollregion rather than the viewport, because a small window
        legitimately cannot show everything at once — it just must never hide it.
        """
        scroller = cramped.scrollers["process_tab"]
        widget = getattr(cramped, button)
        bottom = widget.winfo_rooty() + widget.winfo_height() - scroller.body.winfo_rooty()
        region_bottom = int(scroller.canvas.cget("scrollregion").split()[3])
        assert 0 < bottom <= region_bottom

    def test_the_cramped_window_really_is_cramped(self, cramped):
        """Guards the test above: if everything fitted, it would prove nothing."""
        assert cramped.scrollers["process_tab"].scrollable
