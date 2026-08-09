"""A tab body that scrolls only when it has to.

On a 1280x720 display the Process tab needs more height than the screen can give it.
Clamping the window to the screen — which :func:`theme.apply` does — keeps the window
on-screen but silently cuts the bottom off the content, and the bottom of the Process tab
is where "Make negative" and "Save negative" live. Controls that cannot be seen might as
well not exist, and nothing about the window's appearance says they are there.

The container below solves it without costing anything on a large display: when there is
room to spare the content is stretched to fill the viewport, so panes that expand (the
image preview) behave exactly as they did before. Only when the content genuinely does not
fit does it keep its natural size and become scrollable.
"""

from __future__ import annotations

from tkinter import BOTH, LEFT, RIGHT, VERTICAL, Y, Canvas, ttk

from . import theme


class ScrollableFrame(ttk.Frame):
    """A frame whose contents scroll vertically once they outgrow the viewport.

    Put content in :attr:`body`. The outer frame is what gets packed or added to a
    notebook.
    """

    #: How often to notice that the content's *requested* height has changed. Tk fires
    #: <Configure> on actual geometry changes only, and this container sets the body's
    #: height itself — so a note appearing, a warning being hidden, or any widget being
    #: added or removed changes what the body needs without changing what it currently is,
    #: and no event is fired at all. The failure is one-directional and nasty: content
    #: grows past the viewport, the scrollregion stays where it was, and the new content
    #: is unreachable. Comparing one integer a few times a second costs nothing and cannot
    #: miss a cause.
    WATCH_MS = 200

    def __init__(self, parent, padding: int = 0, **kwargs) -> None:
        super().__init__(parent, **kwargs)

        self.canvas = Canvas(
            self, background=theme.PANEL, highlightthickness=0, borderwidth=0, takefocus=0
        )
        self.scrollbar = ttk.Scrollbar(self, orient=VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        self.body = ttk.Frame(self.canvas, padding=padding)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", lambda _e: self._refit())
        self.canvas.bind("<Configure>", lambda _e: self._refit())

        # bind_all is scoped by enter/leave so the wheel only drives the tab under the
        # pointer, rather than whichever canvas happened to be created last.
        self.canvas.bind("<Enter>", lambda _e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda _e: self.canvas.unbind_all("<MouseWheel>"))

        self._last_request = -1
        self._watch_id: str | None = None
        self.bind("<Destroy>", self._stop_watching)
        self._watch()

    def _watch(self) -> None:
        request = self.body.winfo_reqheight()
        if request != self._last_request:
            self._last_request = request
            self._refit()
        self._watch_id = self.after(self.WATCH_MS, self._watch)

    def _stop_watching(self, event) -> None:
        if event.widget is self and self._watch_id is not None:
            self.after_cancel(self._watch_id)
            self._watch_id = None

    @property
    def scrollable(self) -> bool:
        """True when the content is taller than the viewport showing it."""
        return self.body.winfo_reqheight() > self.canvas.winfo_height()

    def _refit(self) -> None:
        view_w, view_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        needed = self.body.winfo_reqheight()
        # Stretch to the viewport when there is room; keep natural height when there is not.
        self.canvas.itemconfigure(self._window, width=view_w, height=max(needed, view_h))
        self.canvas.configure(scrollregion=(0, 0, view_w, max(needed, view_h)))

        if needed > view_h:
            if not self.scrollbar.winfo_ismapped():
                self.scrollbar.pack(side=RIGHT, fill=Y)
        elif self.scrollbar.winfo_ismapped():
            self.scrollbar.pack_forget()
            self.canvas.yview_moveto(0.0)

    def _on_wheel(self, event) -> None:
        if self.body.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(-int(event.delta / 120), "units")
