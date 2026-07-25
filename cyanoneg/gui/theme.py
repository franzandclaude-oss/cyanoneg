"""Single source of truth for the GUI's look.

Every colour, font and spacing value the interface uses lives here, so restyling is a
change to this file rather than a hunt through ``app.py``.

**On changing colours:** ``THEME = "vista"`` is the native Windows ttk theme. It draws
through the OS and therefore *ignores* most colour settings on ttk widgets — buttons,
frames and entries will keep their system appearance no matter what is set below. The
colours here still apply to plain tk widgets (the curve canvas) and to ttk ``Label``
foregrounds, which is why status text can be tinted but a button cannot.

To take full control of the appearance, set ``THEME = "clam"``: it is drawn by Tk itself
and honours every colour, at the cost of the native look. That single change is the
intended switch point for a visual redesign.
"""

from __future__ import annotations

from tkinter import ttk

#: ttk theme. "vista" = native Windows (colours mostly locked); "clam" = fully styleable.
THEME = "vista"

# --------------------------------------------------------------------------- colour
OK = "#205020"  # success / completed status text
WARN = "#a06000"  # provisional profiles, placeholder warnings
MUTED = "#606060"  # secondary explanatory text
CURVE = "#1040a0"  # measured correction curve
CURVE_REFERENCE = "#cccccc"  # identity diagonal behind the curve
CANVAS_BG = "#ffffff"  # curve/preview canvas background

# --------------------------------------------------------------------------- type
MONO = ("Consolas", 9)  # analysis output, JSON detail — alignment matters
MONO_SMALL = ("Consolas", 8)

# --------------------------------------------------------------------------- metrics
PAD = 10  # tab padding
GROUP_PAD = 8  # padding inside a LabelFrame
GAP = 6  # standard gap between related controls
WIDE_GAP = 16  # gap between control groups on one row

PREVIEW_MAX = 420  # longest edge of the negative preview, px
CURVE_SIZE = 220  # curve preview canvas, px square

WINDOW = "1180x780"
MIN_WINDOW = (1100, 720)


def apply(root) -> None:
    """Apply the theme to a freshly created root window."""
    root.geometry(WINDOW)
    root.minsize(*MIN_WINDOW)
    style = ttk.Style()
    try:
        style.theme_use(THEME)
    except Exception:  # noqa: BLE001 - theme availability varies by platform
        pass
