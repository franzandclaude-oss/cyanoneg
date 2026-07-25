"""Single source of truth for the GUI's look.

Every colour, font and metric the interface uses lives here, so restyling is a change to
this file rather than a hunt through ``app.py``.

**Why dark.** This is a tool for judging tone. A bright surround makes an image read
darker than it is, and a coloured surround skews its hue — the same reason editing
applications default to dark neutral chrome. The panels are deliberately desaturated so
the negative and the soft proof are the only saturated things on screen.

**Why the clam theme.** Windows' native ttk themes draw through the OS and ignore most
colour settings, so buttons and fields would stay light no matter what is set below.
``clam`` is drawn by Tk itself and honours every colour. That is the trade: full control
over the appearance, at the cost of the native Windows look.

Plain tk widgets (canvases, text areas) are not covered by ttk styling, so
:func:`style_text` and :func:`style_canvas` exist to bring them into line.
"""

from __future__ import annotations

from tkinter import ttk

THEME = "clam"

# --------------------------------------------------------------------------- colour
BG = "#1e2024"  # window chrome, behind everything
PANEL = "#26292e"  # raised surfaces: tabs, group boxes
PANEL_HI = "#2f3339"  # hover, table headings
FIELD = "#191b1f"  # recessed: entries, lists, log
BORDER = "#3a3f47"

TEXT = "#e6e8ea"
TEXT_MUTED = "#9aa1a9"  # secondary and explanatory copy
TEXT_DISABLED = "#5c626a"
TEXT_ON_ACCENT = "#ffffff"

ACCENT = "#4a7fb5"  # Prussian blue, the one saturated UI colour
ACCENT_HI = "#5d94cc"
ACCENT_LOW = "#3c6894"

OK = "#6fbf73"  # completed steps, saved files
WARN = "#d4a24c"  # provisional profiles, placeholder blocker
DANGER = "#d4665c"  # failures in a batch run

CANVAS_BG = "#141619"  # image preview and curve backgrounds — darkest surface
CURVE = "#7fb2e8"  # the measured correction curve
CURVE_REFERENCE = "#464c55"  # identity diagonal behind it

# --------------------------------------------------------------------------- type
_UI = "Segoe UI"
FONT = (_UI, 10)
FONT_BOLD = (_UI, 10, "bold")
FONT_SMALL = (_UI, 9)
FONT_HEADING = (_UI, 12, "bold")
FONT_STEP = (_UI, 14, "bold")  # step numerals in the Calibrate wizard
MONO = ("Consolas", 10)
MONO_SMALL = ("Consolas", 9)

# --------------------------------------------------------------------------- metrics
PAD = 16  # tab padding
GROUP_PAD = 12  # inside a group box
GAP = 8  # between related controls
WIDE_GAP = 20  # between control groups
ROW_GAP = 6  # vertical rhythm within a form

FIELD_WIDTH = 34  # every path/text entry, so columns align
NARROW_WIDTH = 8
LABEL_WIDTH = 15  # form label column

PREVIEW_MAX = 460
CURVE_SIZE = 190

#: Preferred size, and the smallest the layout stays usable at. Both are clamped to the
#: actual display in :func:`apply` — a window taller than the screen silently hides its
#: bottom row of controls, and no amount of styling makes that acceptable.
PREFERRED_WINDOW = (1240, 820)
MIN_WINDOW = (1060, 620)
SCREEN_MARGIN = 80  # leave room for the taskbar and window chrome


def apply(root) -> None:
    """Apply the theme to a freshly created root window, sized to fit the display."""
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    width = min(PREFERRED_WINDOW[0], screen_w - SCREEN_MARGIN)
    height = min(PREFERRED_WINDOW[1], screen_h - SCREEN_MARGIN)
    root.geometry(f"{width}x{height}+{max(0, (screen_w - width) // 2)}+{max(0, (screen_h - height) // 2 - 20)}")
    root.minsize(min(MIN_WINDOW[0], width), min(MIN_WINDOW[1], height))
    root.configure(background=BG)

    # The combobox dropdown is a plain Tk listbox and is not reachable via ttk styling.
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", TEXT_ON_ACCENT)
    root.option_add("*TCombobox*Listbox.font", FONT)

    style = ttk.Style()
    try:
        style.theme_use(THEME)
    except Exception:  # noqa: BLE001 - theme availability varies by platform
        return

    style.configure(
        ".",
        background=PANEL,
        foreground=TEXT,
        fieldbackground=FIELD,
        bordercolor=BORDER,
        lightcolor=PANEL,
        darkcolor=PANEL,
        troughcolor=FIELD,
        focuscolor=ACCENT,
        font=FONT,
    )

    style.configure("TFrame", background=PANEL)
    style.configure("Chrome.TFrame", background=BG)

    style.configure("TLabel", background=PANEL, foreground=TEXT)
    style.configure("Heading.TLabel", font=FONT_HEADING)
    style.configure("Muted.TLabel", foreground=TEXT_MUTED)
    style.configure("Warn.TLabel", foreground=WARN)
    style.configure("OK.TLabel", foreground=OK)
    style.configure("Danger.TLabel", foreground=DANGER)
    style.configure("Mono.TLabel", font=MONO_SMALL, foreground=TEXT_MUTED)
    style.configure("Step.TLabel", font=FONT_STEP, foreground=ACCENT)

    style.configure("TButton", padding=(14, 7), relief="flat", background=PANEL_HI, borderwidth=0)
    style.map(
        "TButton",
        background=[("disabled", PANEL), ("pressed", ACCENT_LOW), ("active", BORDER)],
        foreground=[("disabled", TEXT_DISABLED)],
    )
    style.configure("Accent.TButton", background=ACCENT, foreground=TEXT_ON_ACCENT, font=FONT_BOLD)
    style.map(
        "Accent.TButton",
        background=[("disabled", PANEL_HI), ("pressed", ACCENT_LOW), ("active", ACCENT_HI)],
        foreground=[("disabled", TEXT_DISABLED)],
    )
    style.configure("Small.TButton", padding=(8, 5))

    style.configure(
        "TEntry", padding=5, insertcolor=TEXT, fieldbackground=FIELD, foreground=TEXT, borderwidth=1
    )
    style.map("TEntry", fieldbackground=[("disabled", PANEL)], bordercolor=[("focus", ACCENT)])

    style.configure(
        "TCombobox",
        padding=5,
        arrowcolor=TEXT_MUTED,
        arrowsize=14,
        fieldbackground=FIELD,
        background=PANEL_HI,  # the drop-down button, drawn separately from the field
        foreground=TEXT,
        borderwidth=1,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", FIELD), ("disabled", PANEL)],
        background=[("active", BORDER), ("pressed", BORDER), ("disabled", PANEL)],
        foreground=[("disabled", TEXT_DISABLED)],
        bordercolor=[("focus", ACCENT)],
        arrowcolor=[("disabled", TEXT_DISABLED), ("active", TEXT)],
    )

    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(PAD, GAP, PAD, 0))
    style.configure(
        "TNotebook.Tab", background=BG, foreground=TEXT_MUTED, padding=(18, 9), borderwidth=0
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", PANEL)],
        foreground=[("selected", TEXT), ("active", TEXT)],
    )

    style.configure("TLabelframe", background=PANEL, borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=PANEL, foreground=TEXT, font=FONT_BOLD)

    for widget in ("TCheckbutton", "TRadiobutton"):
        style.configure(
            widget,
            background=PANEL,
            focuscolor=PANEL,
            padding=3,
            indicatorbackground=FIELD,
            indicatorforeground=TEXT_ON_ACCENT,
            upperbordercolor=BORDER,
            lowerbordercolor=BORDER,
        )
        style.map(
            widget,
            background=[("active", PANEL)],
            foreground=[("disabled", TEXT_DISABLED)],
            indicatorbackground=[
                ("selected", ACCENT),
                ("active", PANEL_HI),
                ("disabled", PANEL),
                ("!selected", FIELD),
            ],
            upperbordercolor=[("selected", ACCENT), ("focus", ACCENT)],
            lowerbordercolor=[("selected", ACCENT), ("focus", ACCENT)],
        )

    style.configure(
        "TProgressbar", background=ACCENT, troughcolor=FIELD, borderwidth=0, lightcolor=ACCENT, darkcolor=ACCENT
    )

    style.configure(
        "Treeview", background=FIELD, fieldbackground=FIELD, foreground=TEXT, borderwidth=0, rowheight=26
    )
    style.configure("Treeview.Heading", background=PANEL_HI, foreground=TEXT, relief="flat", padding=6)
    style.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", TEXT_ON_ACCENT)])
    style.map("Treeview.Heading", background=[("active", BORDER)])

    style.configure("TScrollbar", background=PANEL_HI, troughcolor=BG, borderwidth=0, arrowcolor=TEXT_MUTED)
    style.map("TScrollbar", background=[("active", BORDER)])

    style.configure("TSeparator", background=BORDER)


def style_text(widget) -> None:
    """Bring a plain tk Text/ScrolledText into the theme."""
    widget.configure(
        background=FIELD,
        foreground=TEXT,
        insertbackground=TEXT,
        selectbackground=ACCENT,
        selectforeground=TEXT_ON_ACCENT,
        relief="flat",
        borderwidth=0,
        padx=GAP,
        pady=GAP,
        font=MONO_SMALL,
    )


def style_canvas(widget) -> None:
    """Bring a plain tk Canvas into the theme."""
    widget.configure(background=CANVAS_BG, highlightthickness=1, highlightbackground=BORDER)
