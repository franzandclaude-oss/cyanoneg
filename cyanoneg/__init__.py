"""Reproducible cyanotype digital negatives.

Encode a paper/chemistry/film combination once as a measured profile, then apply it
deterministically to any image. See PLAN.md for the process rationale.
"""

__version__ = "0.1.0"


def use_utf8_console() -> None:
    """Make stdout/stderr able to carry the punctuation this package prints.

    Windows consoles default to cp1252. That encoding is wider than it looks — en and em
    dashes, ``×`` and ``•`` all survive it — but it has no mapping for ``→``, ``≥``, ``σ``,
    ``∘`` or ``≈``, and this package's module docstrings are full of arrows. argparse feeds
    those docstrings straight to stdout as the ``--help`` description, so on a stock console
    ``--help`` dies with ``UnicodeEncodeError`` instead of printing.

    Today the damage stops at ``--help``: every string the analysis actually prints is
    cp1252-safe. That is luck rather than design, and one ``→`` added to a warning message
    would turn a lost help screen into a lost measurement. Called at the top of each
    ``main()`` so the question never has to be asked again.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # detached or already-wrapped stream
                pass
