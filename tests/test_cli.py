"""The CLIs must survive a legacy Windows console.

Windows consoles default to cp1252, which has no mapping for the arrows filling this
package's module docstrings. argparse feeds those docstrings to stdout as the ``--help``
description, so without a guard ``--help`` dies with ``UnicodeEncodeError``.

Note which characters actually matter: cp1252 encodes en dashes, em dashes, ``×`` and
``•`` perfectly well. It is ``→ ≥ σ ∘ ≈`` that break it. Tests here use ``→`` for that
reason — a dash would prove nothing.

These run in a **subprocess** with ``PYTHONIOENCODING=cp1252``. An in-process test cannot
show the bug: pytest installs its own UTF-8 stdout, so monkeypatching ``sys.stdout`` to a
cp1252 stream does not survive to the moment of the write, and the test then passes
whether or not the bug is present. :func:`test_the_hostile_console_is_real` exists to keep
that from happening silently — it fails if the child stops being hostile.
"""

import os
import subprocess
import sys

import pytest

CLI_MODULES = ["cyanoneg.analyze", "cyanoneg.targets", "cyanoneg.pipeline"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args: str) -> subprocess.CompletedProcess:
    """Run a child interpreter whose stdio encodes like a default Windows console."""
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        timeout=120,
    )


def test_the_hostile_console_is_real():
    """Premise check: without the guard, an arrow does take a command down.

    If this ever stops failing, the other tests in this file have stopped proving anything
    and the encoding setup needs revisiting.
    """
    result = _run("-c", "print('clear film \\u2192 max black')")
    assert result.returncode != 0
    assert b"UnicodeEncodeError" in result.stderr


def test_dashes_were_never_the_problem():
    """Guards the claim in the module docstring, so nobody re-widens the fix by guesswork.

    The density-range warnings quote ``1.2-1.4`` with an en dash and print fine unguarded.
    """
    result = _run("-c", "print('density range 1.2\\u20131.4 \\u2014 ok')")
    assert result.returncode == 0


@pytest.mark.parametrize("module", CLI_MODULES)
def test_help_survives_a_cp1252_console(module):
    result = _run("-m", module, "--help")
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert b"usage:" in result.stdout


@pytest.mark.parametrize("module", CLI_MODULES)
def test_help_carries_the_docstring_through(module):
    """Not just 'did not crash' — the description has to arrive, not get truncated."""
    result = _run("-m", module, "--help")
    assert len(result.stdout) > 400
    assert b"UnicodeEncodeError" not in result.stderr


def test_analyze_subcommand_help_also_survives():
    """The wedge subparser is what gets run against a real scan."""
    result = _run("-m", "cyanoneg.analyze", "wedge", "--help")
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert b"usage:" in result.stdout


# ------------------------------------------------------------------ the helper itself


def test_helper_switches_the_stream_to_utf8(monkeypatch):
    import io

    from cyanoneg import use_utf8_console

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    use_utf8_console()
    assert stream.encoding.lower().replace("-", "") == "utf8"
    assert stream.errors == "replace"


def test_helper_tolerates_detached_streams(monkeypatch):
    """A stream with no reconfigure (or a dead one) must not break the CLI."""
    from cyanoneg import use_utf8_console

    class Dead:
        def reconfigure(self, **kwargs):
            raise ValueError("underlying buffer has been detached")

    monkeypatch.setattr(sys, "stdout", Dead())
    monkeypatch.setattr(sys, "stderr", object())  # no reconfigure attribute at all
    use_utf8_console()  # must not raise
