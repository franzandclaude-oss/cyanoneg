import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cyanoneg.imageio import Image  # noqa: E402

CHART = ROOT / "EDN_RGB_256.tif"


@pytest.fixture
def ramp_image() -> Image:
    """A horizontal 0→1 ramp, RGB, sRGB, 300 ppi — asymmetric on purpose."""
    ramp = np.tile(np.linspace(0.0, 1.0, 128, dtype=np.float32), (64, 1))
    return Image(np.stack([ramp] * 3, axis=-1), "srgb", ppi=300)


@pytest.fixture
def chart_path() -> Path:
    if not CHART.exists():
        pytest.skip("EDN_RGB_256.tif not present")
    return CHART


@pytest.fixture(scope="session")
def tk_root():
    """One Tk root for the whole run.

    Creating and destroying a root per test made Tk fail to initialise every few runs,
    which the GUI fixtures then turned into a skip — and a test that skips at random is a
    test that silently stops guarding anything. Tests get a Toplevel off this instead.
    """
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as e:  # genuinely no display
        pytest.skip(str(e))
    root.withdraw()
    yield root
    root.destroy()
