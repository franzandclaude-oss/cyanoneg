"""Paper profile model: JSON load/save/validate.

One profile per paper/chemistry/film combination. Three fields carry weight beyond
documentation (see PLAN.md):

- ``working_space`` — the space the LUT was measured in and must be applied in. Read by
  the pipeline, not merely recorded.
- ``provisional`` — true means the LUT and blocker are unmeasured; the GUI must mark such
  profiles so a guess is never mistaken for a measurement.
- ``driver_settings`` — the exact Epson driver state used for the calibration print.
  The calibration silently absorbs the driver's transform, so changing these afterwards
  invalidates the profile.

Raw patch measurements are stored alongside the computed LUT so a profile can be recomputed
when the smoothing improves, without reprinting anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .imageio import SPACES, Space
from .lut import DEFAULT_SIZE, Lut

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"


class ProfileError(ValueError):
    """Raised when a profile file is structurally invalid."""


@dataclass
class Profile:
    name: str
    paper: str = ""
    chemistry: str = ""
    printer: str = "Epson ET-1810"
    media_type: str = ""
    film: str = ""
    #: Kept separate from ``film`` rather than folded into its name: when prints start
    #: drifting for no visible reason, a batch change is the first thing to check, and a
    #: fact buried inside a free-text product name cannot be compared between profiles.
    film_batch: str = ""
    working_space: Space = "srgb"
    provisional: bool = True
    driver_settings: dict[str, Any] = field(default_factory=dict)
    blocker: dict[str, Any] = field(
        default_factory=lambda: {"model": "fixed_hue", "rgb": None, "saturation": None}
    )
    exposure: dict[str, Any] = field(default_factory=dict)
    #: How the wedge was scanned: device, colour space, bit depth, ppi, and whether the
    #: scanner's own corrections were off. A curve measured through a different scan path
    #: is not comparable to this one, so the path has to travel with the profile rather
    #: than live in someone's memory. Declared rather than left as a stray JSON key: an
    #: undeclared key is dropped on the first load/save round-trip, which would strip it
    #: from every profile cloned to seed a tricolour set.
    scan_settings: dict[str, Any] = field(default_factory=dict)
    lut: Lut = field(default_factory=Lut.identity)
    measurements: dict[str, Any] = field(default_factory=lambda: {"raw_patches": [], "scan_date": None})

    # ------------------------------------------------------------------ validation

    def validate(self) -> list[str]:
        """Return a list of problems; empty means valid."""
        problems: list[str] = []
        if not self.name.strip():
            problems.append("profile has no name")
        if self.working_space not in SPACES:
            problems.append(f"working_space {self.working_space!r} is not one of {SPACES}")
        if not isinstance(self.provisional, bool):
            problems.append("provisional must be true or false")

        model = self.blocker.get("model")
        if model not in ("fixed_hue", "zone_hue"):
            problems.append(f"unknown blocker model {model!r} (expected 'fixed_hue' or 'zone_hue')")

        def _bad_rgb(value) -> bool:
            return (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or not all(isinstance(v, (int, float)) and 0 <= v <= 255 for v in value)
            )

        if model == "zone_hue":
            zones = self.blocker.get("zones")
            if not isinstance(zones, list) or not zones:
                problems.append("a zone_hue blocker needs a non-empty 'zones' list")
            else:
                ns = []
                for z in zones:
                    if not isinstance(z, dict) or "n" not in z or "rgb" not in z:
                        problems.append(f"zone entry must have 'n' and 'rgb', got {z!r}")
                        continue
                    if not isinstance(z["n"], (int, float)) or not 0.0 <= z["n"] <= 1.0:
                        problems.append(f"zone n must be within [0, 1], got {z['n']!r}")
                    if _bad_rgb(z["rgb"]):
                        problems.append(f"zone rgb must be three 0-255 values, got {z['rgb']!r}")
                    ns.append(z.get("n"))
                if len(set(ns)) != len(ns):
                    problems.append("zone control points must have distinct n values")
        else:
            rgb = self.blocker.get("rgb")
            if rgb is not None and _bad_rgb(rgb):
                problems.append(f"blocker rgb must be three 0-255 values, got {rgb!r}")
            sat = self.blocker.get("saturation")
            if sat is not None and not (isinstance(sat, (int, float)) and 0.0 <= sat <= 1.0):
                problems.append(f"blocker saturation must be within [0, 1], got {sat!r}")
            if not self.provisional and (rgb is None or sat is None):
                problems.append("a non-provisional profile must have measured blocker rgb and saturation")

        if len(self.lut.values) < 2:
            problems.append("lut must hold at least two values")
        elif np.any(np.diff(self.lut.values) < 0):
            problems.append("lut is not monotonic — run enforce_monotonic before saving")
        if np.any(self.lut.values < 0) or np.any(self.lut.values > 1):
            problems.append("lut values must lie within [0, 1]")
        return problems

    @property
    def is_ready_to_print(self) -> bool:
        """True when the profile specifies everything a negative needs (even provisionally)."""
        if self.blocker.get("model") == "zone_hue":
            return bool(self.blocker.get("zones"))
        return self.blocker.get("rgb") is not None and self.blocker.get("saturation") is not None

    # ------------------------------------------------------------------ JSON round-trip

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "paper": self.paper,
            "chemistry": self.chemistry,
            "printer": self.printer,
            "media_type": self.media_type,
            "film": self.film,
            "film_batch": self.film_batch,
            "working_space": self.working_space,
            "provisional": self.provisional,
            "driver_settings": self.driver_settings,
            "blocker": self.blocker,
            "exposure": self.exposure,
            "scan_settings": self.scan_settings,
            "lut": {"size": self.lut.size, "values": [round(float(v), 9) for v in self.lut.values]},
            "measurements": self.measurements,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Profile:
        try:
            lut_d = d.get("lut") or {}
            values = lut_d.get("values") or []
            lut = Lut(np.asarray(values, dtype=np.float64)) if values else Lut.identity(
                int(lut_d.get("size") or DEFAULT_SIZE)
            )
            return cls(
                name=d["name"],
                paper=d.get("paper", ""),
                chemistry=d.get("chemistry", ""),
                printer=d.get("printer", "Epson ET-1810"),
                media_type=d.get("media_type", ""),
                film=d.get("film", ""),
                film_batch=d.get("film_batch", ""),
                working_space=d.get("working_space", "srgb"),
                provisional=d.get("provisional", True),
                driver_settings=d.get("driver_settings", {}),
                blocker=d.get("blocker", {"model": "fixed_hue", "rgb": None, "saturation": None}),
                exposure=d.get("exposure", {}),
                scan_settings=d.get("scan_settings", {}),
                lut=lut,
                measurements=d.get("measurements", {"raw_patches": [], "scan_date": None}),
            )
        except KeyError as e:
            raise ProfileError(f"profile is missing required field {e}") from e

    def save(self, path: str | Path) -> Path:
        problems = self.validate()
        if problems:
            raise ProfileError("refusing to save an invalid profile: " + "; ".join(problems))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Profile:
        path = Path(path)
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ProfileError(f"{path.name} is not valid JSON: {e}") from e
        profile = cls.from_dict(d)
        problems = profile.validate()
        if problems:
            raise ProfileError(f"{path.name} is invalid: " + "; ".join(problems))
        return profile


def list_profiles(directory: str | Path = PROFILE_DIR) -> list[Path]:
    directory = Path(directory)
    return sorted(directory.glob("*.json")) if directory.is_dir() else []
