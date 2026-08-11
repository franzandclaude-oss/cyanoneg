# Tricolour Plan — Revisions B1 and A1–A8

**Written:** 11 August 2026
**Against:** `TRICOLOUR_CONVERTER_PLAN_2026-08-11.md`
**Companion to:** `reviews/TRICOLOUR_PLAN_SECOND_REVIEW_2026-08-11.md` (item numbering follows its §7)

**Scope.** This document specifies the items that need no decision from the operator: **B1** (exposure source-of-truth) and **A1–A8** (implementation acceptance criteria). It is written as a specification to code against, not as commentary.

**Deliberately not covered — pending answers:**

| Pending | Blocks |
|---|---|
| B2 — Stage 2 scan protocol | A7's scan-step wording only |
| P1 — blocked control region | — |
| P2 — per-layer `scale` field | — |
| P3 — yellow wedge 16×k4 vs 32×k2 | A8's worked example only |
| P4 — readable-wedge criterion | — |

Neither dependency is structural; both are a sentence each once the answers land.

**One new finding surfaced while specifying A5.** It is a bug on `main`, not a plan defect, and it lands squarely on Stage 0. See §F5.

---

## F5 — `Profile` silently drops unknown JSON keys on round-trip

**Severity: high for this project.** Reproduced on `origin/tricolour` @ `b829f26`:

```
source keys : [..., 'provisional', 'scan_settings', 'working_space']
roundtrip   : [..., 'provisional', 'working_space']
LOST on round-trip: ['scan_settings']
```

**Mechanism.** `Profile.to_dict` ([profiles.py:126](cyanoneg/profiles.py:126)) writes a fixed 14-key literal and `from_dict` ([profiles.py:143](cyanoneg/profiles.py:143)) reads a fixed 14-key list. `scan_settings` exists in `profiles/CassArt 300 Sm.json` but is not a declared field of the `Profile` dataclass, so it is read past and never written back. No warning.

**Why it matters here specifically.** Stage 0's `seed_provisional_set` clones the base profile three times and saves each. **All three tricolour layer profiles would be born without the scan-path record** — and `HANDOFF.md` is explicit that the scan path is what makes a comparison meaningful:

> SilverFast raw, converted in Photoshop — a different scan path makes the comparison meaningless.

The profile currently records `device: Epson Perfection V600`, colour space, bit depth, ppi and `auto_corrections: all off`. Losing that on the profiles used to calibrate a three-session darkroom cycle is exactly the silent-error class this codebase exists to remove.

It also defeats A5 and A6 before they start: a fingerprint computed over declared fields cannot detect a scan-path mismatch it cannot see.

### Required fix

1. **Declare `scan_settings`** as a `Profile` field (`dict[str, Any]`, default `{}`) and add it to `to_dict` / `from_dict`.
2. **Add a round-trip regression test** that asserts no source key is lost, written so it catches the *next* undeclared field too:

```python
def test_profile_roundtrip_preserves_every_key(tmp_path):
    src = json.loads(Path("profiles/CassArt 300 Sm.json").read_text(encoding="utf-8"))
    out = tmp_path / "rt.json"
    Profile.load("profiles/CassArt 300 Sm.json").save(out)
    rt = json.loads(out.read_text(encoding="utf-8"))
    assert set(src) - set(rt) == set(), f"round-trip dropped {sorted(set(src) - set(rt))}"
```

3. **Do this before `seed_provisional_set` is written**, or the three seeded profiles inherit the loss.

A catch-all `extras` dict was considered and rejected: `scan_settings` is a real field with a real meaning that A5 needs to name explicitly, and an `extras` bag would let genuinely unknown keys accumulate unvalidated.

---

## B1 — Exposure source-of-truth

### Data model (one owner per value)

| Value | Lives in | Never appears in |
|---|---|---|
| `spe_seconds = 810` | each layer's `Profile.exposure` | the set file |
| `exposure_multiplier` | the set file, per layer | any `Profile` |
| computed working seconds | the emitted manifest | any `Profile`, the set file |

**The measured SPE is identical in all three layer profiles, including the provisional clones.** It is a property of paper, lamp and distance, not of the layer.

### Corrections to the plan text

| Location | Current | Replace with |
|---|---|---|
| §6 Stage 0 | "exposure = 810 s × multiplier" | "`spe_seconds` = 810, unchanged from the base profile" |
| §11 Verification 2 | "exposures 1215 / 2228 / 891 s" | "`spe_seconds` == 810 in all three profiles; multipliers 1.5 / 2.75 / 1.1 in the set; the **manifest** shows 1215 / 2228 / 891 s" |

### Single computation point

```python
def layer_exposure(profile: Profile, layer: TricolourLayer) -> dict:
    """The only place a multiplier is ever applied. Raises if the base SPE is absent."""
```

Emitted per layer into the manifest:

```json
"exposure": {
  "base_spe_seconds": 810,
  "exposure_multiplier": 2.75,
  "computed_seconds": 2227.5,
  "instruction_seconds": 2228,
  "instruction_display": "37:08"
}
```

### Rounding policy

`instruction_seconds = floor(computed_seconds + 0.5)` — round half up, to whole seconds. Then `instruction_display` renders `mm:ss`.

`computed_seconds` is rounded to **one decimal place** before serialisation. This is not fussiness: `810 * 1.1` evaluates to `891.0000000000001` in float64, and an artefact meant to be read by a person and diffed between runs should not carry that. Rounding to 0.1 s is far below any meaningful darkroom precision and makes the manifest stable.

The display form is not decoration: `HANDOFF.md` records the measured SPE as **"13:30 = 810 s"**, so the darkroom timer is already operated in `mm:ss`. Rendering it removes a mental conversion at the moment of greatest cost.

Worked, and self-checking against the recorded SPE:

| Layer | × | computed | instruction | display |
|---|---|---|---|---|
| — (base SPE) | 1.0 | 810 | 810 | **13:30** ✓ matches `HANDOFF.md` |
| Magenta | 1.5 | 1215 | 1215 | 20:15 |
| Yellow | 2.75 | 2227.5 | 2228 | 37:08 |
| Cyan | 1.1 | 891 | 891 | 14:51 |

### Guards

- `make_tricolour` reads the multiplier **only** from the set, never from a profile.
- `TricolourSet.validate()` rejects the set if the three resolved profiles disagree on `exposure.spe_seconds` — that disagreement is what makes a multiplier meaningless.
- `seed_provisional_set` writes the base SPE unmodified.

### Tests

```python
def test_seeded_profiles_keep_base_spe():        # all three == 810, none multiplied
def test_manifest_exposure_is_computed_once():   # 810 / 2.75 / 2227.5 / 2228 / "37:08"
def test_set_rejects_disagreeing_spe():          # one layer at 900 → validate() error
def test_no_double_multiplication():             # multiplier appears in set only, never in a profile JSON
```

The last is the one that guards the future: it asserts the *absence* of the multiplier from profile JSON, so a later refactor that "helpfully" caches it there fails immediately.

---

## A1 — Crop-before-detect contract, with a runtime guard

### The contract

```
full scan of the processed sheet
  → read <stem>_tricolour.json → placement[owning wedge slot]
  → crop to that rectangle
  → detect_fiducials(crop, wedge_sidecar)
  → sample_cells(crop, wedge_sidecar, frame, quantity=RESPONSE_QUANTITY[role])
```

Detection **never** runs on a full composed page. Per §4 F3 of the second review, doing so does not degrade — it returns a meaningless homography built from the picture's top corners and the wedge's bottom corners.

### The guard

Documentation is insufficient; the failure is silent and the resulting curve looks plausible. `detect_fiducials` must reject a span it cannot have produced.

**Primary check (scan has a ppi tag).** `_expected_fiducial_size` already converts sidecar print-px to scan-px via scan ppi; reuse that conversion for the *span* between fiducial centres:

```python
ratio = observed_span_scan_px / expected_span_scan_px
if not 0.85 <= ratio <= 1.15:
    raise AnalysisError(
        f"fiducial span is {ratio:.1f}x the size this target should be — "
        "detection is probably running on a whole page rather than a cropped "
        "target slot; crop to the manifest placement first"
    )
```

On the §4.6 page this ratio is roughly 3.5× for a 60 mm wedge slot read against the full sheet, so it triggers decisively.

**Fallback (no ppi tag).** Compare the scale-invariant ratio of fiducial blob size to fiducial span against the sidecar's own value, same ±15% tolerance. For the 60 mm wedge that ratio is ~0.077; measured across the whole page it collapses to ~0.02.

**Aspect ratio is not a usable check** and should not be proposed: the wedge's fiducial span is ~52 mm square (aspect ≈ 1.0), and a full-page mis-pick spans ~190 × 200 mm (aspect ≈ 0.95). The two are indistinguishable by aspect. Scale is the discriminator, not shape.

### Tests

```python
def test_full_page_detection_raises():    # composed page + wedge sidecar → AnalysisError naming the span
def test_cropped_slot_detection_works():  # crop per manifest → four expected corners, hollow one top-left
def test_guard_survives_missing_ppi():    # same page, ppi tag stripped → still raises via the fallback
```

---

## A2 — Correct R6's stated rationale

R6 argues the C/M/Y glyph must sit at ~50% blocker coverage or it becomes a fifth `detect_fiducials` candidate.

**Keep the constraint. Correct the reasoning.** Candidates must already pass ([analyze.py:238](cyanoneg/analyze.py:238)):

```python
candidates = [b for b in blobs if b.squareness > 0.55 and b.fill > 0.45]
```

plus the `_FID_SIZE_BAND` filter. A letterform is unlikely to pass `squareness` or `fill`, so the coverage figure is **defence in depth, not the sole barrier**.

The correction matters because R6 as written invites a future reader to treat ~50% as a precisely load-bearing number and to spend effort tuning it. It is a cheap belt-and-braces constraint and should be labelled as one.

---

## A3 — Process invariants derived from constants

### Constants (authoritative)

```python
CHANNEL_INDEX  = {"cyan": 0, "magenta": 1, "yellow": 2}
SOURCE_CHANNEL = {"cyan": "red", "magenta": "green", "yellow": "blue"}
PRINT_ORDER    = ("magenta", "yellow", "cyan")
```

### `TricolourSet.validate()` must reject

1. `layers.keys() != {"cyan", "magenta", "yellow"}` — exactly three, exactly these.
2. `source_channel` present and `!= SOURCE_CHANNEL[role]`.
3. `print_order` present and `!= PRINT_ORDER.index(role) + 1`.
4. Resolved profiles disagreeing on `exposure.spe_seconds` (B1).
5. Resolved profiles disagreeing on the calibration identity (A5).

Uniqueness checks are **not** sufficient: `{magenta: 3, yellow: 2, cyan: 1}` is unique and wrong in the way that destroys a sheet.

`TricolourLayer.role` is dropped — the containing dict key is authoritative and a second copy can only ever disagree with it.

### Tests

```python
@pytest.mark.parametrize("role,bad", [("magenta","red"), ("yellow","green"), ("cyan","blue")])
def test_rejects_wrong_source_channel(role, bad): ...
def test_rejects_non_myc_print_order():   # C → M → Y is rejected
def test_rejects_extra_or_missing_layer():
```

---

## A4 — `clipped_fraction`, defined exactly

### Definition

> **`clipped_pixel_fraction`** — the fraction of image pixels for which **at least one** RGB component lies outside `[0, 1]` after saturation scaling and **before** clipping.

Denominator is `h × w` (pixels, not components). Measured in linear light, since the scale is applied about the Rec.709 grey axis in linear light.

Also record, as diagnostic only:

> **`clipped_component_fraction`** — out-of-range components ÷ `h × w × 3`.

The **2% warning threshold applies to `clipped_pixel_fraction`**, making it deterministic and testable.

### Signature

```python
def step_saturate(image: Image, amount: float) -> tuple[Image, dict]:
    """Returns (image, {"clipped_pixel_fraction": …, "clipped_component_fraction": …})."""
```

Note this is a change from §5.1's `-> tuple[Image, float]`; the two fractions are worth carrying together and a bare float invites the wrong one being compared to 2%.

### Test

```python
def test_clipped_fraction_exact():
    # 4 pixels: one clips R only, one clips R and B, two in range.
    # clipped_pixel_fraction     == 0.5   (2 of 4 pixels)
    # clipped_component_fraction == 0.25  (3 of 12 components)
```

Asserting both against a hand-counted array is what stops the two definitions being quietly swapped later.

---

## A5 — Calibration identity by whitelist, not by field list

**Depends on F5 being fixed first.** A fingerprint cannot cover a field the round-trip discards.

### Approach: invert the check

A handwritten list of fields that must *agree* rots — a new field added next month is unchecked by default, and unchecked-by-default is the wrong direction for this codebase. Enumerate instead the fields **allowed to differ**:

```python
LAYER_MAY_DIFFER = frozenset({
    "name",          # "… — Magenta"
    "chemistry",     # madder tone vs carbonate bleach vs classic
    "lut",           # identical at Stage 0, divergent after Stage 3
    "measurements",  # each layer's own wedge
    "provisional",   # layers may be promoted independently
})

def shared_calibration_identity(p: Profile) -> dict:
    """Every to_dict() key not in LAYER_MAY_DIFFER."""

def calibration_fingerprint(p: Profile) -> str:
    """sha256 of canonical (sorted-key, separator-normalised) JSON of the above; first 16 hex."""
```

Everything else — `paper`, `film`, `film_batch`, `working_space`, `printer`, `media_type`, `driver_settings`, `blocker`, `exposure`, **`scan_settings`** — must agree across the three layers. A field added later is covered automatically.

### Test

The test must assert the *policy*, not the current field list:

```python
def test_new_profile_field_is_covered_by_default():
    """A field added to Profile and absent from LAYER_MAY_DIFFER must be required to agree."""
    covered = set(Profile("x").to_dict()) - LAYER_MAY_DIFFER
    assert covered == set(shared_calibration_identity(Profile("x")))
```

---

## A6 — Manifest that outlives the profiles

A profile *name* does not preserve a run: the named file can be recalibrated or overwritten the same afternoon.

### Per layer

- `profile_name`
- `calibration_fingerprint` (A5)
- `calibration_identity` — the full snapshot embedded, so the manifest is readable without the profile
- `lut`: `size` plus the values embedded (256 floats ≈ 3 KB per layer; ~9 KB total is cheap for full reproducibility)
- `provisional`
- `exposure` block (B1)

### Per run

`app_version` / git commit where available · source path, pixel dimensions and sha256 · resolved orientation · output pixel dimensions · ppi · `raw_invert` state · `saturation_boost` · both clip fractions (A4) · flip state · full set-file snapshot · wedge slot placements · fiducial geometry.

### Test

```python
def test_manifest_survives_profile_mutation(tmp_path):
    # generate → mutate the named profile on disk → the manifest still identifies
    # the calibration that actually ran, and its fingerprint no longer matches the file
```

---

## A7 — Separate the wall sheet from the record

§5.6 asks the JSON manifest to "read as instructions, not as a data structure." Those are two artefacts.

- **`<stem>_tricolour.json`** — authoritative, machine-readable, optimised for reproducibility (A6).
- **`<stem>_tricolour.md`** — generated alongside, optimised for error prevention at 2 a.m.

### Wall sheet shape

Ordered by `PRINT_ORDER`, times in `mm:ss` with seconds in parentheses:

```
TRICOLOUR — <set name> — <stem>
Sheet: A4 · picture 130 × 100 mm · pre-shrunk

1. MAGENTA   negative <stem>_1M.tif
   Sensitizer 10/10 · expose 20:15 (1215 s)
   Wash → sodium carbonate bleach → madder-root tone
   [scan step — pending B2]

2. YELLOW    negative <stem>_2Y.tif
   Sensitizer 10/10 · expose 37:08 (2228 s)
   Heavy exposure → wash → carbonate bleach to Fe(III) hydroxide
   [scan step — pending B2]

3. CYAN      negative <stem>_3C.tif
   Sensitizer 5/5 (1:1 dilute) · expose 14:51 (891 s)
   Classic cyanotype → wash

ORDER IS NOT NEGOTIABLE. Carbonate bleach destroys Prussian blue; cyan is last.
```

The scan instructions are bracketed because their content is exactly what B2 decides. Everything else here is settled.

---

## A8 — Knot count tied to level count, without disturbing the mono path

### The constraint that matters most

`derive_correction`'s `knots: int = 21` default ([lut.py:241](cyanoneg/lut.py:241), applied via `Lut.smooth_pchip` at [lut.py:287](cyanoneg/lut.py:287)) is **currently in use by a mono calibration that is verified on paper**. The existing 32-level workflow must not change.

So: **`derive_correction`'s default stays 21 and is not touched.** The tricolour path passes `knots` explicitly.

*(Minor correction for the plan: Q2 describes `smooth_pchip` as though it were module-level. It is a method on `Lut` at [lut.py:145](cyanoneg/lut.py:145). The substance of Q2 — that the 21-knot default is inherited — is correct.)*

### The rule

```python
def correction_knots(levels: int) -> int:
    """Knots for a wedge of `levels` distinct levels."""
    return max(5, min(21, round(levels * 2 / 3)))
```

Chosen because it is continuous with both known-good anchor points rather than invented:

| levels | knots | anchor |
|---|---|---|
| 32 | **21** | reproduces the existing verified default exactly |
| 16 | **11** | reproduces the first review's recommendation exactly |

A rule that passes through both existing reference points is defensible in a way a fresh constant is not — the same standard R1 applies to curves.

### Tests

```python
def test_correction_knots_preserves_mono_default():
    assert correction_knots(32) == 21     # existing verified behaviour, unchanged

def test_correction_knots_for_sixteen_levels():
    assert correction_knots(16) == 11

def test_mono_path_still_uses_the_default():
    """derive_correction's signature default is still 21 — the mono workflow is untouched."""
```

**Depends on P3 only for which row of the table print #1 exercises.** If the yellow slot goes to 32 × k2, its knot count is 21 and only magenta and cyan use 11. The rule is unchanged either way.

---

## Summary of code changes specified here

| File | Change | Item |
|---|---|---|
| `cyanoneg/profiles.py` | declare `scan_settings`; add to `to_dict`/`from_dict` | F5 |
| `cyanoneg/profiles.py` | `LAYER_MAY_DIFFER`, `shared_calibration_identity`, `calibration_fingerprint` | A5 |
| `cyanoneg/analyze.py` | fiducial span guard in `detect_fiducials` | A1 |
| `cyanoneg/lut.py` | `correction_knots`; **default `knots=21` untouched** | A8 |
| `cyanoneg/tricolour.py` | `layer_exposure`; constants; `validate()` rules; `step_saturate` returning two fractions | B1, A3, A4 |
| `cyanoneg/tricolour.py` | manifest emission; `<stem>_tricolour.md` wall sheet | A6, A7 |
| plan document | §6 Stage 0 and §11 Verification 2 corrections; R6 rationale | B1, A2 |

**Ordering note.** F5 must land before `seed_provisional_set` is written, and A5 depends on F5. Everything else here is independent.
