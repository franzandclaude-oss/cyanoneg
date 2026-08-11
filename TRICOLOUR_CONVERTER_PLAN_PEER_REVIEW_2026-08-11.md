# Peer Review — Tricolour Cyanotype Support for `cyanoneg`

**Reviewed:** 11 August 2026  
**Source plan:** `TRICOLOUR_CONVERTER_PLAN_2026-08-11.md`  
**Recommendation:** **Approve with required revisions before implementation**

## Executive summary

The plan is strong and is close to being ready for implementation. The overall architecture is sensible, the revisions materially improve the earlier design, and the proposed testing strategy is unusually careful about the kinds of silent errors that would otherwise waste expensive multi-session darkroom work.

In particular, R1, R2 conceptually, R3, R4, direct channel extraction, preservation of the existing mono pipeline, common orientation resolution, and numeric verification of channel mapping are all sound decisions.

However, two internal contradictions should be resolved **before coding starts**, because they affect the calibration protocol and data model:

1. The Stage 2 scan protocol conflicts with the stated purpose of isolated wedges in R2.
2. The handling of `spe_seconds` conflicts with the stated single-source-of-truth exposure model.

Several additional implementation contracts should also be tightened now to prevent ambiguity during coding.

---

## Required revisions before coding

### 1. Resolve the Stage 2 contradiction with isolated wedges

R2 says each layer's wedge is isolated by having the other two negatives place full blocker over that slot. One of the main benefits claimed for this design is that an isolated magenta patch can survive the **full remaining process**—yellow chemistry followed by cyan sensitizing/washing—while still being measured as magenta alone.

That is a valuable experiment because it measures the layer after the subsequent chemistry that may alter it.

However, §6 Stage 2 currently says:

> scan each layer's wedge after that layer is processed and before the next is coated

and §8 says:

> Do not scan after cyan. The earlier layers are gone.

Those instructions conflict with R2's stated rationale.

If the isolation scheme works as intended, the final sheet should still contain:

- a magenta-only wedge that has experienced all later processing;
- a yellow-only wedge that has experienced the cyan cycle;
- a cyan-only wedge.

Therefore the plan should explicitly distinguish **diagnostic intermediate scans** from the **production calibration measurement**.

### Recommendation

Use intermediate scans where practical to diagnose what each subsequent process does, but derive the production per-layer corrections from the isolated wedges after the complete M → Y → C process.

For example:

1. Process magenta.
2. Optional diagnostic scan of the M wedge.
3. Process yellow.
4. Optional diagnostic scan of M and Y wedges.
5. Process cyan.
6. Final scan of all three isolated wedges.
7. Derive the three production corrections from the final isolated measurements.

If there is a physical or chemical reason that the magenta/yellow slots cannot be measured reliably after cyan, document it explicitly. In that case R2's claim that the isolated wedge measures survival through the full remaining process must also be revised.

This issue should be settled before implementation because it determines what the calibration workflow and manifest need to represent.

---

### 2. Resolve the exposure source-of-truth contradiction

§5.2 describes a clean model:

- `spe_seconds = 810` lives in the layer profile;
- the exposure multiplier lives in the tricolour set;
- computed exposure seconds live only in the emitted manifest.

That is the preferable design.

Stage 0, however, says that provisional cloned profiles have:

> exposure = 810 s × multiplier

and Verification step 2 expects the profile JSONs to contain:

- 1215 s;
- 2228 s;
- 891 s.

Those are different data models.

### Recommendation

Keep the measured SPE unchanged in every provisional layer profile:

```text
spe_seconds = 810
```

Keep only these values in the set:

```text
magenta multiplier = 1.5
yellow multiplier  = 2.75
cyan multiplier    = 1.1
```

Compute working exposure at generation time:

```text
magenta = 810 × 1.5  = 1215 s
yellow  = 810 × 2.75 = 2227.5 s
cyan    = 810 × 1.1  = 891 s
```

The manifest should contain the computed result.

Also specify timer rounding explicitly. For example:

```json
{
  "base_spe_seconds": 810,
  "exposure_multiplier": 2.75,
  "computed_seconds": 2227.5,
  "instruction_seconds": 2228
}
```

This avoids both hidden rounding and a future double-application of the multiplier.

---

## Strongly recommended implementation clarifications

### 3. Do not make fixed process invariants freely configurable

The plan says the following are non-negotiable:

- exactly three layers;
- red → cyan;
- green → magenta;
- blue → yellow;
- physical print order M → Y → C.

Yet the proposed set JSON stores both `source_channel` and `print_order`, while `TricolourLayer` also stores `role` even though the containing dictionary supplies the role.

That allows contradictory states to be represented, such as:

```json
"magenta": {
  "source_channel": "red",
  "print_order": 3
}
```

### Recommendation

Prefer deriving these process invariants from constants:

```python
CHANNEL_INDEX = {
    "cyan": 0,
    "magenta": 1,
    "yellow": 2,
}

PRINT_ORDER = ("magenta", "yellow", "cyan")
```

The set should primarily contain values that genuinely vary.

If `source_channel` and `print_order` remain in JSON for readability, `validate()` should require them to match the mandated constants exactly. Merely checking uniqueness is not enough.

Likewise, avoid storing `role` redundantly inside a `TricolourLayer` when its dictionary key already establishes the role unless there is a concrete serialization reason to do so.

---

### 4. Verify that full blocker remains an effective mask at 2.75× exposure

R2 and R3 depend on:

> full blocker → no meaningful exposure

The blocker `(255, 64, 0)` is measured and should absolutely be retained instead of replacing it with unmeasured book values.

However, its calibration is associated with the measured 810 s SPE. The yellow layer operates at approximately 2228 s.

A blocker that is sufficiently opaque at 810 s has not automatically been demonstrated to remain effectively opaque at 2.75× that duration.

This matters for:

- the supposedly protected page background;
- the two non-owner wedge slots;
- the validity of isolated wedge measurements.

### Recommendation

Include a **blocked-control region** in print #1, exposed during the longest yellow exposure.

The control can be small. Its purpose is simply to answer:

> Does maximum blocker coverage remain chemically negligible at the longest working exposure?

This is inexpensive insurance compared with discovering after three wet cycles that supposedly isolated patches received measurable yellow exposure.

---

### 5. Define wedge-fiducial handling on the composed page

The plan carefully prevents the C/M/Y glyph from becoming an extra candidate for `detect_fiducials`.

The same attention should be given to the wedge furniture.

The composed A4 page may contain:

- picture frame/fiducials;
- wedge target borders/fiducials;
- glyph;
- dark patches.

The implementation must make clear **where fiducial detection runs**.

### Recommendation

If the intended scan-back workflow is:

```text
full scan
→ use manifest placement
→ crop owning wedge slot
→ detect wedge fiducials inside crop
→ sample cells
```

state this explicitly.

Add an end-to-end synthetic test using a **composed tricolour page**, rather than testing only a framed picture without wedge furniture.

The test should prove that the actual scan-back path selects the intended four fiducials and cannot silently substitute page furniture or another target's marks.

---

### 6. Document linear-light saturation as an implementation choice

The source requirement is approximately:

> boost saturation 30–50%

The plan translates this into:

> scale about the Rec.709 grey axis in linear light

That may be a good implementation, but it is an interpretation rather than a direct process requirement. It will not necessarily behave like a conventional perceptual saturation control.

### Recommendation

Document the distinction explicitly.

Also define `clipped_fraction` precisely. A useful definition is:

> fraction of image pixels for which at least one RGB component lies outside `[0, 1]` after saturation scaling and before clipping.

Then the proposed 2% warning threshold becomes deterministic and testable.

Consider recording both:

```text
clipped_pixel_fraction
clipped_component_fraction
```

if diagnostic detail is useful, although the first is sufficient for the initial implementation.

---

### 7. Strengthen profile compatibility checking

The proposed validation checks:

- paper;
- film;
- film batch;
- working space;
- driver settings;
- blocker.

This is good, but a handwritten field list can become incomplete as the profile schema evolves.

Other shared apparatus conditions may eventually matter, such as:

- printer;
- UV source;
- exposure distance;
- warmup policy;
- other future calibration fields.

### Recommendation

Define a shared **calibration fingerprint** or canonical compatibility projection.

Conceptually:

```python
shared_calibration_identity(profile)
```

should return the fields that must be identical across the three layer profiles.

An alternative is to define an explicit whitelist of fields that are *allowed to differ* between layers and require all other calibration-critical fields to agree.

The goal is to avoid introducing a new profile field later and forgetting to update `TricolourSet.validate()`.

---

### 8. Make the manifest independently reproducible

The manifest is intended to preserve an expensive physical run.

A profile name alone is not sufficient if that named profile can later be recalibrated or overwritten.

### Recommendation

Record a stable identity for every resolved profile, such as:

- profile hash/fingerprint;
- profile version if one exists;
- or an embedded snapshot of the calibration-critical fields.

Also consider recording:

- resolved tricolour set snapshot;
- source image dimensions;
- resolved orientation;
- output pixel dimensions;
- PPI;
- raw-invert state;
- saturation boost;
- clipping fraction;
- flip state;
- blocker;
- base SPE;
- exposure multiplier;
- exact computed exposure;
- rounded darkroom instruction exposure;
- application/version or commit identifier where available.

The manifest should be sufficient to explain exactly how a successful negative was generated even if the external profile files later change.

---

### 9. Separate machine-readable manifest from darkroom instructions

The plan says the JSON manifest:

> should read as instructions, not as a data structure.

Those are somewhat conflicting goals.

JSON is useful as the authoritative machine-readable record. A darkroom operating sheet has different usability requirements.

### Recommendation

Keep:

```text
<stem>_tricolour.json
```

as the authoritative run manifest.

Then either generate a small additional:

```text
<stem>_tricolour.md
```

or:

```text
<stem>_tricolour.txt
```

or provide an explicit formatted instruction representation in the GUI.

The human-facing output should make the critical sequence immediately obvious, for example:

```text
1. MAGENTA — 1215 s — sensitizer 10/10
   Wash → carbonate bleach → madder-root tone

2. YELLOW — 2228 s — sensitizer 10/10
   Heavy exposure → wash → carbonate bleach to yellow

3. CYAN — 891 s — sensitizer 5/5, 1:1 dilute
   Classic cyanotype → wash
```

The JSON should optimize for reproducibility; the wall sheet should optimize for error prevention.

---

### 10. Implement core functionality before GUI integration

The plan lists the Process-tab mode as part of the print-#1 implementation.

The GUI changes, however, touch:

- mode state;
- profile/set selection;
- tracked settings;
- settings fingerprints;
- saturation controls;
- clipped-% display;
- preview behaviour;
- output generation;
- later calibration controls.

That is a relatively broad integration surface.

### Recommendation

Use two implementation milestones.

#### Milestone A — core

Implement and test:

1. `TricolourSet`;
2. validation;
3. `extract_channel`;
4. `step_saturate`;
5. `step_frame`;
6. `tricolour_page`;
7. `make_tricolour`;
8. provisional set seeding;
9. TIFF naming;
10. manifest generation;
11. numeric verification.

Exercise this through tests and a small direct/CLI harness.

#### Milestone B — GUI

Only after core output is verified:

1. add Mono/Tricolour mode;
2. set selection;
3. saturation UI;
4. three previews;
5. output controls;
6. calibration response-channel UI.

This preserves the proposed product design while making regressions much easier to isolate.

---

## Review of the open questions

### Q1 — Is redundancy 4 enough?

**Recommendation: use 16 levels × redundancy 4 for print #1.**

The first print is an exploratory calibration print rather than the final measurement standard.

Keeping all three isolated wedges on the same physical sheet has an important advantage: it avoids the documented sheet-to-sheet coating variation.

Increasing redundancy enough to force a wedge onto another sheet would compromise that advantage.

For print #1:

```text
levels = 16
redundancy = 4
```

is a reasonable compromise.

Add the blocked control described earlier, then inspect the actual within-level spread from the first scan. That provides evidence for deciding whether future targets require k=8 or k=16.

---

### Q2 — Is `derive_correction(knots=21)` over-parameterised for 16 levels?

**Yes. Do not blindly inherit the 21-knot default.**

With only 16 distinct measured levels, a correction model should not have greater effective flexibility than the measurement supports.

### Recommendation

Make knot count explicit for the tricolour workflow and tie it to the number of distinct measured levels.

A conservative first-print rule such as:

```python
knots = min(11, levels)
```

would be preferable to silently using 21.

The exact rule can later be refined using real residuals and duplicate-patch variance.

The important requirement for coding is that the tricolour workflow **must not accidentally inherit an inappropriate default**.

---

### Q3 — Should yellow levels be non-uniform?

**Recommendation: no, not for print #1.**

The plan's existing recommendation is correct.

A non-uniform yellow target could eventually be valuable, but before the first measurement it would encode assumptions about where the useful response lies.

Use the same uniform 16-level structure for all three layers on print #1.

Then inspect the yellow response and redesign its target only if the data demonstrates that useful separation is concentrated in a narrow region.

---

### Q4 — Reciprocity at 2.75× exposure

**Recommendation: defer the dedicated experiment as planned.**

The Stage 3 measurement occurs at the actual working exposure, so its measured response will include any reciprocity departure.

The wording should distinguish:

> compensating for the combined working-exposure response

from:

> measuring reciprocity itself.

The calibration can absorb the effect without identifying its physical cause.

A dedicated reciprocity experiment is therefore not required before print #1.

---

## Additional conceptual clarification for R1

R1 is a strong revision.

Using the measured CassArt LUT unchanged is much more defensible than inventing plausible-looking magenta/yellow shaping curves.

One sentence should nevertheless be softened.

The measured LUT is best described as:

> the best measured baseline for the shared ink/film/printer/paper system before the new layer-specific chemistry is characterised.

Avoid implying that it is necessarily the true tonal response at 1.5× or 2.75× exposure.

The exposure change itself may alter the response.

That does not weaken R1. It reinforces the reason print #1 is necessary.

---

## Additional tests recommended

In addition to the tests already proposed, add the following.

### Long-exposure blocker test

Construct or measure a blocked control corresponding to the maximum working exposure and verify that it remains below an agreed meaningful-deposition threshold.

This can initially be a physical acceptance test if it cannot be represented synthetically.

### Process-invariant validation

Explicitly test rejection of:

```text
magenta → red
yellow → green
cyan → blue
```

and any non-M→Y→C print order if those fields remain serialised.

### Exposure source-of-truth test

Given:

```text
SPE = 810
multiplier = 2.75
```

assert that:

- profile SPE remains 810;
- set multiplier remains 2.75;
- computed manifest exposure is 2227.5;
- the rounded instruction value follows the defined rounding policy.

This test guards against future double multiplication.

### Full composed-page fiducial test

Create a page containing:

- picture;
- glyph;
- owning wedge;
- non-owner slots;
- all associated furniture.

Run the actual scan-back geometry path and verify that the correct target fiducials are selected.

### Manifest reproducibility test

Generate a result, modify the named source profile afterward, and confirm that the emitted manifest still contains enough immutable information to identify the calibration that actually generated the negatives.

### Saturation clipping definition test

Use synthetic RGB values for which the expected pre-clipping out-of-range pixels are known exactly.

Assert the returned `clipped_fraction` against that known result.

---

## Suggested implementation gate

### No-go until resolved

The following should be corrected in the plan before coding starts:

- Stage 2 final-vs-intermediate wedge measurement protocol;
- `spe_seconds` versus exposure-multiplier ownership.

### Can be acceptance criteria during implementation

These do not require architectural redesign but should be specified before the relevant code is merged:

- fixed channel/order invariants;
- long-exposure blocker control;
- composed-page fiducial/cropping contract;
- exact clipping-fraction definition;
- profile compatibility fingerprint;
- reproducible manifest identity;
- human-readable darkroom instructions;
- knot-count policy.

---

## Recommended implementation sequence

A safe coding sequence would be:

1. Correct the plan's two contradictions.
2. Implement `TricolourSet` and invariant validation.
3. Implement `extract_channel`.
4. Implement and unit-test saturation and clipping reporting.
5. Implement frame/glyph generation.
6. Implement page composition and isolated wedge slots.
7. Implement manifest and exposure computation.
8. Implement provisional set generation.
9. Add full core tests.
10. Generate real TIFFs and run numeric channel checks.
11. Verify registration/page geometry.
12. Integrate the Process-tab GUI.
13. Produce print #1.
14. Evaluate blocker control and isolated wedges.
15. Finalise `analyze.py` quantity support.
16. Finalise `proof.py` quantity handling.
17. Derive measured layer profiles for print #2.

---

## Final recommendation

**Approve with required revisions.**

The design direction is good and does not need a major rewrite. The two contradictions identified above should be corrected before implementation begins because they affect the meaning of the calibration data rather than merely implementation detail.

The central engineering principle for the first physical run should be:

> **Print #1 is an instrumented experiment. Preserve enough controls and metadata that any failure can be attributed to channel mapping, exposure, masking, registration, saturation, or chemistry rather than merely observed.**

The existing plan is already close to this standard. Resolving the calibration-stage and exposure-model contradictions, and tightening the few remaining contracts above, should make it ready for coding.
