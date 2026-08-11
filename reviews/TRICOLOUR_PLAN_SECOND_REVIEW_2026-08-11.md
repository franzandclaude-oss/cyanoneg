# Second Review — Tricolour Cyanotype Support for `cyanoneg`

**Reviewed:** 11 August 2026
**Branch:** `origin/tricolour` @ `d2a3784` (branched cleanly from `origin/main` @ `48522dd`; no divergence)
**Documents under review:**
- `TRICOLOUR_CONVERTER_PLAN_2026-08-11.md` (623 lines) — the plan
- `TRICOLOUR_CONVERTER_PLAN_PEER_REVIEW_2026-08-11.md` (680 lines) — first peer review, verdict "approve with required revisions"

**Contents of branch:** the two documents above. **No code.** Nothing under `cyanoneg/`, `tests/`, or `profiles/` differs from `main`.

**Recommendation:** **Endorse the first review's verdict.** Both of its blocking items are real and must be fixed before coding. This review adds four findings neither document contains, one correction to the first review's proposed remedy, and one framing concern about how print #1's success is defined.

---

## 1. Scope and method

This review does two things the first review did not:

1. **Verifies the plan's claims against the actual code.** The plan cites roughly forty specific line numbers as load-bearing arguments. Those were checked by reading the source at `origin/tricolour`, not by inspection of the plan alone.
2. **Checks the plan's arithmetic.** The plan explicitly invites this ("Numbers are given so a reviewer can check the arithmetic rather than trust it," §4.6).

It does not re-litigate the design decisions the first review already endorsed (R1, R3, R4, direct channel extraction, preservation of the mono pipeline, common orientation resolution). Those are sound and this review agrees.

---

## 2. Verification log

Every claim below was checked against source on `origin/tricolour`. All passed.

| # | Plan's claim | Result |
|---|---|---|
| V1 | `sample_cells` declares `quantity: str = "lstar"` and never references it in the body ([analyze.py:361](cyanoneg/analyze.py:361)) | **Confirmed.** The parameter is declared and genuinely dead. The hook §5.3 depends on exists. |
| V2 | `calibration_page` builds a white canvas ([targets.py:547](cyanoneg/targets.py:547)) | **Confirmed.** `canvas = np.ones((page_h, page_w, 3), dtype=np.float32)`, with a comment explaining the margin is never measured so its value need only be legal. |
| V3 | `step_wedge` already computes full blocker ([targets.py:472](cyanoneg/targets.py:472)) | **Confirmed**, exactly as quoted in R3. |
| V4 | `proof.py` indexes `p["lstar"]` directly | **Confirmed** at `measured_response`; the `by_value` accumulation hardcodes `float(p["lstar"])`. |
| V5 | `soft_proof`'s fallback maps tone onto `PAPER_RGB`/`BLUE_RGB` | **Confirmed.** Meaningless for magenta or yellow, as R4 states. |
| V6 | `DEFAULT_LEVELS = 32`, `DEFAULT_REDUNDANCY = 16` | **Confirmed** ([targets.py:389](cyanoneg/targets.py:389)). |
| V7 | `step_wedge` defaults `patch_mm=5.5`, `border_mm=8.0` | **Confirmed.** |
| V8 | `_grid_shape(64)` returns 8 × 8 | **Confirmed by reading the scoring rule.** `score = abs((cols/rows) - target_aspect)` with `target_aspect=2.0`: 8×8 scores 1.0; 4×16 scores 1.75; 16×4 scores 2.0; 2×32 scores 1.94; 32×2 scores 14.0. 8×8 wins. |
| V9 | Wedge is 60 mm square | **Confirmed.** 8 × 5.5 mm + 2 × 8 mm = 60 mm, both axes. |
| V10 | Three wedges fit the A4 width | **Confirmed.** 3 × 60 + 2 × 5 = 190; (210 − 190) / 2 = 10 mm margin. |
| V11 | `calibration_page` raises rather than crops when parts do not fit | **Confirmed** ([targets.py:539](cyanoneg/targets.py:539)). |
| V12 | Exposure arithmetic | **Confirmed.** 810 × 1.5 = 1215; 810 × 2.75 = 2227.5; 810 × 1.1 = 891. |

**Conclusion:** the plan's technical claims about the codebase are accurate. This is worth stating plainly, because the plan's arguments rest on them and a reader would otherwise have to take forty citations on trust.

---

## 3. Agreement with the first review's blocking items

### 3.1 Exposure source-of-truth contradiction — **agreed, unambiguous**

§5.2 states the clean model: `spe_seconds = 810` lives in each layer's `Profile`, the multiplier lives in the set file, computed seconds appear only in the manifest. But Stage 0 (§6) says provisional profiles carry "exposure = 810 s × multiplier", and Verification step 2 (§11) expects to *see* 1215 / 2228 / 891 in the profile JSONs.

Implement both and the multiplier is applied twice. The first review's remedy is correct. Adopt it, including the explicit rounding policy (`computed_seconds: 2227.5`, `instruction_seconds: 2228`) and the guard test.

### 3.2 Stage 2 protocol contradiction — **agreed it is a contradiction; disagree with the proposed remedy**

The contradiction is real. R2 claims isolated wedges measure each layer "as it survives the **full remaining process**". §6 Stage 2 says to scan each layer before the next is coated. §8 says "Do not scan after cyan. The earlier layers are gone." Those cannot all be true.

The first review resolves it by **moving the production measurement to after cyan**, treating intermediate scans as optional diagnostics.

**This review recommends the opposite resolution.** The reason is that R2's *primary* argument does not depend on the survival claim at all:

> A stacked patch read through the green channel carries magenta's green absorption *plus* yellow's *plus* cyan's. `derive_correction` would then return a curve that linearises the neutral stack […] Three curves derived that way are not three per-layer curves; they are the same neutral-stack curve read three ways.

That argument is sufficient on its own to justify isolation. The "survives the full remaining process" benefit is a *second, unverified* claim bolted onto it — and it is precisely the claim §8 contradicts.

Moving the production measurement to after cyan bets the entire calibration on an untested assumption: that three sensitizer/wash cycles, an alkaline carbonate bleach, and a madder tone leave the isolated magenta patches readable in the green channel. If that assumption is wrong, the *only* production measurement is corrupt, and the failure is silent — a stained wedge still produces a smooth monotone curve.

**Recommended resolution:**

- Keep intermediate scans (scan each layer after processing, before the next coat) as the **production** measurement. This is the conservative choice: each patch is read at the moment it is least contaminated.
- Add a final post-cyan scan of all three slots as an **experiment**, not a measurement.
- **Demote R2's "Buy" bullet** from a stated benefit to an open question, phrased as such.
- The comparison between the intermediate and final scans is what settles the question empirically, and it is free.

This preserves R2 intact — isolation is still correct — while removing the unsupported claim and avoiding a single point of failure.

---

## 4. New findings

These appear in neither document.

### F1 — One control region answers three questions

The first review (item 4) asks for a blocked control to test whether blocker `(255, 64, 0)` remains effectively opaque at 2.75× SPE, since it was characterised at 810 s. That is correct and should be adopted.

**The same control patch, blocked on all three layers and read at the end, also measures the accumulated stain floor from three sensitizer-and-wash cycles.** That floor is exactly the quantity that determines whether the isolated magenta wedge is readable after cyan — i.e. it settles §3.2's open question with data rather than argument.

So a single small region, full blocker on all three negatives, delivers:

1. Does maximum blocker coverage stay chemically negligible at the longest working exposure? (first review's question)
2. What does three cycles of coating and washing deposit on nominally unexposed paper?
3. Is post-hoc reading of the isolated wedges viable at all? (§3.2's question)

Cheap, small, and print #1 is the only opportunity to capture it before the calibration decisions that depend on it. **This is the highest-value single addition to print #1.**

Recommend the control be read in all three channels (`lstar_r`, `lstar_g`, `lstar_b`), not just L\*, since a stain that is invisible in L\* may be significant in blue.

### F2 — Fiducials are a measurement instrument, and the plan uses them only as a visual check

§8 dismisses paper dimensional change as "not a software problem" and bounds it by keeping the image to 130 mm on the long edge. §4.1 describes registration verification as "a light table, not software."

The bound is honest but the residual is not small. At the plan's own stated 0.5–1% figure, across the 150 mm fiducial span of the framed picture, misregistration is **0.75–1.5 mm**. On a 130 mm tricolour image that is visible colour fringing across the entire picture, not an edge artefact.

The standard mitigation is a per-layer scale factor. The number to put in it comes from **scanning layer 1's printed fiducials after processing and comparing the measured span to nominal** — a measurement print #1 is uniquely positioned to provide and which nothing in the plan captures.

**Recommendation:**
- Add a `scale` field per layer to the set JSON, defaulting to `1.0` for print #1.
- Add "measure the printed fiducial span" to Stage 2.
- If the field is not added now, print #2 cannot use it either, because the measurement will not have been taken.

**Note the wedges are immune to this and the picture is not.** Each wedge is sampled through a homography derived from its own four fiducials, which absorbs any uniform scale change in the paper. Shrinkage therefore threatens registration of the picture only — which is worth stating in the plan, because it is a genuinely reassuring property that is not obvious.

### F3 — Fiducial detection on a composed page will fail reliably, not ambiguously

The first review (item 5) asks the plan to "state explicitly" where fiducial detection runs. That understates it. Reading `detect_fiducials` ([analyze.py:216](cyanoneg/analyze.py:216)):

```python
xs = [b.centre[0] for b in candidates]
ys = [b.centre[1] for b in candidates]
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
corner_points = {"a": (x0, y0), "b": (x1, y0), "c": (x0, y1), "d": (x1, y1)}
for key, (cx, cy) in corner_points.items():
    best = min(candidates, key=lambda b: (b.centre[0]-cx)**2 + (b.centre[1]-cy)**2)
```

It takes the bounding box of **all** candidate centres and picks the blob nearest each corner of that box. On the §4.6 page, the picture's fiducials sit at the top and the owning wedge's fiducials ~200 mm below. The bounding box spans both. The four corner picks land on picture-top-left, picture-top-right, wedge-bottom-left, wedge-bottom-right. The resulting homography is not marginal — it is meaningless.

This is not hypothetical. The code carries a comment recording this exact class of failure already occurring with a *single* target:

> the same sheet scanned two ways yielded 60 candidates one time and 82 the other, and in the second the corner picks landed on patches instead of fiducials

**Two further details neither document mentions:**

- **There is already a size-band filter** (`_expected_fiducial_size` + `_FID_SIZE_BAND`) added in response to that incident. It will *not* rescue the composed-page case, because both sets are genuine fiducials. Whether it filters one set out depends on whether the picture frame and the wedge use the same fiducial size, and on which sidecar was passed — accidental behaviour, not a guard.
- Without a ppi tag, `_expected_fiducial_size` falls back to "the largest cluster of similarly-sized candidates." On a composed page that could resolve to either set, silently.

**Recommendation:** the scan-back path must crop to the manifest-recorded slot *before* detection, and this needs a **runtime guard** — not merely documentation. Detection should refuse to run on an image whose dimensions match the full page. The first review's proposed composed-page test is necessary but insufficient on its own; a test proves the happy path works, a guard prevents the unhappy one.

**Minor correction to R6 while here:** R6 argues the C/M/Y glyph must be ~50% blocker coverage or it becomes a fifth candidate blob. The constraint is cheap and worth keeping, but the stated rationale is overstated — candidates must also pass `squareness > 0.55 and fill > 0.45`, which a letterform is unlikely to satisfy. R6 is belt-and-braces, not the sole defence. Worth correcting so a future reader does not treat the coverage figure as more load-bearing than it is.

### F4 — Q1 and Q3 have a fourth option neither document lists

The yellow layer is deliberately exposed at 2.75× SPE. Its wedge is printed through `linear.json`, uncorrected (§4.4), so it ramps from clear film to full blocker. At 2.75× SPE a large fraction of that ramp lands at Dmax, and patches at Dmax measure nothing. With 16 levels this could leave roughly five or six usable levels.

The plan sees this (Q3) and defers it to avoid encoding assumptions before measurement. That reasoning is sound and the first review endorses it. But the options considered under Q1 and Q3 are: change redundancy, change levels uniformly, redistribute levels non-uniformly, or move a wedge to a second sheet.

**A fourth option preserves the "no assumptions" virtue and directly addresses the compression risk: 32 levels × redundancy 2, for the yellow slot only.**

- Same 64 patches, therefore the same 8 × 8 grid and the same 60 mm slot — the §4.6 layout is untouched.
- Doubles level resolution exactly where compression is expected.
- Assumes nothing about *where* the response lives. It trades redundancy for resolution, rather than guessing at a distribution — which is precisely the objection that sank Q3's non-uniform proposal.

The cost is real: k=2 means a dust spike or coating flaw is no longer outvoted by a median over four copies, and `step_wedge`'s docstring argues hard for high k. Whether that trade is worth it depends on which failure is more likely on print #1 — noise, or a wedge that is 60% Dmax and measures nothing.

This also interacts with **Q2**: with 32 measured levels, `derive_correction`'s 21-knot default is no longer over-parameterised, so the yellow layer would not need the special knot rule. The first review's `knots = min(11, levels)` recommendation remains correct for the 16-level magenta and cyan wedges.

**This review does not make a firm recommendation between k=4/16 levels and k=2/32 levels for yellow.** It is a genuine trade and the person who will read the wedge should decide. But it should be an explicit decision rather than an unlisted option.

---

## 5. Framing concern

The plan defines print #1's success as producing three measured per-layer profiles (§6 Stage 3: "Save three measured profiles, flip `provisional: false`. Print #2 is calibrated").

Given that:

- the yellow wedge may be largely compressed into Dmax (F4),
- the isolated-wedge survival question is open (§3.2),
- the blocker's opacity at 2.75× is uncharacterised (first review item 4, F1),
- and R1 correctly notes the cloned LUT is *not* known to be the true response at 1.5× or 2.75×,

there is a material chance print #1 yields no directly usable curve for at least one layer.

That would still be a successful print, and the plan should say so. **Print #1's job is to make print #2's targets designable.** Framing it as "produce three measured profiles" creates pressure to derive curves from data that does not support them — which is the precise failure mode `proof.py`'s docstring exists to prevent, and which R1 was written to avoid:

> a curve invented from a plausible-looking shape "would be worse than none — it would look authoritative while being fiction."

A curve fitted to six usable levels and ten Dmax patches is the same fiction wearing a measurement's clothes. Recommend §6 state an explicit acceptance criterion for what makes a layer's wedge *readable*, and an explicit instruction to leave a layer `provisional: true` if its wedge does not meet it.

---

## 6. Items endorsed without further comment

The first review's remaining recommendations are sound and should be adopted as written:

| Item | Subject | Note |
|---|---|---|
| 3 | Process invariants derived from constants, not freely configurable | Agreed. If `source_channel` / `print_order` stay in JSON for readability, `validate()` must match them against the constants exactly. |
| 6 | Document linear-light saturation as an interpretation; define `clipped_fraction` precisely | Agreed. The proposed definition makes the 2% threshold testable. |
| 7 | Shared calibration fingerprint instead of a handwritten field list | Agreed, and it directly serves the codebase's stated purpose of removing silent-error classes. |
| 8 | Manifest records immutable profile identity, not just a name | Agreed. A named profile that can later be recalibrated does not preserve an expensive physical run. |
| 9 | Separate machine-readable JSON from the human darkroom sheet | Agreed. The plan's "should read as instructions, not as a data structure" is two conflicting goals in one artefact. |
| 10 | Milestone A (core) before Milestone B (GUI) | Agreed, and this is the single most useful process recommendation in the first review. |
| Q2 | Do not inherit the 21-knot default | Agreed for 16-level wedges. See F4 for the interaction. |
| Q3 | Uniform levels for print #1 | Agreed as to rejecting *non-uniform* distribution. See F4 for an option that is not non-uniform. |
| Q4 | Defer the reciprocity experiment; sharpen the wording | Agreed. |
| — | R1 softening ("best measured baseline", not "the true response") | Agreed and important. Feeds directly into §5. |

---

## 7. Recommended disposition

### Blocking — fix in the plan before any code is written

| # | Item | Source |
|---|---|---|
| B1 | Exposure source-of-truth: SPE stays 810 in profiles, multiplier only in the set, computed seconds only in the manifest. Correct Stage 0 and Verification step 2. Define rounding. | First review item 2 |
| B2 | Stage 2 protocol: keep intermediate scans as the production measurement; demote R2's "survives the full remaining process" to an open question; add the post-cyan scan as an experiment. | First review item 1, as amended by §3.2 |

### Required as print #1 design, before the sheet is generated

| # | Item | Source |
|---|---|---|
| P1 | Add the all-layer blocked control region; read it in all three channels. | F1 + first review item 4 |
| P2 | Add per-layer `scale` field (1.0 for print #1) and a fiducial-span measurement to Stage 2. | F2 |
| P3 | Decide explicitly between 16 × k4 and 32 × k2 for the yellow slot. | F4 |
| P4 | State an acceptance criterion for a "readable" wedge, and that failing it means the layer stays `provisional`. | §5 |

### Acceptance criteria during implementation

| # | Item | Source |
|---|---|---|
| A1 | Crop-before-detect contract **plus a runtime guard** refusing full-page detection. | F3, strengthening first review item 5 |
| A2 | Correct R6's rationale to acknowledge the `squareness`/`fill` filters. | F3 |
| A3 | Process invariants validated against constants. | First review item 3 |
| A4 | Exact `clipped_fraction` definition and test. | First review item 6 |
| A5 | Calibration fingerprint for profile compatibility. | First review item 7 |
| A6 | Reproducible manifest identity. | First review item 8 |
| A7 | Separate human-readable darkroom sheet. | First review item 9 |
| A8 | Knot-count policy tied to level count. | First review Q2 |

---

## 8. Overall assessment

The plan is of unusually high quality. Its citations are accurate, its arithmetic is correct, and its two strongest arguments — R1 (clone the measured LUT rather than invent shaping curves) and R2's primary argument (a stacked wedge yields one neutral-stack curve read three ways, not three per-layer curves) — are correct and not obvious.

The first review is also good, and its two blocking items are genuine. This review differs from it on one point of remedy (§3.2) and adds four findings, of which **F1 (the shared control region) and F3 (guaranteed detection failure on a composed page) are the most consequential**: F1 because it converts an argument into a measurement for almost no cost, and F3 because it is a certain failure currently recorded only as a documentation request.

The first review's closing principle is the right one and this review endorses it without qualification:

> Print #1 is an instrumented experiment. Preserve enough controls and metadata that any failure can be attributed to channel mapping, exposure, masking, registration, saturation, or chemistry rather than merely observed.

The additions above are all in service of it. With B1, B2 and P1–P4 settled, the plan is ready for coding.

---

## Appendix — reproducing the verification

```bash
git fetch --all --prune
git log --oneline origin/main..origin/tricolour
git diff --stat origin/main...origin/tricolour
```

Source inspected (all at `origin/tricolour`, identical to `main`):

```bash
git show origin/tricolour:cyanoneg/analyze.py   # detect_fiducials :216, sample_cells :361
git show origin/tricolour:cyanoneg/targets.py   # _grid_shape :393, step_wedge :411, calibration_page :501
git show origin/tricolour:cyanoneg/proof.py     # measured_response :69, soft_proof fallback :200
```

Note for anyone working in this folder: `git add -A` is unsafe here (per `HANDOFF.md` — it once swept 200 MB of photographs into a commit). The working tree currently holds a number of untracked `.tif` files. Stage named files only.
