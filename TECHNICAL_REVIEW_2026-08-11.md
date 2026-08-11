# Cyanoneg Technical Review

**Review date:** 11 August 2026  
**Target runtime:** Windows  
**Primary development environment:** macOS  
**Primary production path:** Python processing pipeline and `.cube` LUT

## Executive summary

The core Cyanoneg processing and calibration architecture is internally consistent. No evidence was found of a polarity, inversion-order, blocker-order, colour-space, or measured-response error that would invalidate the existing CassArt calibration.

The core non-GUI suite completed with **196 passed, 4 skipped**. Six additional GUI/path failures observed on a non-Windows host are attributable to Windows-style paths being interpreted with POSIX `pathlib.Path` semantics. Because Windows is the only supported runtime, these are a test-portability issue rather than a demonstrated production defect.

The existing CassArt `.cube` / Python negative-generation path should remain unchanged. The next commit should be a software-hardening commit, not a recalibration commit.

## Findings

### 1. Core processing pipeline — pass

The processing order matches the documented physical model:

`positive -> tonal LUT -> resize -> invert -> blocker -> mirror -> output`

No evidence was found of reversed curve polarity, double inversion, LUT application after inversion, or blocker application at the wrong stage.

### 2. Colour handling and resize — pass

Working-space conversion is explicit and resizing is performed in linear light. These are appropriate choices for calibration-sensitive image processing.

### 3. CassArt calibration — no change recommended

The measured CassArt profile is non-provisional and uses blocker **RGB (255, 64, 0)** with **saturation 1.0**. Nothing found in the software review indicates that the physical calibration needs to be repeated.

The `.cube` LUT remains the preferred external production format.

### 4. Soft-proof coverage recovery — latent generalisation issue

The proofing path derives effective ink approximately as:

`ink = 1 - min(R, G, B)`

This is valid for the current CassArt profile because saturation is 1.0 and a blocker channel reaches zero at full coverage. For a valid future profile with saturation below 1.0, the minimum channel would not reach zero and the calculation would underestimate true coverage.

**Suggested fix:** recover scalar coverage from the blocker model itself rather than from `min(R,G,B)`. Add regression tests at saturation 1.0 and below 1.0, including zero, intermediate, and full coverage. Verify that current CassArt proof output is unchanged or numerically equivalent.

> **Resolved 11 August 2026** in `1e202c2` — see Resolution below.

### 5. Strict profile JSON validation

`Profile.from_dict()` currently boolean-coerces `provisional` before validation. A malformed value such as:

`"provisional": "false"`

is therefore converted to Python `True` instead of being rejected.

**Suggested fix:** pass the raw value to validation and require a genuine JSON boolean. Add rejection tests for strings, integers and null.

> **Resolved 11 August 2026** in `1e202c2` — see Resolution below.

### 6. Photoshop `.acv` compatibility

The implementation currently treats **19 points** as a supported Photoshop ACV limit, while the handoff records that a 19-point file was rejected by Photoshop on 10 August 2026. Five- and three-point files loaded successfully; the true application limit remains unresolved.

**Suggested fix:** do not describe 19 points as Photoshop-verified. Keep `.acv` experimental / inspection-only until the actual acceptance boundary is measured. Keep `.cube` as the recommended production format.

> **Resolved 11 August 2026** in `2fbaf9a` — the boundary was measured, not assumed: 16. See Resolution below.

### 7. Windows path tests on macOS

Six GUI path tests fail on non-Windows hosts when Windows-style path strings are passed through the host `pathlib.Path`. This is expected POSIX behaviour and is not evidence that the Windows runtime is broken.

**Suggested fix:** use `PureWindowsPath` for tests of pure Windows path manipulation. Mark tests that genuinely require Windows filesystem/API behaviour as Windows-only. Do not complicate production code merely to make Windows runtime semantics execute natively on macOS.

> **Resolved 11 August 2026** in `1e202c2`, by the second of the two suggested routes only. See Resolution below.

## Development and release platform policy

Cyanoneg deliberately uses a split development and deployment model:

- **macOS is the primary development environment.**
- **Windows is the only supported production runtime.**
- **The final Windows executable is built on Windows, not cross-built on macOS.**

Development on macOS may include source editing, Git work, platform-independent tests, calibration mathematics, LUT generation, profile validation, image-processing tests, soft-proof calculations, static analysis and documentation.

Pure Windows path logic may be tested on macOS using `PureWindowsPath`. Tests that depend on Windows filesystem behaviour, Windows APIs, GUI integration or other OS facilities should be explicitly Windows-only.

### Intended workflow

`Develop on Mac -> run portable tests on Mac -> commit/push source -> move to Windows -> run full Windows test suite -> build/package on Windows -> Windows smoke test -> release`

A successful macOS test run validates the portable processing/calibration logic; it does **not** constitute final release validation.

Every releasable executable should be built from an identifiable Git revision and verified on Windows. The final Windows smoke test should cover at least GUI launch, profile loading, single-image processing, batch processing, Windows path handling, representative image I/O, `.cube` handling where applicable, and operation of the packaged executable outside the development environment.

Where practical, process a fixed reference image both from tested Python source and from the packaged executable and compare outputs. This guards against packaging/runtime differences.

## Suggested next commit

Suggested title:

**`Harden profile/proof validation and clean Windows-specific tests`**

Keep the commit deliberately narrow:

1. Correct the ACV compatibility assumptions; do not claim 19 points is Photoshop-supported.
2. Make `provisional` validation strict rather than coercive.
3. Correct soft-proof coverage recovery for blocker saturation below 1.0 and add regressions.
4. Refactor Windows path tests to use explicit Windows semantics or Windows-only skips.
5. Document the macOS-development / Windows-runtime-and-build policy.

Do **not** alter the measured CassArt response data, tonal polarity, pipeline order, inversion, blocker RGB `(255,64,0)`, CassArt saturation `1.0`, `.cube` mathematics, linear-light resizing or mirror behaviour unless a regression test demonstrates a fault.

## Acceptance criteria

The next commit is complete when:

1. Portable tests run cleanly during macOS development.
2. Windows-specific tests are clearly identifiable rather than appearing as unexplained Mac failures.
3. Invalid `provisional` JSON values are rejected.
4. Soft-proof coverage tests pass for saturation below 1.0.
5. Existing CassArt saturation-1.0 behaviour remains unchanged.
6. No code or documentation describes 19 ACV points as verified Photoshop-compatible.
7. `.cube` generation remains unchanged.
8. Core processing order remains unchanged.
9. Current measured CassArt profile data remains unchanged.
10. No requirement is introduced to build the Windows executable from macOS.
11. The final executable is built and tested on Windows from an identifiable tested Git revision.

## Overall assessment

**Core negative-generation pipeline:** strong  
**Calibration analysis:** strong  
**`.cube` production path:** strong / recommended  
**CassArt measured profile:** no software evidence requiring recalibration  
**Soft proofing:** strong for the current profile; one generalisation issue to harden  
**Profile parsing:** good, with one strict-validation weakness  
**`.acv`:** experimental until Photoshop point compatibility is measured  
**Windows runtime:** no production defect identified from Mac path-test failures  
**macOS development:** suitable and intentional

Proceed with development. The next change should harden software and tests while leaving the validated physical calibration path alone.

---

# Resolution / Re-review — 11 August 2026

Added after the fact. **The findings above are left exactly as written**, including the parts
time has overtaken: what a review got right and what it merely inferred is worth being able
to look up later, and a document quietly edited to agree with the outcome cannot be audited.
This section records what was done, not what should have been said.

All seven findings are closed, across two commits:

| # | Finding | Commit | Outcome |
|---|---|---|---|
| 1–3 | Pipeline, colour/resize, CassArt calibration | — | Pass on review; nothing changed, as recommended |
| 4 | Soft-proof coverage recovery | `1e202c2` | Fixed |
| 5 | Strict `provisional` validation | `1e202c2` | Fixed |
| 6 | `.acv` point compatibility | `2fbaf9a` | **Measured** — the limit is 16 |
| 7 | Windows path tests on macOS | `1e202c2` | Marked Windows-only |

## What was done

**4 — coverage recovery.** `blocker.recover_coverage()` now inverts the profile's own blocker
table rather than using `1 - min(R, G, B)`, for `fixed_hue` and `zone_hue` alike, and
`soft_proof` calls it. The review's diagnosis was correct and its reasoning was reproduced
before the fix was written: at saturation 0.4 the old shortcut understates a true coverage of
0.6 by more than 0.1, and a regression test pins exactly that case alongside a
saturation × coverage grid. CassArt output is unchanged, as required — at saturation 1.0 with
a channel reaching zero, the two methods agree analytically and were confirmed to agree
numerically.

**5 — strict validation.** `from_dict` passes `provisional` through untouched so `validate()`
can reject anything that is not a real JSON boolean. One line of production code; the rest is
tests, including the round trip that `"provisional": "false"` must not survive.

**6 — `.acv` compatibility.** This one was not resolved by reasoning, because reasoning is
what produced the wrong answer twice. The spec says 19; Photoshop refused 19. The Curves
dialog caps at 16, but that was also the argument for 19. So files were generated at every
count and loaded one at a time into Photoshop 2026: **16 opens with all points present, 17
and 18 are refused.** `ACV_POINTS` and `ACV_MAX_POINTS` are now both 16, pinned by a test, and
the shipped CassArt `.acv` was regenerated — which incidentally took its worst error from 44
code values to 8, since it had been held at a conservative 5 points while the limit was
unknown. `.acv` remains an inspection format and `.cube` remains the production path, now
permanently rather than provisionally: 16 points cannot follow a correction that is nearly
vertical at the foot, and no larger count exists.

**7 — Windows path tests.** The review offered two routes; only the second was taken. Rewriting
the assertions against `PureWindowsPath` was rejected deliberately, because such a test can
pass on macOS while saying nothing about the ambient `Path` the code actually meets on its one
real runtime — it would convert a visible gap into an invisible one. The seven affected tests
are `skipif`-marked instead, with the reasoning recorded at the marker, so they skip here and
genuinely execute on Windows.

## Test result

**272 passed, 10 skipped, 0 failed.**

Environment matters for this figure and is stated rather than assumed: macOS 26, Python 3.13,
full GUI suite executing against a real hidden Tk root. The 10 skips are the 7 Windows-only
path tests from finding 7 plus 3 that need `EDN_RGB_256.tif`, which is untracked by design.
282 tests collected.

Two of those tests are new since the review: the `.acv` point-count pin, and the comparison
against Photoshop's own shipped presets — which previously looked only in `C:/Program Files`
and therefore never ran during development. It now searches `/Applications` too, so the check
that established the five-curve structure runs on the machine the code is written on rather
than only on the one it ships to.

**A caution on quoting this number.** The split moves a long way with the host while the total
barely moves, so a bare pass count from this suite says less than it appears to. A machine
with no display skips all 37 Tk-dependent tests, and one without Photoshop skips the preset
comparison too; the same 282 tests can then report well over 40 skips and still show zero
failures. That is not a worse result, but it is a much weaker one — zero failures across a run
that excludes the entire GUI says nothing about the GUI. Any figure quoted from this suite,
including the one above, should carry the environment that produced it.

## Acceptance criteria

Criteria 1–10 are met. Criterion 11 — the final executable built and tested on Windows from an
identifiable revision — is by its nature outstanding on macOS; the identifiable revision for it
is `2fbaf9a`.
