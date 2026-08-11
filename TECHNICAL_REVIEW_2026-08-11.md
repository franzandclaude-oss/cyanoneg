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

### 5. Strict profile JSON validation

`Profile.from_dict()` currently boolean-coerces `provisional` before validation. A malformed value such as:

`"provisional": "false"`

is therefore converted to Python `True` instead of being rejected.

**Suggested fix:** pass the raw value to validation and require a genuine JSON boolean. Add rejection tests for strings, integers and null.

### 6. Photoshop `.acv` compatibility

The implementation currently treats **19 points** as a supported Photoshop ACV limit, while the handoff records that a 19-point file was rejected by Photoshop on 10 August 2026. Five- and three-point files loaded successfully; the true application limit remains unresolved.

**Suggested fix:** do not describe 19 points as Photoshop-verified. Keep `.acv` experimental / inspection-only until the actual acceptance boundary is measured. Keep `.cube` as the recommended production format.

### 7. Windows path tests on macOS

Six GUI path tests fail on non-Windows hosts when Windows-style path strings are passed through the host `pathlib.Path`. This is expected POSIX behaviour and is not evidence that the Windows runtime is broken.

**Suggested fix:** use `PureWindowsPath` for tests of pure Windows path manipulation. Mark tests that genuinely require Windows filesystem/API behaviour as Windows-only. Do not complicate production code merely to make Windows runtime semantics execute natively on macOS.

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
