# Handoff — 10 August 2026

Where cyanoneg stands, what is left, and how to do it. Written to be readable after a gap,
or by someone else entirely.

## State

**The calibration is done and it has been verified on paper.** Two prints, two different
subjects, both track the positive as a straight line to within about 2 L\*, and the density
range came out within 1% of what the wedge predicted. That was the whole claim of the
project, and it is now tested rather than asserted.

Everything is committed and pushed. 239 tests passing.

| | |
|---|---|
| Repository | github.com/franzandclaude-oss/cyanoneg (**private**) |
| Profile | `profiles/CassArt 300 Sm.json` — measured, not provisional |
| Launch | `cyanoneg` shortcut on the Desktop, or in the Start Menu |
| Tests | `python -m pytest` from the project folder |

### The numbers that matter

| | |
|---|---|
| Paper / film / batch | CassArt 300 Sm · Film Generic · Batch One |
| Blocker | RGB (255, 64, 0) — hue 15°, saturation 1.0 |
| SPE | **13:30**, lamp at 350 mm |
| Lamp | 100 W UV-A, 390–400 nm |
| Driver | No Color Adjustment · Photo Paper Glossy · High Standard |
| Density range | 1.108 measured on the wedge; 1.117 and 0.984 on the two prints |

**The driver settings are part of the calibration.** Change any of the three and the profile
no longer describes what the printer does.

---

## Settled: no lamp warm-up

**Decided 10 August 2026.** Expose cold, straight from switch-on. `13:30` stands exactly as
measured — the exposure strip that produced it was shot cold, so nothing needs redoing.

The reasoning, in case it is ever revisited: a UV LED reaches near-full output in seconds,
so a warm-up buys consistency that is already there. Print 2 appears to have had one and
lost 0.13 of density range for it — paper white L\* 96.1 against print 1's 99.8, shadows
unchanged in both. Highlights lifted with shadows unmoved is what extra dose does once the
shadows are already at Dmax, and 0.13 is a lot on a paper with only 1.1 to give.

Recorded in the profile as `warmup_seconds: 0`.

## The two jobs left

### 1. Print a wedge strip alongside a photograph

This is the measurement that closes the last open question. The calibration sheet reached
Dmax L\* 32; both prints only reached L\* 37–39. That gap is almost certainly ordinary
coating variation between sessions, but right now it is inferred, not known. A wedge on the
same sheet as the picture, washed in the same tray, settles it.

The full wedge is 192 × 104 mm and will not fit beside a print. Use a compact one:

```bash
python -c "from cyanoneg.targets import step_wedge; step_wedge((255,64,0), saturation=1.0, levels=16, redundancy=4).save('targets')"
```

That gives a **60 × 60 mm** strip — 16 levels, 4 copies each — small enough to sit next to a
240 × 180 mm negative on one A4 sheet. Print both onto film, lay them on the same coated
sheet, expose together, wash together, scan together.

Then read it back:

```bash
python -m cyanoneg.analyze wedge "yourscan.tif" targets/step_wedge.json
```

Compare its density range and Dmax against the numbers in the table above. If they agree,
the process is stable and the remaining error is scanner-related. If they disagree, the
coating varies between sessions and that is worth knowing before blaming anything else.

**Scan it the same way as the calibration sheets** — SilverFast raw, converted in Photoshop.
A different scan path is the one thing that would make the comparison meaningless.

### 2. The .acv point limit — settled, it is 16

**The `.cube` is done and works.** Adjustments → Color Lookup → Load 3D LUT →
`profiles/CassArt 300 Sm.cube`. A 64×64×64 grid, within 0.6 of a code value of the real
curve. This is the one to use for anything real.

**The `.acv` limit is 16 points.** Measured 11 August 2026 against Photoshop 2026, one file
per count: 16 opens with every point present, 17 and 18 are refused, and 19 was already
known to be refused on 10 August. 16 is also the cap in Photoshop's own Curves dialog, so
the file format and the UI agree — the spec's 19 is the outlier, and was never worth
trusting over a loaded file.

Version and curve count were never the problem: the file is version 4 with five curves,
matching all nine presets Photoshop ships in `Presets/Curves`.

`ACV_POINTS` and `ACV_MAX_POINTS` in `cyanoneg/lut.py` are both 16 now, and
`profiles/CassArt 300 Sm.acv` has been regenerated at that count. Nothing is outstanding
here. `acv-test/` is gone; if a copy survives on the Windows machine it can go too — that
one also holds ~120 MB of test TIFFs, untracked.

**The `.acv` stays an inspection format regardless.** At 16 points it misses the deep
shadows by about 8 code values, because the correction is nearly vertical at the foot, and
there is no larger count available to close that gap. Use it to look at the curve's shape;
use the `.cube` to actually process anything.

---

## Open offers, not started

Neither is needed; both were discussed and left.

- **A greyscale "tone only" proof view.** Colour interferes with tonal judgement. A fourth
  preview mode would strip the hue and keep the lightness.
- **Rewriting the git history** to remove `EDN_RGB_256.tif`. It is untracked now but remains
  in the first 30 commits. Irrelevant while the repository is private. **If you ever make it
  public, do this first** — it is a five-minute job done deliberately and a mess done in a
  panic afterwards.

---

## Things worth knowing before changing anything

**The density range is 1.11, just under the classic 1.2–1.4 window.** The analysis warns
about this every run. It is not an error — both prints look right — but it means there is
headroom if you ever want more separation, through longer exposure or stronger chemistry.

**The soft proof is now measured, not modelled.** It interpolates the wedge patches' own
colour, so it predicts both tone and hue from your paper. Two faults were found and fixed
against the real prints: it blended tone in the wrong space (~14 L\* too light in the
midtones), and it rendered a cyanotype as nearly grey. Both were invisible to the test suite
until the tests were made to measure the proof's output rather than its inputs.

**A different paper, film batch or chemistry needs its own profile**, not an edit of this
one. Everything in the file — blocker, curve, density range, SPE — was measured on this
combination.

**`git add -A` is not safe in this folder.** It once swept 200 MB of photographs into a
commit; what caught it was GitHub's file-size limit. Image files are now ignored outright,
but stage named files rather than everything.

---

## If something looks wrong in a print

In the order worth checking:

1. **Driver settings** — the three above. Silently reset by a driver update.
2. **Which profile** — the Process tab must say `CassArt 300 Sm`, not `linear`.
3. **Exposure** — 13:30, 350 mm, exposed cold - no warm-up.
4. **The paper** — a new batch or a different coating session is the usual suspect, which is
   what job 1 measures.
5. **The scan** — if only the *measurements* look wrong and the print looks fine, suspect
   the scan path before the process.

---

## Development and release platform policy — added 11 August 2026

**Develop on macOS; run Cyanoneg and produce the final executable on Windows.** Windows is
the only supported production runtime. macOS is the primary development environment for
source changes, Git work, portable tests, calibration maths, LUT/profile work, image-processing
tests and documentation.

Do not add production compatibility code merely to make Windows-only runtime behaviour work
natively on macOS. Tests of pure Windows path manipulation can use `PureWindowsPath`; tests
that genuinely depend on Windows filesystem behaviour, APIs or GUI integration should be
explicitly Windows-only.

The release workflow is:

**Develop on Mac -> portable tests on Mac -> commit/push -> Windows -> full Windows tests ->
build/package on Windows -> smoke test -> release.**

The Windows executable is **not** to be cross-built on macOS. A clean Mac test run validates
the portable processing/calibration logic, not the release artifact. Every releasable build
should come from an identifiable Git revision and be verified on Windows, including GUI launch,
profile loading, representative single and batch processing, Windows path handling and packaged
execution outside the development environment.

See `TECHNICAL_REVIEW_2026-08-11.md` for the full review and proposed next commit.
