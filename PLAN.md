# Cyanotype Digital Negative Tool

## Current status (2026-07-25)

**All three phases complete (2026-07-25).** Pipeline, target generators, `analyze.py`, the
zone-varying blocker, soft-proofing, batch processing, and the tkinter GUI with four tabs
(Process, Profiles, Calibrate, Batch). 125 tests pass, including the two that matter most: the
synthetic calibration round-trip, and a simulated flatbed scan of a wedge print (scale change,
rotation, perspective, blur, noise, every rotation/mirror orientation) proving the measurement
chain recovers the correction to under 2% interior error.

This revision also folds in direct measurements of the supplied chart and resolves two
contradictions the earlier draft carried — see the *Resolved (2026-07-25)* notes on colour space
and on starting profiles — and records the polarity fix below.

Run the GUI with `.venv\Scripts\python.exe -m cyanoneg`; generate targets with
`.venv\Scripts\python.exe -m cyanoneg.targets --all` (add `--zone-grid` when upgrading to the zone
blocker); analyse scans headless with
`.venv\Scripts\python.exe -m cyanoneg.analyze wedge|grid|zone-grid <scan> <sidecar>`; batch-process
with `.venv\Scripts\python.exe -m cyanoneg.pipeline <folder> --profile linear --width W --height H`;
run tests with `.venv\Scripts\python.exe -m pytest`.

The tool is ready for the first physical calibration. Before that print: the driver check below,
the Photoshop interop check (`.acv` in Curves, `.cube` via Color Lookup), and recording Film 1's
real product name and batch.

### Fixed (2026-07-25): inverted tonal polarity

The pipeline produced tonally inverted prints, and every test passed because the whole system was
self-consistently wrong. `apply_blocker` treated its input as ink density, but `step_invert` hands
it the negative's *pixel value* — the opposite quantity — so a bright area in the positive laid
down no ink at all and the paper went dark where it should have stayed white. Building the
soft proof, the first thing in the project that models what the paper actually does, exposed it.

The convention is now stated once in `blocker.py` and used everywhere: **`v` is the negative's
greyscale pixel value; `v = 0` is black — maximum ink, UV blocked, paper stays white; `v = 1` is
white — clear film, UV passes, paper goes dark.** Since `v = 1 - positive`, a bright positive
gives heavy ink, which is what a real digital negative looks like.

`tests/test_blocker.py` now asserts this against the physics rather than the implementation, so a
self-consistently inverted system cannot satisfy it.

### Phase 3 as built

- **Zone-varying blocker** — `zone_hue` profiles hold density → RGB control points, interpolated
  per channel, exported as a 3D `.cube`. `targets.py --zone-grid` sweeps hue at five densities and
  `analyze_zone_grid` reports the best hue per zone **and whether the hue actually varies** — the
  plan only licenses this upgrade if measurements justify it, so a run that finds one winning hue
  everywhere says to stay with fixed-hue. Two-point zone data reproduces the fixed-hue model to
  float precision, so existing profiles are unaffected.
- **Soft proof** (`proof.py`) — predicts the print from the profile's own measured patches and
  **refuses to render without them**; an invented proof would look authoritative while being
  fiction. Selectable in the Process tab beside the film preview.
- **Batch** — `batch_negatives` over a folder, collecting per-file failures instead of aborting,
  with a CLI and a GUI tab. It refuses to overwrite its own sources.
- **GUI styling** is centralised in `cyanoneg/gui/theme.py`. Note that the native Windows ttk
  theme ignores most colour settings; switching `THEME` to `clam` is the entry point for a real
  visual redesign.

Address the user as **Steven** — never "Steve" (that is only the Windows username).

## Context

This project supports a return to cyanotype printing, producing consistent digital negatives from
scanned 35mm colour negatives. Today the process is manual guesswork in Photoshop:
convert to mono, eyeball a curve, invert, mirror, print, and hope. Every new paper resets that
guesswork, because paper absorbency and sizing materially change contrast and dMax.

The goal is a tool that makes the negative **reproducible** — encode each paper/chemistry
combination once as a measured profile, then apply it deterministically to any image.

### The constraint that shapes everything

Research (notebook `e379917e` + web verification) established the key hardware facts:

- **Epson ET-1810** is a 4-colour **dye** printer (ink series 104). No photo black, no Advanced
  Black & White mode, no Color Density slider, and unsupported by QuadToneRIP (no EcoTank is).
  The driver offers **zero** ink-density control.
- **Film**: inkjet-coated OHP film that has worked before is already to hand. Budget film can reach
  a lower Dmax and its coating may vary between batches; cyanotype's short scale is forgiving of
  both. Film (and batch) is recorded as a profile variable, never hardcoded — switching film means
  recalibrating.

Because the driver gives us nothing, **all** contrast and UV-density control must happen in the
image data. This makes **colour blocking mandatory**: the negative is printed as a coloured image
(amber/red/yellow-green), never greyscale, since greyscale mode on a 4-ink dye machine discards the
three best UV blockers. Dye printers have precedent favouring red; Harmon's yellow-green ratio was
calibrated on pigment. **The tool must measure the best hue, not assume one.**

Cyanotype is the one alt-process where dye ink is genuinely viable — it needs only ~1.2–1.4 density
range. Pt/Pd would not be achievable on this hardware.

### Decisions taken

| Decision | Choice |
|---|---|
| Processing | Python does all pixel work; Photoshop optional, for review/print only |
| Interface | Simple tkinter GUI |
| Calibration input | Flatbed scan of test print; tool locates and reads patches |
| Scan input | Already-inverted lab scans (positives); raw-scan path kept as a flag |
| Scope | Phased — working pipeline first, then measured calibration |

---

## Environment (verified)

- Python **3.14.6**, pip 26.1.2 at `C:\Users\steve\AppData\Local\Python\pythoncore-3.14-64`
- `pywin32` 312 present; **tkinter 8.6 present** (no GUI dependency needed)
- git 2.53.0 present
- Photoshop **2025 and 2026** both installed
- **Done:** venv at `Cyanotype\.venv\` with `numpy` 2.5.1, `pillow` 12.3.0, `tifffile` 2026.7.14,
  `imagecodecs` 2026.6.26 (the last is required — tifffile cannot decode LZW without it)

### Supplied asset: `EDN_RGB_256.tif`

Peter Mrhar's official EDN 256-step calibration chart is already in the project folder. Verified:

- 1507 × 1507 px at **300 ppi** → prints 127.6 mm (5.02 in) square, fits A4 comfortably
- **16-bit RGB**, LZW compressed, the standard 3144-byte **sRGB IEC61966-2.1** profile embedded
  (note: *not* Gray Gamma 2.2)
- Fully neutral (R=G=B). **285 distinct values**, not 256: 257 sit on the 8-bit grid, the other 28
  come from anti-aliased label text and account for only 0.52% of pixels (11,785 of 2,271,049). All
  256 patch levels are present and cover the complete 0–255 range, but **patch sampling must reject
  label pixels** — sample a central sub-region, take a median, and flag cells with a large spread.
- Patch cells measure **79 × 79 px** (~6.7 mm at 300 ppi), so ≥11×11 averaging has ample room
- Supplied as a **positive** — per the EDN workflow it must be inverted and mirrored before printing
- Layout is **sequential, not randomised**, so it carries no anti-spike protection and no black
  border. Usable as-is for a first calibration; `targets.py` should still generate a randomised
  bordered variant, since sequential 256-step tablets are the format most prone to spikes.

Because it is sRGB rather than Gray Gamma 2.2, the pipeline must treat the chart's colour space
explicitly and match it when reading the scan back, or the measured values won't correspond.

**Resolved (2026-07-25): the working space is sRGB, and it is never implicit.** Earlier drafts of
this document said the chart is sRGB but also that the LUT is applied in "gamma 2.2 space" and that
targets are "gamma 2.2 tagged" — those contradict, and sRGB's linear toe differs materially from a
pure 2.2 power curve in the shadows, exactly where cyanotype's long toe already makes life hard.
The space is therefore a single explicit parameter, `working_space` (`srgb` | `gamma22` | `linear`),
recorded in every profile and defaulting to **`srgb`**, so chart, generated targets, LUT and scan
all live in one declared space. Code must never infer a space from an untagged file — it raises
instead. This is the most likely source of a silent tonal error in the whole tool.

---

## Architecture

Project root: **`C:\Users\steve\Desktop\Claude\Cyanotype\`** (git-initialised).

All projects live in subfolders of `C:\Users\steve\Desktop\Claude\`.

```
cyanoneg/
  imageio.py    load/save 16-bit TIFF, working_space transfer curves, ICC tagging
  mono.py       RGB -> monochrome channel mixer + per-channel noise report
  lut.py        1D LUT build/apply/smooth/monotonic; .cube and .acv export
  blocker.py    greyscale -> colour-blocked RGB
  profiles.py   paper profile model, JSON load/save/validate
  targets.py    generate calibration targets + layout sidecar JSON
  analyze.py    read scanned targets -> measurements        [Phase 2]
  pipeline.py   orchestration
  gui/app.py    tkinter interface
profiles/       saved profiles (.json, git-tracked)
targets/        generated target files (.tif ignored, sidecar .json tracked)
tests/
```

### Processing pipeline — order is critical

Per Reeder/Anderson and Ware, the correction curve applies to the **positive**, before inversion.
Getting this order wrong silently produces wrong tones.

1. Load positive scan (16-bit TIFF preferred), work in float32
2. **Mono conversion** — channel mixer, default 30R/59G/11B, weights configurable.
   Report per-channel noise so the grainiest channel (usually blue) can be dropped.
3. **Apply paper LUT** to the positive, in the profile's declared `working_space` (default `srgb`)
   — this must match the space the step tablet was measured in, or readings won't correspond
4. **Resize** to print size at output ppi (360 default)
5. **Invert** to negative
6. **Colour block** — map greyscale to blocker hue
7. **Flip horizontal** — so ink side sits against emulsion
8. **Export** print-ready TIFF

### Colour blocking model

Phase 1 uses a fixed-hue model: `RGB = white − g · (white − blocker)`, with a saturation scalar
controlling density range (Harmon's approach — one curve, saturation sets DR). Phase 3 can upgrade
to EDN-style zone-varying hue via 3D LUT if measurements justify it.

### Paper profile format

JSON, one file per paper/chemistry combination. **Raw patch measurements are stored alongside the
computed LUT**, so profiles can be recomputed if the smoothing algorithm later improves — without
reprinting anything.

```json
{
  "name": "Paper 1 — Chemistry",
  "paper": "Paper 1",
  "chemistry": "Chemistry",
  "printer": "Epson ET-1810",
  "media_type": "Epson Glossy Photo Paper",
  "film": "Film 1",
  "working_space": "srgb",
  "provisional": true,
  "driver_settings": {
    "color_correction": "<No Color Adjustment | Color Controls, all sliders 0>",
    "media_type": "Epson Glossy Photo Paper",
    "quality": "..."
  },
  "blocker": { "model": "fixed_hue", "rgb": [64,128,0], "saturation": 1.0 },
  "exposure": { "spe_seconds": 480, "uv_source": "..." },
  "lut": { "size": 256, "values": [] },
  "measurements": { "raw_patches": [], "scan_date": "..." }
}
```

Three fields carry weight beyond documentation:

- **`working_space`** — the space the LUT was measured in and must be applied in (see above). Read
  by the pipeline, not just recorded.
- **`provisional`** — `true` means the LUT and blocker are *not* measured. The GUI marks such
  profiles clearly so an un-calibrated guess is never mistaken for a measurement.
- **`driver_settings`** — the exact Epson driver state used for the calibration print. Because the
  ET-1810 offers no ink-density control, the calibration silently absorbs whatever transform the
  driver applies; changing any of these settings afterwards invalidates the profile.

### Calibration targets (`targets.py`)

Generated in the **same colour-blocked form** as real negatives — otherwise the calibration doesn't
transfer to actual prints.

- **Exposure strip** — determines SPE (minimum UV time to max black through clear film base)
- **HSB blocker grid** — finds the optimal UV-blocking hue for this printer/ink/film
- **256-step randomised anti-spike wedge**, carrying:
  - randomised patch positions with redundant values (averaged later to cancel spikes)
  - a thick continuous black border — prevents cyanotype **edge-etch/peptization**, where
    chemistry migrates laterally through paper fibres and corrupts edge patches
  - corner fiducials for automatic detection
  - tagged with the declared `working_space` (default `srgb`), matching the supplied EDN chart

Each generator also writes a **sidecar JSON** beside the TIFF recording the RNG seed, grid geometry
and position → intended value. `analyze.py` reads the layout from the sidecar rather than
re-deriving it, so a target and its reading can never disagree about what was printed where.

Targets are generated **colour-blocked but never LUT-corrected** — they print through `linear.json`
only. A target pushed through a paper curve would measure the process plus that curve.

256 steps rather than the traditional 100: it yields ~62 mathematically useful values versus ~17.

### Analysis (`analyze.py`, Phase 2)

Detect fiducials → perspective-correct → locate patches geometrically → sample each with ≥11×11
averaging → normalise against paper-white and max-black references → average redundant patches →
swap input/output to derive the correction → enforce monotonicity → smooth with a monotone spline
(PCHIP) → save profile, export `.cube` and `.acv`.

Reports measured density range and warns if it falls outside cyanotype's 1.2–1.4 window, plus
flags residual spikes.

### Shipped starting profiles

**Resolved (2026-07-25): no published starting curve ships.** Earlier drafts said Phase 1 would ship
one. Every published cyanotype curve is bound to a specific printer, ink, film and paper, and none
is verifiable for a 4-ink dye ET-1810 on `Film 1`. Shipping a guess would bake an unmeasured error
into the very first calibration print — the one measurement everything else is derived from.

Instead, Phase 1 uses this document's own Harmon model (*one curve, saturation sets DR*) and ships:

- **`profiles/linear.json`** — identity LUT, `provisional: false`. **All calibration targets print
  through this**, so measurements capture the process itself rather than the process plus somebody
  else's curve.
- **`profiles/paper1-provisional.json`** — identity LUT, `provisional: true`, blocker hue and
  saturation left blank until the HSB grid is read.

The HSB blocker grid already sweeps hue × saturation, so the first sheet yields both the best
UV-blocking hue and a saturation that lands density range in the 1.2–1.4 window. A *shaped* tone
curve then comes from Phase 2 measurement, which is the only honest source for one.

### Photoshop interop

Python writes `.acv` (documented binary format, no Photoshop needed) and `.cube` files so curves can
be inspected or applied manually in Photoshop for QA. `.atn` action files are **not** used — they
cannot be authored programmatically and offer nothing a script doesn't.

---

## Phasing

Phasing is for incremental delivery, not because anything is blocked — Phase 2 can follow
immediately. Phase 1 gets usable negatives out of the printer before the calibration maths lands.

**Phase 1 — get a printable negative end-to-end**
- Project skeleton, deps, git init
- `imageio`, `mono`, `lut`, `blocker`, `profiles`, `pipeline`
- `targets.py` — all three target generators
- GUI: Process tab + Profiles tab
- Ships two starting profiles so negatives can be produced immediately — see below
- Full synthetic test suite (below)

**Phase 2 — measured calibration**
- `analyze.py` patch detection and reading
- GUI Calibrate tab: 3-step wizard (exposure → blocker → linearisation)
- First real measured profile

**Phase 3 — refinement** *(complete)*
- Zone-varying colour blocker (EDN-style 3D LUT)
- Soft-proof preview in GUI
- Batch processing

---

## Verification

**The key test, runnable without consuming film, paper or chemistry:** a synthetic round-trip.
Generate a step wedge, push it through a simulated non-linear "process response" curve of known
shape, feed the result into the analysis code, and assert the recovered LUT inverts that response to
within tolerance. This validates the entire calibration mathematics — the part most likely to be
subtly wrong — before a single test print is committed.

Also:
- Unit tests: identity LUT is a no-op; monotonicity enforcement; gamma round-trips; target geometry
- `.acv` files open correctly in Photoshop 2026's Curves dialog; `.cube` loads via Color Lookup
- Pipeline ordering test: asserts curve-before-invert and flip-last
- Visual: process a real scan, confirm the negative is mirrored, coloured, and correctly oriented
- End-to-end (Phase 2): print wedge on inkjet film → coat → expose → process → scan → profile →
  print an image → verify tonal linearity

---

## Driver check (one-off, needed before the first calibration print)

Settings → Bluetooth & devices → Printers & scanners → EPSON ET-1810 → Printing preferences.

- **Main** tab: set Media Type + Quality, and ensure **Color** is selected, never Grayscale.
- **More Options** tab → **Color Correction** → select **Custom** → **Advanced**.
- Look for **"No Color Adjustment"**.

If present, use it — the tool then has direct control of the blocker hue. If absent (likely on a
consumer EcoTank), use **Color Controls** with all sliders at zero and treat that as fixed forever;
the ColorBlocker calibration measures end-to-end and absorbs the driver's transform into the
profile. Either way the profile must record the exact driver settings used, since changing them
invalidates the calibration.

## Open items

- The first profile uses placeholder names — **`Film 1`**, **`Paper 1`**, **`Chemistry`**. Replace
  the film entry with the real product name and batch when convenient: if results later degrade
  unexpectedly, a film batch change is the first thing to suspect, and a placeholder records nothing.
- Confirm whether the ET-1810 driver exposes "No Color Adjustment" (see above). If it doesn't,
  driver colour management must be compensated for empirically in the curve rather than assumed
  linear — the calibration still works either way, it just has to absorb the driver's behaviour.
- Media Type is the only pseudo-ink-limit lever available — test Epson Glossy Photo Paper vs Photo
  Quality Inkjet for highest Dmax without pooling.
- Dye on PET dries slowly; negatives must be fully dry before contact printing.
