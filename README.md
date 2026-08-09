# cyanoneg

Reproducible digital negatives for cyanotype printing.

A cyanotype negative is printed onto film, laid on coated paper, and exposed to UV. Getting
a good one usually means pushing curves around in Photoshop until a print looks right, then
never being quite able to do it again. This tool replaces that with a measurement: print a
calibration chart, scan it, and the correction curve is derived from what the paper
actually did.

It was written for one printer and one process — an Epson ET-1810 on hand-coated paper —
but nothing in it is specific to those beyond the recorded profile.

## The constraint that shapes everything

The ET-1810 is a four-ink dye EcoTank. No ink-density control, no Advanced B&W mode, no
QuadToneRIP support. Every lever the usual approaches rely on lives in the driver, and this
printer has none of them.

So the negatives are **colour**, not greyscale. A hue is chosen for how well its ink blocks
UV, and its saturation sets the density range. That decision propagates through the whole
program: it is why there is a blocker model, why the calibration prints a hue × saturation
grid before anything else, and why a finished negative looks like an orange transparency
rather than a black-and-white one.

## What it does

1. **Generates calibration targets** — an exposure strip, a hue × saturation blocker grid,
   and a step wedge — printed through an identity profile so the measurements capture the
   process rather than the process plus somebody else's curve.
2. **Reads the scanned prints back** — locating the chart by its fiducials, sampling each
   patch, and rejecting outliers rather than averaging them in.
3. **Derives the correction curve** by inverting the measured response.
4. **Applies it** to a photograph, in a fixed order, and writes a negative ready to print.
5. **Soft-proofs** — predicts the print from the measurements, so you can look before
   committing film. It refuses to proof a profile that has no measurements, because a proof
   invented from nothing would look authoritative and be fiction.

## Running it

Requires Python 3.13+, numpy, Pillow, tifffile and imagecodecs.

```bash
python -m cyanoneg.gui.app
```

The command line covers the same ground:

```bash
python -m cyanoneg.targets --all --out targets/
```

```bash
python -m cyanoneg.analyze wedge scan.tif targets/step_wedge.json --export curve
```

```bash
python -m cyanoneg.pipeline photo.tif --profile linear --width 240 --height 180
```

## Where to start reading

| File | What it holds |
|---|---|
| [`PLAN.md`](PLAN.md) | The design document. Why each decision was made, including the ones that were reversed |
| [`cyanoneg/pipeline.py`](cyanoneg/pipeline.py) | The spine: seven ordered steps from photograph to negative |
| [`cyanoneg/lut.py`](cyanoneg/lut.py) | The calibration maths — `derive_correction` inverts a measured response |
| [`cyanoneg/imageio.py`](cyanoneg/imageio.py) | Colour spaces and ICC handling |
| [`cyanoneg/blocker.py`](cyanoneg/blocker.py) | The colour-blocking model |
| [`cyanoneg/analyze.py`](cyanoneg/analyze.py) | Finding and measuring a chart in a scan |

Two rules are load-bearing and worth knowing before reading anything else.

**The colour space is always explicit.** Nothing guesses. A file with no profile and no
declared space raises rather than assuming, because applying a curve in the wrong space is
a tonal error that looks entirely plausible on screen and cannot be recovered from later. A
profile the file itself declares is read and converted — that is a declaration, not a
guess.

**The pipeline order is fixed and asserted by a test.** Curve before inversion, mirror last.
Both are silent failures: get them wrong and you still get a negative, just not the right
one.

## Tests

```bash
python -m pytest
```

Each test's docstring says what would go wrong if it did not exist. Several exist because
the thing they describe actually happened — a spike detector that flagged 31 of 32 levels
on a good print, a fiducial finder that could not tell a corner mark from a dark patch, a
scroll container that hid the buttons it was written to reveal.

There is a synthetic round trip in `tests/simulate.py` that runs the whole calibration —
generate a wedge, apply a known non-linear response, measure it, derive the correction,
check it inverts — with no film, paper or chemistry consumed.

## Profiles

A profile is JSON, and readable. It records the paper, film and batch, the exact printer
driver settings, the exposure, the measured blocker, the curve, and the raw patch readings
the curve was computed from — so it can be recomputed later without reprinting anything.

`profiles/linear.json` is the identity baseline that all calibration targets print through.
The other is a real, measured calibration.

## Licence

None yet. Ask before reusing.
