# Tricolour cyanotype support for cyanoneg

**Status:** ready for technical peer review. Not started — no code written.
**Reviewers:** please read §"Revisions since the approved draft" and §"Open questions" first;
those are where the plan changed and where it is still uncertain.

---

## 1. Context

`cyanoneg` turns a positive into **one** monochrome UV-blocked negative, calibrated against a
measured paper profile. That calibration is finished and verified on paper (`HANDOFF.md`):
CassArt 300 Sm, blocker RGB (255,64,0) at saturation 1.0, SPE **13:30 = 810 s**
(`profiles/CassArt 300 Sm.json`, `exposure.spe_seconds`), 100 W 390–400 nm UV LED at 350 mm,
Epson ET-1810 with the driver locked to No Color Adjustment · Photo Paper Glossy · High Standard,
exposed cold (`warmup_seconds: 0`).

The goal is **tricolour cyanotype**: three sequential cyanotype layers on one sheet, each from its
own negative, each chemically transformed to a different colour, building a subtractive CMY image.
The app must produce three registered, per-layer-calibrated negatives from one RGB positive.

The process spec below comes from the user's research library (Golaz / Bary / Bind; NotebookLM
"Tricolour Cyanotype: Method, Negatives and Toning"). It is a domain requirement, not a design
choice — getting the channel mapping or the print order wrong wastes a multi-day darkroom cycle.

### 1.1 The process

| Layer | Source channel | Negative | Becomes | Exposure | Sensitizer |
|---|---|---|---|---|---|
| 1 | **Green** | Magenta | bleached, then madder-root toned red-magenta | 1.2–2× SPE | 10/10 |
| 2 | **Blue** | Yellow | heavily overexposed, then bleached to yellow Fe(OH)₃ | **2.5–3× SPE** | 10/10 |
| 3 | **Red** | Cyan | classic untoned Prussian blue | 1–1.2× SPE | **5/5** (1:1 dilute) |

**Physical print order is Magenta → Yellow → Cyan, and it is not negotiable.** The alkaline
(sodium carbonate, pH 10–12) bleach used by the magenta and yellow layers destroys Prussian blue,
so cyan must be last. The yellow layer also sits as a tannin barrier between the madder-toned
magenta layer and the cyan sensitizer.

Digital flow the sources prescribe: RGB positive → **boost saturation 30–50%** → split channels →
per-channel curve → invert → flip. The saturation boost exists because bleaching and botanical
toning yield muted, matte colour.

---

## 2. Revisions since the approved draft

Six changes, four of them substantive. Each replaces something in the version that was approved,
so reviewers who read the earlier draft should read this section as a diff.

### R1 — Provisional curves are the measured CassArt LUT, cloned verbatim

**Was:** *"a generated provisional shaping curve (magenta: contrast boost; yellow: highlight
protection; cyan: identity)."*

**Now:** all three seeded profiles carry the **measured CassArt 300 Sm LUT unchanged**.

Nothing has measured how bleaching or madder toning reshapes the scale on this paper. A
"contrast boost" for magenta and a "highlight protection" for yellow are numbers with no
measurement behind them, and they would be indistinguishable in the output from numbers that
*are* measured. `proof.py`'s own docstring rules on exactly this case: a curve invented from a
plausible-looking shape "would be worse than none — it would look authoritative while being
fiction."

The measured LUT is not a guess: it is the true response of this ink, film, printer and paper
through the exposure and wash, which the first two of the three layers still share up to the
point of bleaching. It is the best-supported starting point available, and it has the property
that print #1's departures from linear are attributable to the chemistry rather than to a curve
someone invented.

**What differentiates the three layers on print #1 is the exposure multiplier, not the curve.**
That is honest: exposure is the one per-layer quantity the sources actually specify with numbers.

### R2 — Wedges are isolated per layer, not stacked

**Was:** *"patch (r,c) of yellow lands on patch (r,c) of magenta and you measure the stack —
which is what the correction should linearise."*

**Now:** each layer gets its **own** wedge slot on the page. In a given slot, the owning layer
prints its wedge; the other two layers print **full blocker** over that slot, so they deposit
nothing there.

The stacked version measures the wrong thing. A stacked patch read through the green channel
carries magenta's green absorption *plus* yellow's *plus* cyan's. `derive_correction` would then
return a curve that linearises the neutral stack, and feeding that curve back into the magenta
layer alone is not a defined operation. Three curves derived that way are not three per-layer
curves; they are the same neutral-stack curve read three ways.

Isolation costs one thing and buys another:

- **Cost:** the wedge no longer measures a patch that has other colorants sitting on it.
- **Buy:** the wedge *does* measure the layer as it survives the **full remaining process** —
  the isolated magenta patches still go through the yellow layer's carbonate bleach and the cyan
  layer's sensitizer and wash, because they are on the same sheet. That is the interaction that
  actually threatens the layer, and it is captured.

The stack is a legitimate thing to want to measure, but it is a *second* experiment (a small
neutral ramp printed by all three layers together), and it belongs after print #2 when the
per-layer curves exist to interpret it against.

### R3 — Page background is full blocker, not white

`calibration_page()` ([targets.py:501](cyanoneg/targets.py:501)) builds its page canvas as
`np.ones(...)` — white, i.e. **clear film** ([targets.py:547](cyanoneg/targets.py:547)). That is
right for its own use: the calibration page's margins are never coated and never measured.

On a tricolour page it is wrong three times over. Clear film passes UV, so every uncoated margin
region receives full exposure — on three successive layers. Anything that *is* coated outside the
picture goes to Dmax, then gets bleached and toned, then gets Prussian blue on top. The
`tricolour_page` compose step therefore initialises the canvas to **full blocker**, matching the
value `step_wedge` already computes for itself at [targets.py:472](cyanoneg/targets.py:472):

```python
full_blocker = apply_blocker(np.zeros((1, 1)), blocker_rgb, saturation)[0, 0]
```

This is also what makes R2's masking free: "the other two layers print full blocker over this
slot" is the same as "leave the background alone."

### R4 — `proof.py` must not silently fall back to L\*

**Was:** *"record `measurements["response_quantity"]` and read `p.get(quantity, p["lstar"])`."*

**Now:** read `quantity = measurements.get("response_quantity", "lstar")` and then index
`p[quantity]` directly, letting `KeyError` become a `ProofUnavailable` with a message that names
the profile and the missing quantity.

`p.get(quantity, p["lstar"])` means a yellow profile whose patches were somehow written without
`lstar_b` silently soft-proofs against L\*, which on a yellow layer is nearly flat — the proof
would render an almost blank image and look like a legitimate prediction. The default on
`measurements` (not on the patch) preserves every existing profile: they have no
`response_quantity`, so they resolve to `"lstar"` and behave exactly as now.

There are **three** call sites, not one: `measured_response`
([proof.py:69](cyanoneg/proof.py:69)), `measured_colour` ([proof.py:118](cyanoneg/proof.py:118))
and `measured_endpoints` ([proof.py:147](cyanoneg/proof.py:147)). All three must move together
or the proof mixes axes.

Separately, `soft_proof`'s no-colour fallback branch
([proof.py:200–205](cyanoneg/proof.py:200)) maps tone onto the `PAPER_RGB`/`BLUE_RGB` Prussian
blue constants. That is meaningless for a magenta or yellow layer. Any non-cyan layer reaching
that branch must **raise**, not render.

### R5 — `registration.ticks` dropped

The approved set JSON had `"ticks": true` and `step_frame` was described as adding them, but
nothing ever specified what a tick was, where it went, or what read it. Four corner fiducials
already over-determine the homography that `detect_fiducials` feeds. Removed rather than left as
a field nobody implements.

### R6 — Glyph placement and density are constrained, not free

The C/M/Y letter must satisfy two constraints that were not stated:

1. **~50% blocker coverage, not clear film.** `detect_fiducials`
   ([analyze.py:230–234](cyanoneg/analyze.py:230)) thresholds on the dark tail of the scan. A
   clear-film glyph prints as dark as a fiducial and becomes a fifth candidate blob.
2. **Picture border only, never a wedge's border.** `detect_fiducials` takes the four *extreme*
   corners of all candidate blobs ([analyze.py:263–275](cyanoneg/analyze.py:263)) — a stray dark
   mark outside the true corner span silently redefines the frame. The code comment at
   [analyze.py:247–252](cyanoneg/analyze.py:247) records this failure happening once already,
   with dark wedge patches.

---

## 3. Decisions carried forward from the approved draft

- **Provisional first, measure from print #1.** Per-layer curves are genuinely required
  (bleaching and toning compress the scale differently per layer) but cannot be measured before
  a first print exists. See R1 for what "provisional" now means.
- **Three profiles + one set file.** Each layer is an ordinary `Profile` JSON; `Profile` is
  **not** modified. `step_apply_lut` ([pipeline.py:78](cyanoneg/pipeline.py:78)) and
  `step_blocker` ([pipeline.py:118](cyanoneg/pipeline.py:118)) already read
  `.lut` / `.working_space` / `.blocker` off a `Profile`, and the Calibrate wizard already
  operates on exactly one profile at a time — precisely a layer's granularity.
- **Exactly three layers**, keyed `cyan` / `magenta` / `yellow`. No general N-layer set.
- **Keep the measured blocker (255,64,0) sat 1.0 on all three layers.** UV blocking is a property
  of ink on film, not of paper chemistry or toning, and it has been measured on this printer. The
  book's green (R0,G90,B30) and amber (C0,M50,Y50,K0) suggestions are not evidence; adopting them
  blind would discard a real measurement and cost two darkroom sessions.
- **Registration furniture: fiducials + border + C/M/Y letter.** The first two come from one
  `apply_frame()` call regardless; the letter is a few lines that stop three near-identical orange
  transparencies being mixed up in a dim room.

---

## 4. Approach

### 4.1 Reuse, not new code

| Existing | Where | Used for |
|---|---|---|
| `apply_frame()` | [targets.py:54](cyanoneg/targets.py:54) | border + 4 corner fiducials (top-left hollow → rotation *and* mirror unambiguous); returns fiducial geometry |
| `_finish()` convention | [targets.py:116](cyanoneg/targets.py:116) | establishes frame-then-flip; the new chain matches it |
| `step_wedge()` | [targets.py:411](cyanoneg/targets.py:411) | the per-layer wedge, unchanged |
| `calibration_page()` | [targets.py:501](cyanoneg/targets.py:501) | **the template for `tricolour_page`** — see 4.5 |
| `PrintSize.oriented_for()` | [pipeline.py:53](cyanoneg/pipeline.py:53) | resolved **once**, then passed to all three layers |
| `step_apply_lut` / `step_resize` / `step_invert` / `step_blocker` / `step_flip` / `save_tiff` | `pipeline.py` | unchanged |
| `Lut`, `derive_correction` | [lut.py:109](cyanoneg/lut.py:109), [lut.py:237](cyanoneg/lut.py:237) | unchanged; `derive_correction` is quantity-agnostic |
| `detect_fiducials`, `sample_cells` | [analyze.py:216](cyanoneg/analyze.py:216), [analyze.py:361](cyanoneg/analyze.py:361) | scan-back; `sample_cells` gains a live `quantity` |

`apply_frame` blocks UV at the sheet edge (clean white paper margin, absorbs edge-etch) and its
fiducials print **dark**. Layer 1 prints those marks onto the paper; layers 2 and 3 align to the
printed marks. Identical absolute positions on all three means a fringed mark is visible
misregistration — the check is a light table, not software.

`targets.py` does not import `pipeline`, so `tricolour.py` importing both is acyclic. No lift, no
variant module.

### 4.2 Channel extraction is a slice, not a mix

Do **not** route through `to_mono()` with unit weights. Two independent reasons:

- it normalises weights ([mono.py:40](cyanoneg/mono.py:40)) and defaults to `mix_in="linear"`,
  round-tripping through `to_linear`/`from_linear` in float32 for no gain;
- it **early-returns on mono input** ([mono.py:35](cyanoneg/mono.py:35)), which would silently
  emit three identical negatives from a greyscale source.

`extract_channel()` is a direct `data[..., i]` slice that **raises** on `image.is_mono`. Byte-exact
against the slice, by test.

### 4.3 Mandated step order (tricolour only)

```
load → [raw invert] → saturate → extract_channel → LUT → resize → invert → blocker
     → frame (+ glyph) → [compose page: picture + 3 wedge slots] → flip → export
```

The existing six-step mono order is **unchanged**;
[tests/test_pipeline.py:31–52](tests/test_pipeline.py:31) must keep asserting it verbatim.

Ordering constraints, each of which is a real dependency rather than a preference:

- `saturate` **before** `extract_channel` — saturation is a three-channel operation.
- `frame` **after** `blocker` — `apply_frame` writes into an `(h, w, 3)` canvas, so its input must
  already be RGB.
- `frame` and `compose` **before** `flip` — matching `_finish()`, so emitted fiducial and slot
  geometry is in **print orientation**, which is the orientation a scan is measured in.
- the glyph is drawn on a small sub-canvas, **that sub-canvas is mirrored**, then pasted into the
  border — so it reads correctly on the film, which is the side you actually read.

### 4.4 Saturation boost and the LUT are independent by construction

Worth stating because it looks like a conflict and is not. The saturation boost changes pixel
values before the LUT; if the LUT had been measured through the boost, applying both would
double-count.

It was not. Targets are printed through `linear.json` by design — the wedge is never
LUT-corrected and never saturated. So the measured curve describes *the process*, and the boost
is a creative transform on the image only.

This imposes one implementation constraint: **the wedge is pasted into the page already final**.
It must not pass through `step_saturate` or `step_apply_lut` on its way onto the sheet. In
practice that means the compose step takes finished wedge canvases as inputs rather than raw
patch data.

### 4.5 Page composition

`calibration_page()` ([targets.py:501](cyanoneg/targets.py:501)) already solves nearly all of
this, and `tricolour_page` should be written by reading it closely rather than from scratch:

- it works in **print orientation** and undoes each part's mirror on the way in
  (`part.film[:, ::-1]`, [targets.py:552](cyanoneg/targets.py:552));
- it records a `placement` per part with `x_px / y_px / w_px / h_px / x_mm / y_mm / sidecar` —
  exactly the manifest a scan-back needs to find a slot;
- it **raises** when the parts do not fit ([targets.py:539–543](cyanoneg/targets.py:539)) rather
  than cropping.

Two deltas: the background is full blocker, not white (R3); and the same three slot rectangles
are laid out on all three layers, with the two non-owning layers leaving their slots at background
(R2).

Its docstring also carries the argument for why this page exists at all: **0.19 log units** of
sheet-to-sheet coating variation. A wedge on a different sheet from the picture measures a
different sheet.

### 4.6 Page layout, in millimetres

Everything below fits A4 (210 × 297 mm) with room to spare. Numbers are given so a reviewer can
check the arithmetic rather than trust it.

```
  ┌─ A4 210 × 297 ────────────────────────────────────┐
  │  10                                          10   │
  │   ┌──────────── picture 150 × 120 ─────────┐      │   picture 130 × 100 image
  │   │  border 10                             │      │   + 10 border  = 150 × 120
  │   │        ┌──────────────────────┐        │      │
  │   │        │   130 × 100 image    │        │      │
  │   │        └──────────────────────┘        │      │
  │   └────────────────────────────────────────┘      │
  │            ↕ gap                                  │
  │   ┌ 60×60 ┐ 5 ┌ 60×60 ┐ 5 ┌ 60×60 ┐               │   3 × 60 + 2 × 5 = 190
  │   │   M   │   │   Y   │   │   C   │               │   (210 − 190) / 2 = 10 margin
  │   └───────┘   └───────┘   └───────┘               │
  │                                                   │   block height ≈ 200 of 297
  └───────────────────────────────────────────────────┘
```

The 60 × 60 mm wedge is the compact recipe HANDOFF.md already sanctions for exactly this job
(its open job #1: "print a wedge strip alongside a photograph"):

```python
step_wedge((255, 64, 0), saturation=1.0, levels=16, redundancy=4)
```

16 levels × 4 copies = 64 patches; `_grid_shape()` ([targets.py:393](cyanoneg/targets.py:393))
gives **8 × 8**; 8 × 5.5 mm + 2 × 8 mm border = **60 mm** square. This is *not* the module default
— `DEFAULT_LEVELS = 32`, `DEFAULT_REDUNDANCY = 16`
([targets.py:389–390](cyanoneg/targets.py:389)) — which is the subject of open question Q1.

The 130 mm long edge is deliberate and is a **paper** constraint, not a layout one: see §8, first
trap.

---

## 5. Files

### 5.1 New: `cyanoneg/tricolour.py`

```python
CHANNEL_INDEX = {"cyan": 0, "magenta": 1, "yellow": 2}   # red→cyan, green→magenta, blue→yellow
PRINT_ORDER   = ("magenta", "yellow", "cyan")            # physical order; cyan last, always
RESPONSE_QUANTITY = {"cyan": "lstar_r", "magenta": "lstar_g", "yellow": "lstar_b"}

def extract_channel(image: Image, layer: str) -> Image
    """Direct data[..., CHANNEL_INDEX[layer]] slice. Raises on image.is_mono."""

def step_saturate(image: Image, amount: float) -> tuple[Image, float]
    """Scale about the Rec.709 grey axis in linear light. Returns (image, clipped_fraction)."""

def step_frame(image: Image, layer: str, border_px: int, blocker_rgb, saturation,
               *, glyph: bool = True) -> tuple[Image, dict]
    """apply_frame + mirrored C/M/Y glyph at ~50% blocker coverage. Returns (framed, geometry)."""

def tricolour_page(picture: Image, wedges: dict[str, Image] | None, page: PrintSize,
                   blocker_rgb, saturation, owner: str) -> tuple[Image, dict]
    """Compose one layer's sheet. Background = full blocker. `owner`'s wedge is drawn;
    the other two slots are left at background. Returns (page, placement dict)."""

@dataclass
class TricolourLayer:
    role: str; source_channel: str; profile: str; print_order: int
    exposure_multiplier: float; sensitizer: str; chemistry: str

@dataclass
class TricolourSet:
    name: str; saturation_boost: float; border_mm: float
    registration: dict                       # {"fiducials": bool, "letter": bool}
    layers: dict[str, TricolourLayer]        # exactly the three keys
    @classmethod
    def load(cls, path) -> TricolourSet
    def save(self, path) -> Path
    def validate(self) -> list[str]
    def resolve(self, profile_dir=PROFILE_DIR) -> dict[str, Profile]

@dataclass
class TricolourResult:
    paths: dict[str, Path]                   # role → written TIFF
    manifest: dict                           # what lands in <stem>_tricolour.json
    fiducials: dict                          # geometry, identical across layers
    clipped_fraction: float                  # from step_saturate
    warnings: list[str]

def make_tricolour(source: Image, tset: TricolourSet, print_size: PrintSize, *,
                   output_dir, stem, wedges: bool = False, **kw) -> TricolourResult

def seed_provisional_set(base: Profile, name: str) -> tuple[TricolourSet, list[Profile]]
```

`TricolourSet.validate()` must assert the three layers agree on **`paper`, `film`, `film_batch`,
`working_space`, `driver_settings` and `blocker`**. The first five are the silent-error class this
codebase exists to remove — a set whose layers were measured under different driver settings is
not a set. `blocker` is on the list because the page background and the non-owner slot masks are
one colour: three layers with different blockers cannot share a page.

`step_saturate` returns the clipped-pixel fraction because boosting saturation clips
out-of-gamut colour **silently**. Warn above ~2%.

### 5.2 Set JSON

```json
{
  "name": "CassArt 300 Sm — Tricolour",
  "saturation_boost": 1.35,
  "border_mm": 10.0,
  "registration": { "fiducials": true, "letter": true },
  "layers": {
    "magenta": { "print_order": 1, "source_channel": "green",
                 "profile": "CassArt 300 Sm — Magenta", "exposure_multiplier": 1.5,
                 "sensitizer": "10/10",
                 "chemistry": "expose → wash → sodium carbonate bleach → madder root tone" },
    "yellow":  { "print_order": 2, "source_channel": "blue",
                 "profile": "CassArt 300 Sm — Yellow",  "exposure_multiplier": 2.75,
                 "sensitizer": "10/10",
                 "chemistry": "heavy overexpose → wash → carbonate bleach to Fe(III) hydroxide" },
    "cyan":    { "print_order": 3, "source_channel": "red",
                 "profile": "CassArt 300 Sm — Cyan",    "exposure_multiplier": 1.1,
                 "sensitizer": "5/5 (1:1 dilute)", "chemistry": "classic, untoned" }
  }
}
```

**Exposure has exactly one source of truth in each place it appears.** The *multiplier* lives in
the set file; `spe_seconds` (810) lives in each layer's `Profile.exposure`; the *computed seconds*
appear only in the emitted manifest. Nothing is stored twice, so nothing can disagree.

### 5.3 Modified: `cyanoneg/analyze.py` — channel-aware wedge reading

`analyze_wedge` normalises response on **L\***
([analyze.py:457–467](cyanoneg/analyze.py:457)). Correct for blue-on-white; it **will fail on a
yellow layer**, where L\* barely moves across the whole density range. Each colorant must be read
through its complementary channel: magenta → **green**, yellow → **blue**, cyan → **red**.

The hook already exists. `sample_cells(scan, sidecar, frame, quantity="lstar")`
([analyze.py:361](cyanoneg/analyze.py:361)) declares `quantity` and **never references it in the
body** (verified: no use in the body, no caller passes it). Per-patch `rgb` is already stored in
**linear light** ([analyze.py:374](cyanoneg/analyze.py:374)), which is exactly what `lightness()`
([analyze.py:77](cyanoneg/analyze.py:77)) expects.

| Site | Change |
|---|---|
| [analyze.py:403–407](cyanoneg/analyze.py:403) | alongside `lstar`, record `lstar_r/g/b = lightness(rgb[i])` and `y_r/y_g/y_b`. Same `lightness()`, so the 0–100 scale and every existing threshold keep their meaning. |
| [analyze.py:405](cyanoneg/analyze.py:405) | `rgb` is only recorded when the scan is RGB (`ndim == 3`). **New failure case:** requesting a channel quantity on a mono scan must raise a clear error, not `KeyError`. |
| [analyze.py:437](cyanoneg/analyze.py:437) | `analyze_wedge(..., quantity: str = "lstar")`; pass it through at [:444](cyanoneg/analyze.py:444). |
| [analyze.py:457,458,474](cyanoneg/analyze.py:457) | `s["lstar"]` → `s[quantity]`. |
| [analyze.py:514–515](cyanoneg/analyze.py:514) | `y_linear` → the matching `y_*`. |
| [analyze.py:517–526](cyanoneg/analyze.py:517) | gate the `DR_WINDOW` warning on `quantity == "lstar"`. `DR_WINDOW` ([analyze.py:33](cyanoneg/analyze.py:33)) is a Prussian-blue number; leaving it live means every yellow analysis screams and the user learns to ignore warnings. |
| [analyze.py:459](cyanoneg/analyze.py:459) | **leave the `paper − black < 5.0` guard exactly as is.** It now does real work: it is what catches "you tried to read a yellow layer in L\*". |
| [analyze.py:535–540](cyanoneg/analyze.py:535) | write `measurements["response_quantity"]` alongside `raw_patches`. |
| CLI | `--quantity {lstar,lstar_r,lstar_g,lstar_b}` on the `wedge` subcommand. |

`lut.derive_correction` ([lut.py:237](cyanoneg/lut.py:237)) needs **no change** — it is
quantity-agnostic.

### 5.4 Modified: `cyanoneg/proof.py`

Per R4. Three sites move together, plus the fallback branch:

| Site | Change |
|---|---|
| [proof.py:69](cyanoneg/proof.py:69) `measured_response` | `quantity = (profile.measurements or {}).get("response_quantity", "lstar")`; index `p[quantity]`; missing key → `ProofUnavailable` naming profile and quantity |
| [proof.py:118](cyanoneg/proof.py:118) `measured_colour` | same |
| [proof.py:147](cyanoneg/proof.py:147) `measured_endpoints` | same |
| [proof.py:200–205](cyanoneg/proof.py:200) fallback | raise for any non-cyan layer instead of mapping onto `PAPER_RGB`/`BLUE_RGB` |

### 5.5 Modified: `cyanoneg/gui/app.py` — a mode, not a fifth tab

A fifth tab would duplicate source / space / raw-scan / print-size / ppi / auto-orient / output
plus the `_tracked` and `_settings_fingerprint` machinery
([app.py:328–329](cyanoneg/gui/app.py:328), [app.py:379](cyanoneg/gui/app.py:379)). Instead, a
Mono/Tricolour radio drives a small set of swaps:

| Anchor | Mono | Tricolour |
|---|---|---|
| [app.py:270–276](cyanoneg/gui/app.py:270) Profile box | "Paper profile" combobox | mode radio; combobox lists **sets** |
| [app.py:262](cyanoneg/gui/app.py:262) | "Channel weights" | "Saturation boost" (+ clipped-% readout) |
| [app.py:333–336](cyanoneg/gui/app.py:333) | "Make negative" | "Make 3 negatives" |
| preview | one image | three layer thumbnails |
| [app.py:1049](cyanoneg/gui/app.py:1049) Calibrate step 3 | Analyse | + "Response channel" combobox, defaulted from the layer role |

Untouched: [app.py:289](cyanoneg/gui/app.py:289) print size, [app.py:296](cyanoneg/gui/app.py:296)
ppi, [app.py:300](cyanoneg/gui/app.py:300) auto-orient,
[app.py:309](cyanoneg/gui/app.py:309) output.

### 5.6 Output

`<stem>_1M.tif`, `<stem>_2Y.tif`, `<stem>_3C.tif` — numbered so the files sort in **printing**
order — plus `<stem>_tricolour.json` recording per-layer exposure **in seconds**, dilution,
chemistry, blocker, source profile, wedge slot placements and fiducial geometry.

That manifest is meant to be printed and taped to the darkroom wall. It should read as
instructions, not as a data structure.

---

## 6. Calibration workflow

**Stage 0 — no darkroom.** `seed_provisional_set` clones `profiles/CassArt 300 Sm.json` three
times: same blocker, same driver settings, **same measured LUT** (R1), exposure = 810 s ×
multiplier. All three `provisional: true`, non-negotiable.

**Stage 1 — print #1 *is* the calibration.** One A4 sheet per §4.6: the picture at 130 × 100 mm
plus three isolated 60 × 60 mm wedge slots. Print the three negatives, check registration on a
light table, then coat and expose Magenta → Yellow → Cyan.

**Stage 2 — scan each layer's wedge after that layer is processed and before the next is coated.**
Once cyan is down, magenta and yellow are unrecoverable. **One forgotten scan costs a full
three-session cycle.** Scan the same way as the existing calibration sheets — SilverFast raw,
converted in Photoshop (HANDOFF.md: a different scan path makes the comparison meaningless).

**Stage 3 — three `analyze_wedge` runs** with `lstar_g` (magenta), `lstar_b` (yellow), `lstar_r`
(cyan). Save three measured profiles, flip `provisional: false`. Print #2 is calibrated.

**Also run an exposure strip for yellow only.** Its 2.5–3× range is the widest, and the whole
layer depends on surviving the bleach.

---

## 7. Open questions for review

These are genuine uncertainties, not rhetorical. A reviewer's answer changes the plan.

**Q1 — Is redundancy 4 enough?**
`step_wedge`'s docstring ([targets.py:380–388](cyanoneg/targets.py:380)) argues hard for k = 16:
with 16 copies scattered across the sheet, a single dust spike or coating flaw is outvoted by the
median. HANDOFF.md sanctions k = 4 for the compact strip, but for size reasons, and against a
*known-good* process. Print #1 is neither — it is the first tricolour sheet ever made on this
paper, and its wedges are the only measurement of it. Options: accept k = 4 and 60 mm; go to
k = 8 (levels 16 → 128 patches → 12 × 11 grid, ~82 × 76 mm, three of which no longer fit side by
side); or drop to two wedge slots per sheet and print the third layer's wedge on a second sheet
(which reintroduces the 0.19-log-unit sheet variation that §4.5 exists to avoid).

**Q2 — Is `derive_correction(knots=21)` over-parameterised on 16 levels?**
`smooth_pchip` defaults to 21 knots ([lut.py:145](cyanoneg/lut.py:145)) and `derive_correction`
inherits it ([lut.py:237](cyanoneg/lut.py:237)). With 32 measured levels that is a smoother; with
16 it has more knots than data and becomes an interpolator that follows noise. Nothing in the
existing tests catches this because the existing wedge has 32 levels. Either the tricolour wedge
keeps 32 levels (and Q1's size problem gets worse) or `derive_correction` is called with a knot
count tied to the level count.

**Q3 — Should yellow's levels be distributed non-uniformly?**
The yellow layer is deliberately overexposed 2.5–3×, so most of its 16 levels may land in a
compressed region while the useful separation happens across a few. A geometric or
exposure-weighted level distribution would measure where the tone actually moves. Against: it is
speculation before print #1, it makes the three wedges structurally different, and print #1's
uniform wedge is precisely the experiment that would tell us. **Recommendation: uniform for print
#1, revisit with data.**

**Q4 — Reciprocity across a 2.75× exposure range.**
The measured LUT was derived at 1× SPE. Iron(III) photochemistry is not guaranteed to be
reciprocity-linear at 2.75× — if it is not, the yellow layer's curve is wrong in a way the
per-layer measurement at Stage 3 will absorb without ever naming. Considered and **deferred**: it
does not change what print #1 should be, and the Stage 3 measurement is taken at the working
exposure, so the effect is captured even if it is never separately identified. Worth naming here
so it is not later mistaken for a coating problem.

---

## 8. Traps

- **Paper dimensional change is not a software problem.** Cotton rag grows 0.5–1% wet, so a 240 mm
  print moves 1–2 mm across three wet/dry cycles and no fiducial fixes that. Pre-shrink the sheets
  (hot soak 30–60 min, dry fully, before coating) and keep print #1 to roughly **130 mm on the
  long edge**. Dry flat or on a cold fan only — heat warps the dimensional scale. This is why
  §4.6 says 130 × 100 mm.
- **Do not scan after cyan.** The earlier layers are gone.
- **Do not hardcode the book's blocker values** over the measured (255,64,0).
- **Silent saturation clipping.** Report the fraction; warn above ~2%.
- **`DR_WINDOW` noise on yellow.** Gate it, or every warning in the app becomes worthless.
- **Do not build the composite soft-proof before the layers are measured.** `proof.py`'s docstring
  forbids exactly that fiction, and R4 exists to keep it forbidden.
- **`git add -A` is unsafe in this folder** (HANDOFF.md) — it once swept 200 MB of photographs into
  a commit. Stage named files.
- **Windows is the release runtime** (HANDOFF.md, 11 Aug policy). Develop and run portable tests on
  macOS; build, GUI-test and smoke-test on Windows.

---

## 9. Ranking

**Required for print #1** — one module and one GUI panel:
`TricolourSet` · `extract_channel` · `step_saturate` · `step_frame` · `tricolour_page` ·
`make_tricolour` · output naming · `seed_provisional_set` · the Process-tab mode.

**Required for print #2, not #1:** the `analyze.py` quantity change and the `proof.py` change.
They are what turn print #1's wedges into three measured profiles.

**Later refinement:** composite colour soft-proof (only once all three are measured),
`batch_negatives` tricolour support, per-layer blocker re-measurement, zone blockers, the
neutral-stack experiment from R2.

---

## 10. Tests

**New `tests/test_tricolour.py`**

- Step order asserted in a **sibling class**, not by editing
  [tests/test_pipeline.py:31–52](tests/test_pipeline.py:31) — the mono order test must survive
  verbatim. Patch the tricolour chain, assert the new order, ×3.
- `extract_channel` byte-exact against `data[..., i]`; mono input raises.
- All three layers land **byte-identical** fiducial coordinates and identical page dimensions.
- Page background is full blocker, not white (R3) — assert against
  `apply_blocker(zeros, rgb, sat)`.
- In each layer's page, the two non-owned wedge slots are uniform background (R2).
- `_1M` / `_2Y` / `_3C` naming, and the manifest carries exposure in seconds.
- `TricolourSet.validate()` rejects mismatched `film_batch`, `driver_settings` **and `blocker`**.
- Saturation clip fraction is reported and the >2% warning fires.
- Glyph does not add a `detect_fiducials` candidate: run detection on a framed layer and assert
  exactly four blobs at the expected corners (R6).

**Extend `tests/test_analyze.py`**

- A synthetic **yellow** wedge, near-flat in L\* but monotone in blue: assert L\* raises at
  [analyze.py:459](cyanoneg/analyze.py:459) and that `lstar_b` derives a sane monotone curve.
  *That test is the entire point of the analyze change.*
- Channel quantity on a **mono** scan raises a clear error (§5.3, the `ndim == 3` case).

**Extend `tests/test_proof.py`**

- A profile with `response_quantity: "lstar_b"` but patches lacking `lstar_b` raises
  `ProofUnavailable` naming the quantity — it does not silently proof in L\* (R4).
- Existing profiles with no `response_quantity` behave **exactly** as now.

---

## 11. Verification

1. **Baseline first.** Run `python -m pytest` and record the passing count *before* any change
   (HANDOFF.md's last recorded figure is 239; take the real number from the run rather than
   trusting either that or the earlier draft's 272). Re-run after; the mono order test must be
   untouched.
2. Seed the provisional set; inspect the three generated profile JSONs: same blocker, same driver
   settings, same LUT values, `provisional: true`, exposures 1215 / 2228 / 891 s.
3. Run the GUI (`python -m cyanoneg`) end to end on a real colour positive at 130 × 100 mm.
   Confirm three files appear, named in print order, plus the manifest.
4. **Numeric channel check, not eyeball:** load each TIFF, run
   `blocker.recover_coverage` to recover ink, and assert the recovered coverage correlates with
   the corresponding source channel (magenta ↔ green) far more strongly than with the other two.
   The eyeball version of this test passes on a wrong permutation.
5. Open all three TIFFs: identical dimensions, identical fiducial positions, C/M/Y glyph reading
   the right way round **on the film**, and the two non-owned wedge slots flat in each.
6. Print the three negatives onto film; overlay on a light table — fiducials stack with no fringe.
7. **Only then** commit film to paper, following Magenta → Yellow → Cyan.
