# TRICOLOUR — CassArt 300 Sm — Tricolour

Negatives: `TRESS_1M` · `_2Y` · `_3C`
Sheet: 216 x 278 mm · picture 130 x 100 mm · pre-shrink the paper

## ORDER IS NOT NEGOTIABLE

The carbonate bleach used by magenta and yellow destroys Prussian blue.
Cyan is last. Always.

## DO NOT SKIP A SCAN

Each layer's wedge is scanned after that layer dries and before the next is
coated. Those scans are the measurement this whole print exists to produce.
Whether a wedge can still be read after cyan is what this print is testing,
so it cannot be relied on — a missed scan costs a full three-session cycle.

## 1. MAGENTA — `TRESS_1M.tif`

- **Expose 20:15**  (1215 s = 810 s SPE x 1.5)
- Sensitizer: 10/10
- Then: expose → wash → sodium carbonate bleach → madder root tone
- Dry flat (cold air only — heat warps the dimensional scale).
- **SCAN the magenta wedge slot now**, before coating the next layer.
  SilverFast raw, converted in Photoshop — the same path as the existing
  calibration sheets, or the comparison means nothing.

## 2. YELLOW — `TRESS_2Y.tif`

- **Expose 37:08**  (2228 s = 810 s SPE x 2.75)
- Sensitizer: 10/10
- Then: heavy overexpose → wash → carbonate bleach to Fe(III) hydroxide
- Dry flat (cold air only — heat warps the dimensional scale).
- **SCAN the yellow wedge slot now**, before coating the next layer.
  SilverFast raw, converted in Photoshop — the same path as the existing
  calibration sheets, or the comparison means nothing.

## 3. CYAN — `TRESS_3C.tif`

- **Expose 14:51**  (891 s = 810 s SPE x 1.1)
- Sensitizer: 5/5 (1:1 dilute)
- Then: classic, untoned
- Dry flat (cold air only — heat warps the dimensional scale).
- **SCAN the cyan wedge slot now**.
  SilverFast raw, converted in Photoshop — the same path as the existing
  calibration sheets, or the comparison means nothing.

## After cyan — the experiment

- Scan all three wedge slots **and** the blocked control region.
- Read the control in **all three channels** — a stain invisible in L* can be
  large in blue, which is the layer most at risk.

The curves come from the scans taken between layers, not from these. These
answer a different question: comparing them against the earlier scans, for the
same physical patches, shows whether an isolated wedge survives the full
remaining process. The control gives the stain floor that makes that readable —
if a wedge shifted by about what the control shifted, the shift is stain rather
than damage.

If they do survive, later prints can drop the intermediate scans. That is a
conclusion to earn from this print, not one to assume before it.

## Measure the fiducial span

While the sheet is on the scanner, measure the printed picture fiducials against
their nominal spacing. That difference is the paper's shrinkage, and it is the
number that fills each layer's `scale` field for print #2. Nothing else measures
it, and it cannot be recovered later.

## Warnings

- saturation boost 1.35 clipped 19.3% of pixels; colour driven out of gamut here cannot be recovered by re-processing
- magenta, yellow, cyan still provisional — this print is the experiment that measures them, not a calibrated result
