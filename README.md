# PCB Golden-Board Re-inspection API

A FastAPI service that re-inspects a circuit-board image against a golden
reference. It removes the two nuisance factors a re-inspection camera
introduces — **global affine misalignment** and **linear exposure drift** —
and then either localizes *real* defects or judges the electrical
continuity of two copper nets. When the correction is not trustworthy
(solid-color input, periodic texture with multiple alignments, poor
correlation, insufficient overlap, untrusted conductor segmentation), the
service returns `undeterminable` instead of ever reporting a false "pass"
or "normal".

## Endpoints

- `GET /health` — liveness probe
- `POST /api/v1/inspect` — defect re-inspection (unchanged): multipart
  upload of two 512×512 grayscale PNGs, returns the defect mask/boxes in
  inspected-image coordinates.
- `POST /api/v1/continuity` — two-net continuity analysis. Upload:
  - `golden`, `inspected`: two 512×512 grayscale PNGs;
  - `labels`: 512×512 label-mask PNG in **golden coordinates**, pixel
    values `0` (background), `1` and `2` (the two nets);
  - `pads`: JSON array of four `[x, y]` pad centers in golden
    coordinates — the first two are the pads of net 1, the last two the
    pads of net 2.
  - Inputs that are not valid 512×512 images, malformed pad JSON, illegal
    label values, touching nets, or disconnected golden pads return HTTP
    400 (client error).
  - A content-level inability to decide is HTTP 200 with
    `status: "undeterminable"` and a machine-readable `reason_code`.

### Continuity method

1. The same trusted multi-scale ECC affine registration and robust
   Huber-IRLS exposure compensation as `/api/v1/inspect`
   (`app/inspection/alignment.py`) align the inspected board to the golden
   frame and restore its intensities to the golden scale.
2. The inspected conductor is reconstructed with a midpoint threshold
   between robust copper/background medians. A golden self-consistency
   gate (the thresholded golden image must reproduce the supplied labels
   at Dice ≥ 0.90), a corrected-intensity separation gate (≥ 30 gray
   levels) and per-net area plausibility gates make segmentation
   untrustworthiness an explicit `SEGMENTATION_NOT_TRUSTED` verdict.
3. **Open circuit (per net):** each net is judged only inside its own
   Voronoi region — the two pads must lie on one 8-connected conductor
   component. Because the test never leaves the net's region, a cross-net
   detour created by a short cannot mask an intra-net break.
4. **Short circuit (cross-net):** a conductor component touching both
   labels is reported as a short only when its mid-gap neck survives 3×3
   erosion, i.e. the bridge is at least 3 px wide. Only isolated 1–2 px
   specks are removed; there is no morphological opening, so a genuine
   3 px-wide bridge or neck is never erased by denoising.

### Continuity response `status`

| value             | meaning                                                  |
|-------------------|----------------------------------------------------------|
| `normal`          | both nets connect their pads and no cross-net short      |
| `open`            | one or both nets have an internal open circuit           |
| `short`           | a ≥3 px-wide conductor bridge joins the two nets         |
| `open_and_short`  | both defects coexist (the open is never hidden)          |
| `undeterminable`  | registration or segmentation not trustworthy; **never** a false normal |

Additional `reason_code`s for continuity: `SEGMENTATION_NOT_TRUSTED`,
`PAD_OUTSIDE_OVERLAP` (plus all registration/exposure codes). Returned
masks (`conductor_mask_png_base64`, `short_mask_png_base64`, debug panel)
are in **golden-frame coordinates**.

## What it does

1. **Trust gates** — reject solid-color / near-flat images and detect
   periodic textures whose autocorrelation admits an alternative alignment
   (the registration would be non-unique).
2. **Robust affine registration** — multi-scale ECC (`MOTION_AFFINE`) with
   residual-mask refits, so defect pixels are treated as outliers and
   cannot be absorbed into the warp.
3. **Plausibility / correlation / overlap gates** — wild warps, weak
   correlation and frames with too little overlap are refused.
4. **Robust linear exposure compensation** — Huber-IRLS fit of
   `I = gain * G + bias` on Gaussian-smoothed overlap pixels (smoothing
   removes the sub-pixel-registration slope bias); the inspected image is
   brought back to the golden scale by the inverse relation
   `I_corrected = (I - bias) / gain`.
5. **Defect detection** (inspect endpoint) — gradient-adaptive residual
   thresholding plus same-sign coherence and area filters, evaluated only
   inside the eroded overlap. The mask and boxes are returned in
   **inspected-image coordinates**. Non-overlap pixels are never judged.

## Coordinate convention

The reported `warp_matrix` is a 2×3 affine mapping
**golden coordinates → inspected coordinates**
(`[x_i, y_i]^T = W [x_g, y_g, 1]^T`). This matches OpenCV 4.10's
`findTransformECC` output convention (verified empirically:
`warpAffine(inspected, W, WARP_INVERSE_MAP)` reconstructs the golden image).

## Running with Docker Compose

```bash
docker compose up --build
```

Then open **http://localhost:8000/docs** — Swagger UI lets you upload the
PNG files through the `POST /api/v1/inspect` or `POST /api/v1/continuity`
forms and view the JSON result, including base64 masks and a 2×2 debug
panel.

- `GET /health` — liveness probe
- `POST /api/v1/inspect` — multipart upload of two 512×512 grayscale PNGs
- `POST /api/v1/continuity` — the same two PNGs plus a label mask and the
  four pad centers (see above)
- Inputs that are not 512×512 grayscale images, or invalid continuity
  annotations, return HTTP 400.
- A content-level inability to decide is HTTP 200 with
  `status: "undeterminable"` and a machine-readable `reason_code`.

### Response `status`

| value             | meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `pass`            | decidable, no defect found in the overlap region               |
| `defect`          | decidable, defect mask/boxes returned in inspected coordinates |
| `undeterminable`  | correction not trustworthy; **never** a false pass             |

`reason_code` for undeterminable: `SOLID_COLOR`, `LOW_TEXTURE`,
`PERIODIC_TEXTURE`, `ECC_FAILED`, `LOW_CORRELATION`, `IMPLAUSIBLE_WARP`,
`INSUFFICIENT_OVERLAP`, `LOW_INLIER_RATIO`, `EXPOSURE_FIT_FAILED`,
`IMPLAUSIBLE_EXPOSURE`.

## Local development (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# test-only extras (pytest + the httpx TestClient dependency)
pip install -r requirements-dev.txt
```

## Synthetic data and independent evaluation

Everything test-related lives under `tests/` and is import-isolated from
the detector.

```bash
# Regenerate the fixed-seed suite (20 defect + 4 clean + 4 reject cases)
python -m tests.synth.generate_samples samples

# Independent evaluation; writes samples + ground-truth manifest, then scores
python -m tests.evaluate samples

# API / functional tests (defect pipeline + continuity pipeline)
python -m pytest -q
```

### Synthetic envelope (fixed seed `20261117`)

- translation ≤ 8 px per axis, rotation ≤ 3°, gain 0.8–1.2, bias ±15
- added/removed blocks covering 1 %–10 % of the frame, gray difference ≥ 40
- extra reject cases: solid color (flat and lightly noisy) and periodic
  stripe/checker textures shifted by one full period (a real defect block
  is embedded but must not be reported because the alignment is ambiguous)

### Continuity fixture envelope

- two non-periodic nets, trace width 7 px, pad size 14 px, copper/background
  gray difference 125; open gaps 10 px and cross-net bridges 3–6 px wide
- same affine/exposure/noise disturbance as the defect suite; the label
  mask and the four pad centers are kept in golden coordinates

### Measured acceptance results

Running `python -m tests.evaluate samples` on the reference environment:

| criterion                              | limit   | measured              |
|----------------------------------------|---------|-----------------------|
| four-corner mapping error (mean, px)   | ≤ 0.5   | ~0.10                 |
| four-corner mapping error (worst, px)  | —       | ~0.21                 |
| strict pixel F1 (mean over 20 defects) | ≥ 0.90  | ~0.97                 |
| strict pixel F1 (worst case)           | —       | ~0.95                 |
| status correctness (28/28)             | all     | pass/defect/reject OK |
| solid & periodic reject cases          | reject  | undeterminable        |

### Detector independence guarantee

The detector is only ever called with two numpy arrays
(`inspect_images(golden, inspected)`). `tests/evaluate.py` additionally
performs a static audit that fails if any module under `app/inspection/`
references the generator, a manifest, ground truth, or performs file I/O.
The detector cannot read the generation parameters or ground-truth masks.

## Repository layout

```
app/
  main.py                  FastAPI app, upload validation, response schemas
  inspection/
    io_utils.py            PNG decode/encode, 512x512 grayscale/label validation
    geometry.py            robust multi-scale ECC, warp helpers, trust bounds
    trust.py               solid-color / low-texture / periodicity gates
    exposure.py            Huber-IRLS gain/bias fit, residual defect detection
    alignment.py           shared trusted registration + exposure front-end
    annotation.py          label-mask / pad-center validation (HTTP 400 errors)
    detector.py            defect re-inspection orchestration (array-only entry)
    continuity.py          two-net open/short analysis (array-only entry)
tests/
  synth/generator.py       fixed-seed synthetic board + defect generator
  synth/generate_samples.py  CLI to persist the suite
  continuity_synth.py      fixed-seed two-net boards (open/short/both fixtures)
  evaluate.py              independent scorer + detector independence audit
  test_api.py              defect-pipeline pytest API tests
  test_continuity.py       continuity-pipeline pytest API tests
Dockerfile, docker-compose.yml, requirements.txt, requirements-dev.txt
```
