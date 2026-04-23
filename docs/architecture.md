# Architecture: pps-cheques processing pipeline

## Purpose

Extracts structured fields from scanned Argentine bank checks (Cheques de Pago Diferido).
Given one or more PDF files, the system produces a JSON with one record per check containing
the `monto` (amount) and `fecha_emision` (issue date) for each one.

---

## Module map

```
main.py
├── pdf_processor           — PDF → page images
├── check_detector          — page image → cropped check images
└── ChequeExtractor         — cropped check image → DatosCheque
    ├── MontoExtractor          — OCR + heuristics → monto
    ├── FechaEmisionExtractor   — OCR → (crop_tokens, scan_window_tokens)
    └── LLMValidator (optional) — tokens → structured fields with confidence
        ├── infer_fecha()          — focused fecha call (closed-set month inference)
        └── extract_fields()       — general monto + fecha fallback call
```

---

## Processing pipeline (per PDF)

### Step 1 — PDF to page images (`pdf_processor.py`)

- Library: **PyMuPDF** (`fitz`)
- Each page rendered at **300 DPI** (zoom = 300/72 ≈ 4.17×)
- Output: list of RGB numpy arrays, one per page

### Step 2 — Check detection (`check_detector.py`)

- Library: **OpenCV**
- Strategy: Canny edge detection → dilation → external contours
- Filter criteria per contour:
  - Area ≥ 5% of page area
  - Aspect ratio between 1.3 and 5.0 (wider than tall)
  - Width ≥ 40% of page width, height ≥ 8% of page height
- Output: list of cropped RGB arrays sorted top-to-bottom, one per check

### Step 3 — Field extraction (`ChequeExtractor`)

Runs for each cropped check image. Internally has three sub-steps.

#### 3a — Monto OCR (`MontoExtractor`)

Two-pass zone-based strategy to find the handwritten amount next to `$`:

**Pass 1 — dollar-anchored crop**
- Scans the top-right 40% of the check (`y: 0–40%, x: 40–100%`) with docTR
- Searches for a token containing `$` to get its normalized center `(cx, cy)`
- Crops a window around that position (±30% height, −10% width to right edge)
- Reads the crop three times with different preprocessors: raw, Otsu threshold, 2× upscale + Otsu

**Pass 2 — fixed zones** (always runs, fills gaps)
- Tries three progressively larger zones (e.g. `x: 63–100%, y: 0–25%`)
- Stops early if a candidate with score ≥ 5.0 is already found

**Candidate scoring** — each candidate string gets a float score:
- +5.0 if it matches Argentine thousands format (`\d{1,4}(\.\d{3})+(,\d{1,2})?`)
- +1.0 if it is a plain 6+ digit number, with penalties for 8-digit strings (likely a check serial number) and 9+ digits
- +0.05 per digit

**Normalization**: `"2.500.000"` → `2500000.0`, `"802.470,20"` → `802470.2`

Returns: `MontoOCRResult { monto, monto_raw, monto_score, zona_tokens }`
where `zona_tokens` is the raw `OCRResult` list from the top-right zone, reused in Step 3c.

#### 3b — Fecha zone OCR (`FechaEmisionExtractor`)

Hybrid strategy using a two-pass approach: wide OCR scan to locate the emission line, then a
focused crop re-OCR. Returns **two token lists**: `(crop_tokens, scan_window_tokens)`.

**Pass 1 — Wide scan**
- Scans `y: 0–55%, x: 10–100%` of the check with docTR
- Used to detect the vertical position (`cy`) of the emission date line

**Emission line detection** (in priority order):

1. **DE-cluster** — looks for pairs of `"DE"` tokens at the same `cy` (within 0.06), excludes:
   - Clusters near known boilerplate text (`360`, `dias`)
   - Clusters at or above the `EL` token (payment date line)

2. **EL-anchor fallback** — if no valid DE-cluster is found, locates the `"EL"` token
   (even when fused with adjacent text, e.g. `EL19DE`, `ELZo`). The emission line
   is estimated as **1.5 × token height above** the `EL` token.

3. **Keyword fallback** — if neither anchor is found, returns all wide-scan tokens
   (or a subset near tokens that match known month names or 4-digit years).

**Pass 2 — Crop re-OCR** (when a `cy` was detected)
- Cuts a horizontal band around `cy_emision` from the original image
- Right boundary is trimmed at the leftmost long numeric token on the emission line
  (the check identifier, e.g. `13547481`), to exclude it from the date crop
- Re-runs docTR (or an optional `crop_ocr_reader`) on the crop
- Optional: if a `vision_fn` is provided, tries the vision LLM on the crop first
  (see Vision LLM path below)

**Scan-window tokens**
- Simultaneously extracts, from the wide scan, the tokens within `±_VENTANA_CY (0.07)`
  of `cy_emision`. These are positionally clean (not re-OCR'd) and free from
  boilerplate rows. Used by `infer_fecha` instead of the crop re-OCR tokens.

**Vision LLM path** (optional, `--vision-llm`)
- Triggered before the crop re-OCR if a `vision_fn` was injected at construction
- Crop is upscaled to ≥ 200px height (LANCZOS) before sending to the vision model
  to compensate for the extreme aspect ratio (~100×1500px)
- Only accepts output that matches `^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$` exactly
- If the vision model refuses or gives free text, falls through to regular crop re-OCR

Returns: `tuple[list[OCRResult], list[OCRResult]]` → `(crop_tokens, scan_window_tokens)`

#### 3c — LLM validation (`LLMValidator`, optional)

- Runtime: **Ollama** (local HTTP server, default `http://localhost:11434`)
- Model: configurable (default `llama3.2`)
- Temperature: 0 (deterministic)
- **Two HTTP calls per check** (both optional, both gracefully degraded on failure)

---

**Call 1 — `infer_fecha` (focused fecha inference)**

Purpose: infer the emission date from the raw OCR characters, using a closed set of 12
Spanish month names and explicit OCR confusion rules.

Input: `scan_window_tokens` from Step 3b (the wide-scan tokens within ±0.07 of `cy_emision`).
These are cleaner than the crop re-OCR tokens because they retain positional context and
are not contaminated by adjacent boilerplate rows.

The user prompt provides:
- The exact token characters as OCR read them (no coordinates, sorted left-to-right by cx)
- The complete closed list of 12 Spanish months with their ISO numbers
- Common OCR digit confusion rules: `Z↔2`, `o↔0`, `l↔1`, `S↔5`, `G↔6`
  with examples (`"Z025"` → 2025, `"Nor"` → Noviembre)
- The maximum allowed date (`today_max = date.today().isoformat()`) — dates in the future
  are rejected outright

Expected output: `YYYY-MM-DD` or `null`. Any other format → treated as failure.

Post-processing:
- Output validated against `^\d{4}-\d{2}-\d{2}$`
- Rejected if `normalized > today_max`
- On success: `confidence = 0.88` (hardcoded — reflects that this is an inference, not a read)

---

**Call 2 — `extract_fields` (general monto + fecha fallback)**

Input: all tokens from Step 3a (`zona_tokens`) + Step 3b (`crop_tokens`), sorted by row then column.
Also receives batch context: `monto_raw` strings of previously processed checks in this PDF.

System prompt: domain expert on Argentine CPD checks; knows Argentine numeric format; knows
the `CIUDAD, DD DE MES DE AAAA` structure; instructs the model to output JSON only.

**LLM JSON output schema**:
```json
{
  "monto":         { "value": "4.000.000", "confidence": 0.97, "reasoning": "..." },
  "fecha_emision": { "value": "2026-02-11", "confidence": 0.95, "reasoning": "..." }
}
```

**Confidence calibration** (instructed in system prompt):
| Range | Meaning |
|-------|---------|
| 0.95–1.00 | Unambiguous, standard format |
| 0.80–0.94 | Readable with OCR noise, reconstruction confident |
| 0.60–0.79 | Partially reconstructed using batch context |
| 0.00–0.59 | LLM is guessing — treat as unreliable |

---

**Field selection logic** (in `ChequeExtractor`):

```
fecha_emision:
  1. infer_fecha result, if confidence ≥ 0.70          ← primary path
  2. extract_fields["fecha_emision"], if confidence ≥ 0.70  ← fallback
  3. null

monto:
  extract_fields["monto"], if confidence ≥ 0.70 and normalized is not None
  otherwise: OCR heuristic value from Step 3a
```

**Failure modes** (graceful degradation):
- Ollama unreachable → `confidence = 0.0`, OCR value kept, pipeline continues
- LLM returns malformed JSON → same fallback
- `infer_fecha` returns future date → rejected, fallback to `extract_fields` fecha
- `infer_fecha` returns unexpected format → `_FAILED_RESULT` (conf = 0.0)

**Timeout**: 180 seconds per call (CPU-only inference takes 15–90s per call)

---

## Data model (`DatosCheque`)

| Field | Type | Source |
|-------|------|--------|
| `monto` | `float \| None` | OCR heuristic, overridden by LLM if confidence ≥ 0.70 |
| `monto_raw` | `str` | Raw string as read by OCR or LLM |
| `monto_score` | `float` | Heuristic score from OCR pass (independent of LLM) |
| `monto_llm_confidence` | `float \| None` | LLM confidence for monto (0.0–1.0), `None` if LLM disabled |
| `fecha_emision` | `str \| None` | ISO date `"YYYY-MM-DD"`, LLM only |
| `fecha_emision_raw` | `str \| None` | Date as extracted by LLM before normalization |
| `fecha_emision_llm_confidence` | `float \| None` | LLM confidence for fecha |
| `imagen_path` | `str` | Path to saved PNG crop of this check |
| `pdf_origen` | `str` | Source PDF filename |
| `pagina` | `int` | Page number within the PDF |
| `indice_en_pagina` | `int` | Check index on its page (1-based) |

---

## OCR abstraction (`OCRReader`)

`OCRReader` is an abstract base class. Concrete implementations:

| Class | Library | CLI flag | Status |
|-------|---------|----------|--------|
| `DocTRReader` | python-doctr (`db_resnet50` + `crnn_vgg16_bn`) | default | Default for wide scan and crop re-OCR |
| `TrOCRReader` | transformers TrOCR (handwritten model) | `--trocr` | Optional crop re-OCR for fecha |
| `SuryaReader` | surya-ocr | `--surya` | Full replacement for DocTRReader |
| `TesseractReader` | pytesseract | — | Available, not used by default |
| `EasyOCRReader` | easyocr | — | Available, not used by default |

All implementations return `list[OCRResult]` where coordinates `(cx, cy)` are normalized to `[0, 1]`
relative to the input image crop.

`OCRResult` fields: `text`, `confidence`, `cx`, `cy`, `height` (all normalized 0–1).

---

## CLI flags (`main.py procesar`)

| Flag | Effect |
|------|--------|
| `--sin-llm` | Disable LLM entirely (OCR heuristics only) |
| `--llm-model` | Ollama model for text LLM (default: `llama3.2`) |
| `--llm-url` | Ollama server URL (default: `http://localhost:11434`) |
| `--trocr` | Use TrOCR for the fecha crop re-OCR step |
| `--surya` | Use Surya as the full OCR engine instead of docTR |
| `--vision-llm` | Enable vision LLM for fecha crop (requires `ollama pull llava:7b`) |
| `--vision-model` | Vision model name (default: `llava:7b`) |
| `--debug` | Write intermediate images and full debug log to `output/debug/` |

---

## External dependencies

| Dependency | Role | Where used |
|------------|------|-----------|
| **PyMuPDF** | PDF rendering | `pdf_processor.py` |
| **OpenCV** | Image processing, edge detection, thresholding | `check_detector.py`, `monto_extractor.py` |
| **python-doctr** (PyTorch backend) | OCR engine | `ocr_readers.DocTRReader` |
| **Pillow** | Image save/load, crop upscaling for vision LLM | `pdf_processor.py`, `fecha_emision_extractor.py` |
| **NumPy** | Array operations throughout | all modules |
| **Ollama** (local server) | LLM and vision LLM inference | `llm_validator.py`, `llm_backends.py` via HTTP |
| **httpx** | HTTP client for Ollama API | `llm_backends.py` |

---

## Key design decisions

**Zone-based OCR instead of full-image OCR**
The check is not scanned as a whole. Two targeted crops are sent to docTR:
- top-right zone for `monto`
- upper 55% for `fecha_emision` (wide scan), then a narrower band for the crop re-OCR

This limits the token set sent to the LLM and avoids running the model on irrelevant regions.

**OCR tokens are reused across steps**
`zona_tokens` from the monto pass is cached in `MontoOCRResult` and passed directly to the LLM —
the same docTR call covers both the heuristic scoring and the LLM input.

**Two separate LLM calls for fecha and monto**
`infer_fecha` is a focused call with a tight, specialized prompt (closed-set months, OCR confusion
rules, future-date rejection). `extract_fields` is a broader call covering monto and fecha fallback.
Splitting them improves fecha accuracy by removing noise from the monto zone tokens.

**Scan-window tokens preferred over crop re-OCR for fecha inference**
The crop re-OCR can introduce new problems: it may drop tokens that were found in context (e.g.
losing `"16"` and `"Nor"` from a date like `16 DE Nor DE Z025`), or pick up boilerplate text from
adjacent rows. The wide-scan tokens filtered to ±0.07 cy around the emission line are more
positionally stable and are passed to `infer_fecha` instead.

**Closed-set month inference**
There are exactly 12 possible Spanish month names. Even heavily garbled OCR output can be matched
to a month unambiguously (`"Nor"` → only Noviembre starts with N-o-r). The LLM prompt enforces
this constraint explicitly, making month identification robust to partial OCR.

**Future-date rejection**
`infer_fecha` passes `today_max = date.today().isoformat()` to the LLM as a hard upper bound,
and also validates the returned date server-side. This prevents OCR digit confusions like `Z→2`
from producing years like 8025 or 2226.

**Vision LLM is a supplementary path, not the primary one**
`llava:7b` takes 60–70s per crop on CPU and often refuses to output a bare ISO date despite
prompt engineering. It is kept as an optional flag (`--vision-llm`) and falls through to regular
crop re-OCR if it does not return a valid ISO string.

**No heuristic fallback for `fecha_emision`**
Date parsing from raw OCR tokens is ambiguous (month names, city names, noise). Only the LLM
extracts dates. If both LLM calls fail or are disabled, `fecha_emision` is `null`.

**Batch context accumulates incrementally**
Each check within a PDF sees the `monto_raw` values of all previously processed checks as context.
This helps the LLM detect outliers (e.g. a misread `1.000.000` vs a batch of `4.000.000` values).

**LLM is optional and non-blocking**
`ChequeExtractor` works without an `LLMValidator`. Every field has a defined fallback. Ollama
errors never propagate as exceptions.
