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
    └── FechaEmisionExtractor   — OCR + optional Vision LLM → fecha_emision
        ├── vision_fn (optional)   — crop image → ISO date string (Vision LLM path)
        └── crop_ocr_reader        — fecha crop → OCR tokens (fallback)
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
focused crop fed to the Vision LLM (primary) or a crop re-OCR (fallback).
Returns **two token lists**: `(crop_tokens, scan_window_tokens)`.

**Pass 1 — Wide scan**
- Scans `y: 0–55%, x: 10–100%` of the check with docTR
- Used to detect the vertical position (`cy`) of the emission date line

**Emission line detection** (in priority order):

1. **City-comma anchor** — looks for a token matching `^[A-Za-záéíóú...]+,$` (a city name
   ending in comma, e.g. `FEDERAL,`, `CUATIA,`, `QUILMES,`) with `cy > 0.30`. When found:
   - `cy_emision` is set to the token's `cy`
   - `x_left` is set to the token's absolute x position, so the crop starts just after the city name
   - This is the most accurate anchor because the city name always precedes the date on the same line

2. **DE-cluster** — looks for pairs of `"DE"` tokens at the same `cy` (within 0.06), excludes:
   - Clusters near known boilerplate text (`360`, `dias`)
   - Clusters at or above the `EL` token (payment date line)

3. **EL-anchor fallback** — if no valid DE-cluster is found, locates the `"EL"` token
   (even when fused with adjacent text, e.g. `EL19DE`, `ELZo`). The emission line
   is estimated as **1.5 × token height above** the `EL` token.

4. **Keyword fallback** — if no anchor is found, returns all wide-scan tokens
   (or a subset near tokens that match known month names or 4-digit years).

**Pass 2 — Vision LLM or crop re-OCR** (when a `cy` was detected)
- Cuts a horizontal band around `cy_emision` from the original image
- Right boundary is trimmed at the leftmost long numeric token on the emission line
  (the check identifier, e.g. `13547481`), to exclude it from the date crop
- If a `vision_fn` is provided (see Vision LLM path below), it is called first on the crop
- Otherwise falls through to the `crop_ocr_reader`

**Scan-window tokens**
- Simultaneously extracts, from the wide scan, the tokens within `±_VENTANA_CY (0.07)`
  of `cy_emision`. Returned alongside crop tokens and available for downstream use.

**Vision LLM path** (optional, `--vision-llm`)
- Triggered before the crop re-OCR if a `vision_fn` was injected at construction
- Crop is upscaled to ≥ 200px height (LANCZOS) before sending to the vision model
  to compensate for the extreme aspect ratio (~100×1500px)
- Prompt instructs the model to read the date from the image, output ISO `YYYY-MM-DD` only,
  and commit to a best reading even under imperfect handwriting
- Response is parsed with `re.search` for any `YYYY-MM-DD` pattern (model may add explanation text)
- If a valid ISO date is found it is returned directly as the fecha result (no OCR fallback needed)
- If the model returns nothing or no date is found, falls through to regular crop re-OCR

Returns: `tuple[list[OCRResult], list[OCRResult]]` → `(crop_tokens, scan_window_tokens)`

#### 3c — Field assembly (`ChequeExtractor`)

The text LLM validator (`LLMValidator`) is present in the codebase but **disabled by default**.
When disabled, field assembly is:

```
fecha_emision:
  Vision LLM result (if --vision-llm and model returned a valid ISO date)
  otherwise: "-" (no heuristic date parsing from raw OCR tokens)

monto:
  OCR heuristic value from Step 3a (always)
```

The text LLM can be re-enabled with `--con-llm` (see CLI flags).

---

## Data model (`DatosCheque`)

| Field | Type | Source |
|-------|------|--------|
| `monto` | `float \| None` | OCR heuristic (Step 3a) |
| `monto_raw` | `str` | Raw string as read by OCR |
| `monto_score` | `float` | Heuristic score from OCR pass |
| `monto_llm_confidence` | `float \| None` | Always `None` (text LLM disabled by default) |
| `fecha_emision` | `str \| None` | ISO date `"YYYY-MM-DD"` from Vision LLM, or `"-"` if not found |
| `fecha_emision_raw` | `str \| None` | Same as `fecha_emision` (Vision LLM returns ISO directly) |
| `fecha_emision_llm_confidence` | `float \| None` | Always `None` (text LLM disabled by default) |
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
| `--con-llm` | Enable text LLM validation (requires Ollama; disabled by default) |
| `--llm-model` | Ollama model for text LLM (default: `llama3.2`) |
| `--llm-url` | Ollama server URL (default: `http://localhost:11434`) |
| `--trocr` | Use TrOCR for the fecha crop re-OCR step |
| `--surya` | Use Surya as the full OCR engine instead of docTR |
| `--vision-llm` | Enable Vision LLM for fecha date reading (requires `ollama pull llava:7b`) |
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
- upper 55% for `fecha_emision` (wide scan), then a narrower band for the crop

This limits the token set and avoids running the model on irrelevant regions.

**City-comma anchor for date crop positioning**
The emission date line on Argentine CPD checks always starts with a city name followed by a
comma (`CAPITAL FEDERAL,`, `CURUZU CUATIA,`, etc.). When this token is found in the wide scan,
it provides both the vertical position (`cy`) and the left boundary (`x_left`) for the date
crop — cutting out the city name and leaving only the handwritten date portion for the Vision LLM.
This is the most reliable anchor strategy and is attempted first.

**Vision LLM as the primary fecha extractor**
`FechaEmisionExtractor` calls the Vision LLM (when enabled) before crop re-OCR. The Vision LLM
reads the date directly from the image crop, bypassing OCR confusion with handwritten characters.
Output is parsed with `re.search` to extract any ISO date from the response, so the model can
include natural-language explanation without breaking the pipeline.

**Crop upscaling before Vision LLM**
Date strips are ~100×1500px — extreme aspect ratios that vision models compress into 336×336
and lose detail. Upscaling to ≥ 200px height (LANCZOS, preserving aspect ratio) before sending
improves legibility for the model.

**Text LLM disabled by default**
The `LLMValidator` (text LLM for `infer_fecha` / `extract_fields`) is available but not loaded
unless `--con-llm` is passed. This avoids loading a second model (`llama3.2`) into memory when
the Vision LLM is in use, and reflects that the Vision LLM now covers the fecha extraction path.

**No heuristic fallback for `fecha_emision`**
Date parsing from raw OCR tokens is ambiguous (month names, city names, noise). When the Vision
LLM is disabled or fails, `fecha_emision` is `"-"` rather than a heuristic guess.

**OCR tokens are reused across steps**
`zona_tokens` from the monto pass is cached in `MontoOCRResult`. Scan-window tokens from the
fecha wide scan are returned alongside crop tokens and can be used by downstream components
(e.g. the text LLM if re-enabled).

**LLM components are optional and non-blocking**
`ChequeExtractor` works without any LLM. Every field has a defined fallback. Ollama errors
never propagate as exceptions.
