# Architecture: pps-cheques processing pipeline

## Purpose

Extracts structured fields from scanned Argentine bank checks (Cheques de Pago Diferido).
Given one or more PDF files, the system produces a JSON with one record per check containing
the `monto` (amount) and `fecha_emision` (issue date) for each one.

---

## Module map

```
main.py
├── pdf_processor        — PDF → page images
├── check_detector       — page image → cropped check images
└── ChequeExtractor      — cropped check image → DatosCheque
    ├── MontoExtractor       — OCR + heuristics → monto
    ├── FechaEmisionExtractor — OCR → fecha zone tokens
    └── LLMValidator (optional) — tokens → structured fields with confidence
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

- Scans the upper-center band of the check (`y: 0–45%, x: 15–80%`) with docTR
- Looks for the token `"EL"` as a spatial anchor (always printed at the start of the payment date line, one row below the issue date)
- If found: keeps only tokens in the band `[el_cy × 0.30, el_cy − 0.05]` to exclude the check header above and the `"EL ..."` line below
- If not found: returns all tokens from the zone

Returns: `list[OCRResult]` — tokens from the fecha zone only

#### 3c — LLM validation (`LLMValidator`, optional)

- Runtime: **Ollama** (local HTTP server, default `http://localhost:11434`)
- Model: configurable (default `llama3.2`)
- One HTTP call per check (`POST /api/chat`, streaming disabled, temperature 0)

**Input to LLM**:
- All tokens from Step 3a (`zona_tokens`) + Step 3b (fecha tokens), sorted by row then column
- Batch context: `monto_raw` strings from previously processed checks in the same PDF (accumulated incrementally), used to help the LLM detect OCR anomalies
- System prompt: domain expert on Argentine CPD checks; knows Argentine numeric format; knows the `CIUDAD, DD DE MES DE AAAA` structure; instructs the model to output JSON only

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

**Field selection logic** (in `ChequeExtractor`):
- `monto`: use LLM value if `confidence ≥ 0.70`; otherwise keep OCR heuristic value
- `fecha_emision`: use LLM value if `confidence ≥ 0.70`; otherwise `null` (no OCR fallback for dates)

**Failure modes** (graceful degradation):
- Ollama unreachable → `confidence = 0.0`, OCR value kept, pipeline continues
- LLM returns malformed JSON → same fallback
- LLM explicitly returns `null` for a field → treated as missing (uses `or {}` not `.get(key, {})`)

**Timeout**: 180 seconds (CPU-only inference on llama3.2 takes 15–90s depending on hardware)

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

| Class | Library | Status |
|-------|---------|--------|
| `DocTRReader` | python-doctr (`db_resnet50` + `crnn_vgg16_bn`) | Default, used in production |
| `TesseractReader` | pytesseract | Available, not used by default |
| `EasyOCRReader` | easyocr | Available, not used by default |

All implementations return `list[OCRResult]` where coordinates `(cx, cy)` are normalized to `[0, 1]` relative to the input image crop.

---

## External dependencies

| Dependency | Role | Where used |
|------------|------|-----------|
| **PyMuPDF** | PDF rendering | `pdf_processor.py` |
| **OpenCV** | Image processing, edge detection, thresholding | `check_detector.py`, `monto_extractor.py` |
| **python-doctr** (PyTorch backend) | OCR engine | `ocr_readers.DocTRReader` |
| **Pillow** | Image save/load | `pdf_processor.py` |
| **NumPy** | Array operations throughout | all modules |
| **Ollama** (local server) | LLM inference | `llm_validator.py` via HTTP |
| **httpx** | HTTP client for Ollama API | `llm_validator.py` |

---

## Key design decisions

**Zone-based OCR instead of full-image OCR**
The check is not scanned as a whole. Two targeted crops are sent to docTR:
- top-right zone for `monto` (already required for heuristics)
- upper-center band for `fecha_emision`

This avoids redundant model calls and limits the token set sent to the LLM.

**OCR tokens are reused across steps**
`zona_sup_tokens` from the monto pass is cached in `MontoOCRResult` and passed directly to the LLM — the same docTR call covers both the heuristic scoring and the LLM input.

**No heuristic fallback for `fecha_emision`**
Date parsing from raw OCR tokens is ambiguous (month names, city names, noise). Only the LLM extracts dates. If the LLM is disabled or low-confidence, `fecha_emision` is `null`.

**Batch context accumulates incrementally**
Each check within a PDF sees the `monto_raw` values of all previously processed checks as context. This helps the LLM detect outliers (e.g. a misread `1.000.000` vs a batch of `4.000.000` values).

**LLM is optional and non-blocking**
`ChequeExtractor` works without an `LLMValidator`. Every field has a defined fallback. Ollama errors never propagate as exceptions.
