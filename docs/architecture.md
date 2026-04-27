# Architecture: pps-cheques processing pipeline

## Purpose

Extracts structured fields from scanned Argentine bank checks (Cheques de Pago Diferido).
Given one or more PDF files, the system produces a JSON with one record per check containing
the `monto` (amount), `fecha_emision` (issue date), and `fecha_pago` (payment date) for each one.

---

## Module map

```
main.py
├── pdf_processor           — PDF → page images
├── check_detector          — page image → cropped check images
└── ChequeExtractor         — cropped check image → DatosCheque
    ├── MontoExtractor          — OCR + heuristics → monto
    ├── FechaEmisionExtractor   — OCR + structural parse → fecha_emision
    │   └── fecha_extractor     — shared date utilities (Fecha, FechaResult, parsers)
    └── FechaPagoExtractor      — OCR + structural parse → fecha_pago
        └── fecha_extractor     — (same shared module)
```

Optional LLM path (requires `--con-llm`):
```
ChequeExtractor
└── LLMValidator (ThreadPoolExecutor, up to 3 parallel calls)
    ├── infer_fecha → fecha_emision (when OCR parse incomplete)
    ├── infer_fecha → fecha_pago    (when OCR parse incomplete)
    └── extract_fields → monto     (when OCR heuristic finds nothing)
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

Runs for each cropped check image. Has three OCR sub-steps, then optional parallel LLM calls.

#### 3a — Monto OCR (`MontoExtractor`)

Two-pass zone-based strategy to find the handwritten amount next to `$`:

**Pass 1 — dollar-anchored crop**
- Scans the top-right zone (`y: 0–40%, x: 40–100%`) with docTR
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
where `zona_tokens` is the raw `OCRResult` list from the top-right zone, reused in Step 3d if LLM is enabled.

#### 3b — Fecha emision OCR (`FechaEmisionExtractor`)

Wide-scan strategy over `y: 0–55%, x: 10–80%` of the check.

**Emission line detection** (in priority order):

1. **City-comma anchor** — looks for a token matching `^[A-Za-záéíóú...]+,$` (a city name
   ending in comma, e.g. `FEDERAL,`, `CUATIA,`, `QUILMES,`) with `cy > 0.30`. When found:
   - `cy_emision` is set to the token's `cy`
   - The scan window is restricted to tokens to the right of the city token on the same row
   - This is the most accurate anchor because the city name always precedes the date on the same line

2. **DE-cluster** — looks for pairs of `"DE"` tokens at the same `cy` (within 0.06), excludes:
   - Clusters near known boilerplate text (`360`, `dias`)
   - Clusters at or above the `EL` token (payment date line)

3. **EL-anchor fallback** — if no valid DE-cluster is found, locates the `"EL"` token
   (even when fused with adjacent text, e.g. `EL19DE`, `ELZo`). The emission line
   is estimated as **1.5 × token height above** the `EL` token.

4. **Keyword fallback** — if no anchor is found, returns tokens in the upper 40% of the check
   (bounded below by the `"plazo"` boilerplate token when present), optionally narrowed to
   a row with known month or year tokens.

**Structural parse attempt**: once the scan window is obtained, `_filtrar_tokens_fecha_estructura`
looks for the `DIA DE MES DE ANNO` pattern. If a complete, valid date is found, `fecha_iso` is
set directly. Otherwise, `FechaResult` carries the raw tokens and a `partial` `Fecha` (with
whichever components OCR did recognize) for the LLM to complete.

Returns: `FechaResult { fecha_iso, tokens, partial }`

#### 3c — Fecha pago OCR (`FechaPagoExtractor`)

Mirrors `FechaEmisionExtractor` but targets the payment date line (lower on the check).
Scans `y: 0–55%, x: 10–70%`.

**Payment line detection** (in priority order):

1. **EL anchor** — finds all tokens starting with `"EL"` with `cy > 0.35`, takes the lowest
   one (highest `cy`) to avoid confusing it with the emission line.

2. **PAGUESE anchor** — if no `EL` token is found, locates `"Paguese"/"PAGUESE"`. The payment
   date is estimated **1.5 × token height above** it.

3. **Lower DE cluster** — finds the lowest DE-cluster (highest `cy`) with ≥ 2 tokens, excluding
   boilerplate. Complements the cluster that `FechaEmisionExtractor` discarded.

4. **Keyword fallback** — groups month/year anchor tokens into `cy` bands, takes the lowest band
   (payment date is always below the emission date).

Same structural parse as 3b. Returns: `FechaResult { fecha_iso, tokens, partial }`

#### 3d — Parallel LLM calls (`ChequeExtractor`, optional)

When `--con-llm` is passed, up to three LLM calls are launched in parallel via
`ThreadPoolExecutor(max_workers=3)`:

- `infer_fecha(fecha_result.tokens, today_max=today, partial=partial_fecha)` for `fecha_emision`
  (only if `fecha_iso` is `None` after Step 3b)
- `infer_fecha(fecha_pago_result.tokens, max_future_days=365, partial=partial_fecha)` for `fecha_pago`
  (only if `fecha_iso` is `None` after Step 3c)
- `extract_fields(monto_result.zona_tokens, [])` for `monto`
  (only if `monto` is `None` after Step 3a)

After each LLM call, `_apply_fecha_overrides` locks in OCR-confirmed date components
(day, month, or year from `partial`) over the LLM result to prevent model drift.

```
fecha_emision:
  OCR structural parse result (if complete)
  otherwise: LLM infer_fecha result (if --con-llm and model returned a valid ISO date)
  otherwise: None

fecha_pago:
  OCR structural parse result (if complete)
  otherwise: LLM infer_fecha result (if --con-llm and model returned a valid ISO date)
  otherwise: None

monto:
  OCR heuristic value from Step 3a (if found)
  otherwise: LLM extract_fields result (if --con-llm and confidence >= 0.70)
  otherwise: None
```

---

## Shared date utilities (`fecha_extractor.py`)

Contains dataclasses and parsing functions used by both date extractors.

### `Fecha` dataclass

Holds per-component validation results:

| Field | Type | Meaning |
|-------|------|---------|
| `dia` | `str \| None` | Zero-padded day `"01"`–`"31"`, `None` if unrecognized |
| `mes` | `str \| None` | Month number `"01"`–`"12"`, `None` if unrecognized |
| `anno` | `str \| None` | 4-digit year `"2020"`–`"2030"`, `None` if out of range |
| `dia_raw` | `str \| None` | Raw OCR text for the day slot |
| `mes_raw` | `str \| None` | Raw OCR text for the month slot |
| `anno_raw` | `str \| None` | Raw OCR text for the year slot |

`to_iso()` returns `"YYYY-MM-DD"` only when all three components are known.

### `FechaResult` dataclass

| Field | Type | Meaning |
|-------|------|---------|
| `fecha_iso` | `str \| None` | ISO date if directly parsed, else `None` |
| `tokens` | `list[OCRResult]` | Tokens from the relevant scan window |
| `partial` | `Fecha \| None` | Components the OCR confirmed (for LLM hint) |

### Key regex constants

| Name | Pattern | Purpose |
|------|---------|---------|
| `_DE_RE` | `^[^a-zA-Z0-9]*[Dd][Ee][^a-zA-Z0-9]*$` | Match "DE" tokens |
| `_EL_INICIO_RE` | `^[Ee][Ll]` | Detect "EL" (possibly fused) |
| `_BOILERPLATE_RE` | `^(360\|dias?)$` | Legal boilerplate exclusion |
| `_VENTANA_CY` | `0.07` | Row grouping threshold |

---

## Data model (`DatosCheque`)

| Field | Type | Source |
|-------|------|--------|
| `monto` | `float \| None` | OCR heuristic (Step 3a), or LLM if OCR found nothing |
| `monto_raw` | `str` | Raw string as read by OCR or LLM |
| `monto_score` | `float` | Heuristic score from OCR pass |
| `monto_llm_confidence` | `float \| None` | LLM confidence, or `None` if OCR succeeded |
| `fecha_emision` | `str \| None` | ISO date from OCR parse or LLM, or `None` |
| `fecha_emision_raw` | `str \| None` | Same as `fecha_emision` (ISO directly) |
| `fecha_emision_llm_confidence` | `float \| None` | LLM confidence if LLM was used |
| `fecha_pago` | `str \| None` | ISO date from OCR parse or LLM, or `None` |
| `fecha_pago_raw` | `str \| None` | Same as `fecha_pago` (ISO directly) |
| `fecha_pago_llm_confidence` | `float \| None` | LLM confidence if LLM was used |
| `imagen_path` | `str` | Path to saved PNG crop of this check |
| `pdf_origen` | `str` | Source PDF filename |
| `pagina` | `int` | Page number within the PDF |
| `indice_en_pagina` | `int` | Check index on its page (1-based) |

### JSON output format

Results are written to `cheques.json` as an **array of runs**. Each run represents one program
execution and contains the file name, timestamp, and the checks processed:

```json
[
  {
    "fecha_proceso": "2026-04-24T10:00:00",
    "nombre_archivo": "Scan CH1.pdf",
    "cheques": [{ ... }, { ... }]
  }
]
```

Each execution appends a new run to the existing array (does not overwrite).

---

## OCR abstraction (`OCRReader`)

`OCRReader` is an abstract base class. Concrete implementations:

| Class | Library | CLI flag | Status |
|-------|---------|----------|--------|
| `DocTRReader` | python-doctr (`db_resnet50` + `crnn_vgg16_bn`) | default | Default for all OCR zones |
| `TrOCRReader` | transformers TrOCR (handwritten model) | `--trocr` | Optional, used for wide scan |
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
| `--trocr` | Use TrOCR for the OCR step |
| `--surya` | Use Surya as the full OCR engine instead of docTR |
| `--vision-llm` | Flag exists in CLI but has no effect (implementation currently disabled) |
| `--vision-model` | Vision model name (default: `llava:7b`) — only relevant if `--vision-llm` is re-enabled |
| `--debug` | Write intermediate images and full debug log to `output/debug/` |

---

## External dependencies

| Dependency | Role | Where used |
|------------|------|-----------|
| **PyMuPDF** | PDF rendering | `pdf_processor.py` |
| **OpenCV** | Image processing, edge detection, thresholding | `check_detector.py`, `monto_extractor.py` |
| **python-doctr** (PyTorch backend) | OCR engine | `ocr_readers.DocTRReader` |
| **Pillow** | Image save/load, debug crop images | `pdf_processor.py`, date extractors |
| **NumPy** | Array operations throughout | all modules |
| **Ollama** (local server) | LLM inference | `llm_validator.py`, `llm_backends.py` via HTTP |
| **httpx** | HTTP client for Ollama API | `llm_backends.py` |

---

## Key design decisions

**Zone-based OCR instead of full-image OCR**
The check is not scanned as a whole. Three targeted zones are sent to docTR:
- top-right zone for `monto`
- upper 55% (left 70%) for `fecha_emision` (wide scan)
- upper 55% (left 60%) for `fecha_pago` (wide scan)

This limits the token set and avoids running the model on irrelevant regions.

**City-comma anchor for emission date positioning**
The emission date line on Argentine CPD checks always starts with a city name followed by a
comma (`CAPITAL FEDERAL,`, `CURUZU CUATIA,`, etc.). When this token is found in the wide scan,
it provides both the vertical position (`cy`) and the left boundary for the scan window —
cutting out the city name and leaving only the handwritten date tokens. This is the most
reliable anchor strategy and is attempted first.

**EL anchor disambiguates emission vs. payment date**
Both date lines appear in the upper portion of the check. The key differentiator is the "EL"
token that precedes the payment date (`"EL DD DE MES DE AAAA"`). `FechaEmisionExtractor`
excludes DE-clusters that are at or below the EL token; `FechaPagoExtractor` uses the EL token
as its primary anchor (taking the lowest one if multiple appear).

**Structural parse before LLM**
Both date extractors attempt to parse `DIA DE MES DE ANNO` from OCR tokens before calling
the LLM. When handwriting is clear enough for docTR to read all components correctly, the
LLM is never called for that field — saving latency. The `partial` Fecha records which
components OCR confirmed so the LLM only needs to infer the uncertain ones.

**OCR overrides on LLM output**
After the LLM returns a date, `_apply_fecha_overrides` replaces any component that OCR
confirmed (day, month, or year in the `partial` Fecha) with the OCR value. This prevents
temperature drift in the model from corrupting a component that OCR already read correctly.

**Parallel LLM calls**
When `--con-llm` is active and multiple fields need LLM inference, all calls are dispatched
concurrently via `ThreadPoolExecutor(max_workers=3)`. Total latency is bounded by the slowest
single call rather than the sum.

**Text LLM disabled by default**
The `LLMValidator` is available but not loaded unless `--con-llm` is passed. Without it,
`fecha_emision` and `fecha_pago` are `None` whenever the OCR structural parse is incomplete.
This avoids requiring a running Ollama server for the default use case.

**Vision LLM removed**
An earlier implementation supported a `--vision-llm` path where a vision model (llava:7b)
would read the date directly from an image crop. This was removed from the active pipeline;
the `--vision-llm` CLI flag still exists but has no effect. The text LLM with structural
OCR hints now covers the same use case with lower resource requirements.

**Append-only JSON output**
Each run appends to the existing `cheques.json` array rather than overwriting it. This
preserves a full history of processing runs and allows comparing results across different
parameter combinations (e.g., with and without `--con-llm`).

**LLM components are optional and non-blocking**
`ChequeExtractor` works without any LLM. Every field has a defined fallback. Ollama errors
never propagate as exceptions.
