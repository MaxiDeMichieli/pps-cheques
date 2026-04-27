# Documento de implementacion - Lector de cheques escaneados

## 1. Objetivo del proyecto

Desarrollar un programa en Python que lea archivos PDF con cheques argentinos de pago diferido escaneados, detecte cada cheque individual dentro de la pagina, recorte su imagen, extraiga los datos relevantes mediante OCR local (sin enviar datos por internet), y los almacene en un archivo JSON junto con las imagenes individuales.

## 2. Estructura del proyecto

```
C:\Folders\UNGS\PPS\Proyecto\
├── main.py                    # CLI - punto de entrada
├── requirements.txt           # Dependencias
├── src/
│   ├── __init__.py
│   ├── models.py                   # Modelo de datos y persistencia JSON
│   ├── detection/
│   │   └── check_detector.py       # Detección y recorte de cheques
│   ├── extractors/
│   │   ├── cheque_extractor.py     # Orquestador principal de extracción
│   │   ├── monto_extractor.py      # Extracción del monto usando OCRReader
│   │   ├── fecha_extractor.py      # Utilidades OCR compartidas para fechas
│   │   ├── fecha_emision_extractor.py  # Extracción de fecha de emisión
│   │   └── fecha_pago_extractor.py     # Extracción de fecha de pago
│   ├── ocr/
│   │   └── ocr_readers.py          # Abstracción de lectores OCR (docTR, TrOCR, Surya, etc.)
│   ├── llm/
│   │   ├── llm_backends.py         # Backends HTTP para LLMs (Ollama, etc.)
│   │   └── llm_validator.py        # Validación de campos con LLM
│   └── pdf/
│       └── pdf_processor.py       # Conversión PDF -> imágenes
├── output/     
│   ├── cheques.json                # Datos extraidos (array de runs)
│   └── images/                     # Imagenes recortadas de cada cheque
└── docs/     
    ├── implementacion.md           # Este documento
    └── pruebas_modelos.md          # Pruebas de modelos OCR
```

## 3. Pipeline de procesamiento

El procesamiento de un PDF sigue estos pasos secuenciales:

```
PDF de entrada
      |
      v
[1] Conversion a imagen (300 DPI)
      |
      v
[2] Deteccion de cheques individuales (OpenCV)
      |
      v
[3] Recorte y guardado de cada cheque como PNG
      |
      v
[4] OCR de zonas relevantes (abstraccion OCRReader)
      |
      +---> [4a] Extraccion de monto (heuristica + OCR)
      |
      +---> [4b] Extraccion de fecha de emision (OCR + parsing estructural)
      |
      +---> [4c] Extraccion de fecha de pago (OCR + parsing estructural)
      |
      v
[5] Validacion/refinamiento con LLM (opcional, paralelo)
      |
      +---> [5a] Inferencia de fecha_emision (si no se pudo parsear)
      |
      +---> [5b] Inferencia de fecha_pago (si no se pudo parsear)
      |
      +---> [5c] Extraccion de monto (solo si OCR no encontro nada)
      |
      v
[6] Normalizacion y guardado en JSON
```

**Nota:** Los pasos [4a], [4b] y [4c] son coordinados por `ChequeExtractor`. Las llamadas al LLM en [5] se ejecutan en paralelo con `ThreadPoolExecutor`.

## 4. Detalle de cada modulo

### 4.1 Conversion PDF a imagenes (`pdf_processor.py`)

**Biblioteca utilizada:** PyMuPDF (fitz)

Convierte cada pagina del PDF a una imagen RGB a 300 DPI. Se eligio 300 DPI como balance entre calidad de OCR y uso de memoria. Una pagina A4 a 300 DPI genera una imagen de aproximadamente 2481x3508 pixeles.

**Alternativa descartada:** `pdf2image` requiere Poppler como dependencia externa del sistema operativo, lo cual complica la instalacion en Windows. PyMuPDF es autocontenido.

### 4.2 Deteccion y recorte de cheques (`check_detector.py`)

**Biblioteca utilizada:** OpenCV

Detecta los cheques individuales dentro de una pagina escaneada que puede contener multiples cheques. El algoritmo:

1. Convierte la imagen a escala de grises.
2. Aplica GaussianBlur (kernel 5x5) para reducir ruido.
3. Detecta bordes con el algoritmo de Canny (umbrales 30-100).
4. Dilata los bordes con un kernel rectangular 15x15 (3 iteraciones) para cerrar huecos en los contornos.
5. Encuentra contornos externos con `findContours`.
6. Filtra los contornos por:
   - **Area minima:** mayor al 5% del area total de la pagina.
   - **Relacion de aspecto:** entre 1.3 y 5.0 (los cheques son rectangulares horizontales).
   - **Tamaño minimo:** ancho mayor al 40% de la pagina, alto mayor al 8%.
7. Ordena los cheques de arriba hacia abajo segun su posicion Y.
8. Recorta cada cheque con un margen de 5 pixeles.

**Resultado:** 12 de 12 cheques detectados correctamente en los 4 PDFs de prueba.

**Nota:** Se implemento y probo un fallback por deteccion de color verde/turquesa (usando espacio HSV), pero se descarto porque el metodo de Canny ya detectaba el 100% de los cheques, mientras que el metodo por color solo detectaba 5 de 12 (muchos cheques no tienen fondo verde intenso).

### 4.3 Extraccion de monto (`monto_extractor.py`)

**Biblioteca utilizada:** docTR (python-doctr) con arquitectura db_resnet50 + crnn_vgg16_bn, accedida via abstraccion `OCRReader`

Este es el modulo mas complejo del sistema. Extrae el monto numerico escrito a mano en el recuadro de la esquina superior derecha de cada cheque.

#### Estrategia de extraccion (multi-fallback)

**Paso 1 - Anclaje en el signo $:**
- Se busca el simbolo `$` con OCR en la zona superior del cheque (40% superior, 60% derecho).
- Si se encuentra, se recorta una zona alrededor del `$` (30% de margen vertical, 10% de margen a la izquierda).
- Se ejecuta OCR sobre ese recorte con 3 variantes de preprocesamiento:
  - Imagen cruda (sin procesar)
  - Binarizacion Otsu
  - Escalado x2 + binarizacion Otsu

**Paso 2 - Zonas fijas:**
- Se prueban 3 recortes fijos de la esquina superior derecha con tamaño creciente:
  - Zona chica: 63% derecho, 25% superior
  - Zona media: 58% derecho, 32% superior
  - Zona grande: 50% derecho, 40% superior
- Cada zona se procesa con OCR crudo y con binarizacion Otsu.
- Si algun candidato alcanza un score de 5.0 o mas (formato con puntos de miles), se detiene la busqueda.

#### Seleccion de candidatos (scoring)

De todos los textos detectados por OCR, se filtran los que pueden ser montos:

1. **Formato con puntos de miles** (ej: `2.500.000`, `1350.000`): reciben score base de 5.0. Son la deteccion mas confiable.
2. **Numeros puros de 6+ digitos** (ej: `4000000`): reciben score base de 1.0, con penalizaciones:
   - 9+ digitos: -2.0 (probablemente basura concatenada)
   - 8 digitos: -0.8 (tipico numero de cheque)
   - 7 digitos sin `$` cerca: -0.3
3. **Monto pegado al `$`** en el mismo texto (ej: `$6.200.000-`): se extrae el numero.

Todos los candidatos reciben un bonus de +0.05 por cada digito. El candidato con mayor score se selecciona como resultado.

#### Limpieza de texto OCR

Antes de evaluar cada texto, se aplican estas limpiezas:
- Se eliminan guiones finales (`-`) que algunos emisores agregan despues del monto.
- Se eliminan comas finales.
- Se remueve el signo `$` o la letra `S` al inicio (el OCR frecuentemente confunde `$` con `S`).

#### Normalizacion a valor numerico

El monto detectado se convierte a `float` siguiendo las convenciones argentinas:
- Los puntos son separadores de miles: `2.500.000` se convierte a `2500000.0`
- La coma es separador decimal: `802.470,20` se convierte a `802470.20`
- Los numeros sin formato se toman tal cual: `4000000` se convierte a `4000000.0`

Retorna: `MontoOCRResult { monto, monto_raw, monto_score, zona_tokens }` donde `zona_tokens` son los tokens OCR de la zona superior derecha, reutilizados opcionalmente por el LLM.

### 4.4 Abstraccion de lectores OCR (`ocr_readers.py`)

**Proposito:** Desacoplar la logica de extraccion de campos de la libreria OCR concreta (docTR, Tesseract, EasyOCR, etc.).

Define una interfaz comun `OCRReader` que permite intercambiar implementaciones sin modificar los extractores de campos.

#### Interfaz base

```python
class OCRReader(ABC):
    def read(self, img: np.ndarray) -> list[OCRResult]:
        """Lee texto de una imagen.
        
        Returns:
            Lista de OCRResult con texto, confianza y posiciones normalizadas (0-1).
        """
        pass
```

#### OCRResult

Cada token detectado es un `OCRResult` con:
- `text`: Palabra detectada.
- `confidence`: Score de confianza (0.0-1.0).
- `cx`, `cy`: Centro normalizado del token en la imagen (0-1).
- `height`: Altura normalizada del token (0-1).

#### Implementaciones disponibles

| Clase | Libreria | Flag CLI | Uso |
|-------|---------|----------|-----|
| `DocTRReader` | python-doctr (`db_resnet50` + `crnn_vgg16_bn`) | default | Motor principal |
| `TrOCRReader` | transformers TrOCR (handwritten model) | `--trocr` | OCR alternativo para fecha (opcional) |
| `SuryaReader` | surya-ocr | `--surya` | Reemplazo completo de DocTRReader |
| `TesseractReader` | pytesseract | — | Disponible, no usado por defecto |
| `EasyOCRReader` | easyocr | — | Disponible, no usado por defecto |

### 4.5 Utilidades compartidas de fechas (`fecha_extractor.py`)

**Proposito:** Centralizar estructuras de datos y logica de parsing OCR compartida entre `FechaEmisionExtractor` y `FechaPagoExtractor`.

#### Dataclasses principales

**`Fecha`**: Componentes de una fecha con slots validados y raw:
- `dia`, `mes`, `anno`: Valores limpios confirmados por OCR (`"15"`, `"03"`, `"2026"`) o `None`.
- `dia_raw`, `mes_raw`, `anno_raw`: Texto OCR original para que el LLM pueda razonar sobre componentes no reconocidos.
- `to_iso()`: Retorna `"YYYY-MM-DD"` si los tres componentes son conocidos.
- `any_known()`, `all_known()`: Chequeos de completitud.

**`FechaResult`**: Resultado de un extractor de fecha:
- `fecha_iso`: Fecha en formato ISO si se pudo parsear directamente, `None` si no.
- `tokens`: Lista de `OCRResult` de la zona relevante, enviados al LLM cuando `fecha_iso` es `None`.
- `partial`: `Fecha` con los componentes que el OCR si reconocio (para guiar al LLM).

#### Funciones de parsing estructural

- **`_filtrar_tokens_fecha_estructura(tokens)`**: Busca la estructura `DIA DE MES DE ANNO` en una lista de tokens OCR. Retorna `(combined, source_tokens, partial_fecha)`. Si la fecha esta completa y valida, `combined` tiene un unico token con la fecha formateada.
- **`_fecha_completa_a_iso(text)`**: Convierte `"15 DE Marzo DE 2026"` a `"2026-03-15"`, o `None` si el formato es invalido.
- **`_expandir_tokens_de(tokens)`**: Separa tokens fusionados con "DE" embebido (ej: `"16DE"` → `["16", "DE"]`).
- **`_agrupar_de_clusters(de_tokens)`**: Agrupa tokens "DE" por proximidad vertical (umbral 0.06) para identificar lineas de fecha.

#### Expresiones regulares clave

- `_DE_RE`: Acepta tokens "DE" con posibles caracteres no alfanumericos alrededor.
- `_EL_INICIO_RE`: Detecta "EL" incluso fusionado con el siguiente token (ej: `"EL19DE"`, `"ELZo"`).
- `_BOILERPLATE_RE`: Identifica tokens del texto legal `"360 dias"` para excluirlos.
- `_VENTANA_CY`: Umbral `0.07` para agrupar tokens en la misma fila normalizada.

### 4.6 Extraccion de fecha de emision (`fecha_emision_extractor.py`)

**Proposito:** Extraer la fecha de emision del cheque (formato "CIUDAD, DD DE MES DE AAAA").

Retorna un `FechaResult` con la fecha ISO si se pudo parsear directamente, o los tokens relevantes para que el LLM infiera la fecha.

#### Estrategia (en orden de prioridad)

**Scan amplio inicial:** Se lee con OCR la zona `y: 0-55%, x: 10-80%` del cheque.

**1. Ancla ciudad-coma:**
- Busca un token que sea un nombre de ciudad terminado en coma (ej: `FEDERAL,`, `CUATIA,`, `QUILMES,`), con `cy > 0.30`.
- Cuando se encuentra: `cy_emision` = `cy` del token, ventana = tokens a la derecha del token ciudad con `cy` similar.
- Es el ancla mas confiable porque la ciudad siempre precede a la fecha en la misma linea.

**2. Cluster DE / ancla EL:**
- Si no hay ciudad-coma, busca pares de tokens "DE" en la misma fila (`cy` similar, umbral 0.06).
- Descarta clusters cercanos a boilerplate (`360`, `dias`) o en la misma linea que el token "EL" (linea de pago).
- Si no hay cluster DE valido, estima `cy_emision` como `1.5 × altura del token` por encima del token "EL".

**3. Fallback:**
- Toma todos los tokens del scan por encima del 40% del cheque (o del token "plazo" del texto legal, si aparece antes).
- Dentro de esa zona, si hay tokens de mes o año conocidos, acota la ventana a su fila.

#### Parseo directo vs. paso al LLM

Una vez obtenida la ventana de tokens, se intenta parsear la estructura `DIA DE MES DE ANNO` directamente:
- Si la fecha esta completa y valida (dia 1-31, mes conocido, año 2020-2030), se retorna `FechaResult(fecha_iso=..., tokens=...)`.
- Si falta algun componente: se retorna `FechaResult(fecha_iso=None, tokens=source_tokens, partial=Fecha(...))` para que el LLM infiera los componentes faltantes.

### 4.7 Extraccion de fecha de pago (`fecha_pago_extractor.py`)

**Proposito:** Extraer la fecha de pago del cheque (linea que comienza con "EL DD DE MES DE AAAA").

Complementa a `FechaEmisionExtractor`: mientras ese busca la linea de emision (mas arriba), este busca la linea de pago (mas abajo).

#### Estrategia (en orden de prioridad)

**Scan amplio inicial:** Se lee con OCR la zona `y: 0-55%, x: 10-70%` del cheque (zona izquierda, donde suele estar la fecha de pago).

**1. Ancla EL:**
- Busca tokens que comiencen con "EL" (incluso fusionado: `"EL19DE"`) con `cy > 0.35`.
- Toma el token "EL" mas bajo (mayor `cy`) para evitar confundirlo con la linea de emision.
- La ventana son los tokens en la misma fila (`±_VENTANA_CY`).

**2. Ancla PAGUESE:**
- Si no hay token "EL", busca el token `"Paguese"/"PAGUESE"` con `cy > 0.35`.
- La fecha de pago esta una linea **por encima** de ese token: `cy_pago = cy_paguese - 1.5 × altura_token`.

**3. Cluster DE inferior:**
- Busca el cluster de tokens "DE" mas bajo (mayor `cy`) con al menos 2 tokens, descartando boilerplate.
- Es el complemento del cluster que `FechaEmisionExtractor` descarto por estar en la linea de pago.

**4. Fallback por keywords:**
- Agrupa tokens de mes/año conocidos en bandas por `cy`.
- Toma la banda mas baja (la fecha de pago es siempre mas abajo que la de emision).

Al igual que `FechaEmisionExtractor`, intenta parsear directamente y, si no puede, retorna `FechaResult(fecha_iso=None, tokens=..., partial=Fecha(...))` para el LLM.

### 4.8 Backends de LLM (`llm_backends.py`)

**Proposito:** Proporcionar una interfaz comun para diferentes backends de LLM.

#### Interfaz base

```python
class LLMBackend(ABC):
    def chat(self, messages: list[dict]) -> str | None:
        """Envia mensajes al LLM y retorna la respuesta, o None si falla."""
        pass
```

#### Implementacion disponible

**OllamaBackend**: Conecta a un servidor Ollama local (`http://localhost:11434`) via POST `/api/chat`.
- Parametros: `model` (default: `"llama3.2"`), `base_url`, `timeout` (180s).
- Retorna la respuesta del LLM o `None` si hay timeout/error.

### 4.9 Validacion y extraccion con LLM (`llm_validator.py`)

**Proposito:** Usar un LLM para validar, corregir y extraer campos estructurados a partir de tokens OCR.

Este modulo es **opcional** y mejora la precision cuando el OCR estructural no puede parsear una fecha completa.

#### Metodos principales

**`extract_fields(ocr_tokens, batch_context)`**: Extrae monto y fecha_emision en un unico llamado al LLM. Usado cuando el monto no fue detectado por la heuristica OCR.

**`infer_fecha(fecha_tokens, today_max, max_future_days, partial_fecha)`**: Especializado en inferir fechas a partir de tokens crudos con escritura imperfecta. Caracteristicas:
- Acepta un `partial_fecha` (`Fecha`) con los componentes que el OCR ya confirmo; el LLM solo debe inferir los que faltan.
- `today_max`: fecha maxima para `fecha_emision` (no puede ser futura).
- `max_future_days`: dias maximos en el futuro para `fecha_pago` (tipicamente 365).
- Incluye reglas de correccion OCR en el prompt: `Z↔2`, `o/O↔0`, `l↔1`, `S↔5`, confusiones de "I" con "1" en el dia, etc.
- El LLM responde solo con `YYYY-MM-DD` o `null`.
- Retorna `LLMExtractionResult` con confianza 0.88 si la fecha es valida.

#### Sistema de prompts para fechas

El LLM recibe:
- **Prompt del sistema:** Instrucciones sobre como ser un experto en fechas de cheques argentinos.
- **Tokens OCR ordenados por posicion:** texto de los tokens de la linea de fecha.
- **Restriccion de fecha:** `today_max` o `max_future_days` segun el campo.
- **Hint parcial:** Si `partial_fecha` tiene componentes confirmados, el prompt indica cuales usar exactamente y cuales inferir.

#### Overrides OCR sobre resultado LLM

En `ChequeExtractor`, despues de recibir la fecha del LLM, se aplica `_apply_fecha_overrides`: los componentes que el OCR confirmo con certeza (dia, mes, o año) sobreescriben los del LLM para evitar drift del modelo.

#### Estado de usar/no usar LLM

`LLMValidator` es **opcional**. Si no se proporciona al `ChequeExtractor`:
- Se usa solo el parseo directo OCR para las fechas.
- `fecha_emision` y `fecha_pago` quedan en `None` si no se pudo parsear completamente.
- `monto` usa solo la heuristica OCR.

Si se proporciona (`--con-llm`), se invoca cuando:
- `fecha_emision` es `None` despues del parseo OCR.
- `fecha_pago` es `None` despues del parseo OCR.
- `monto` es `None` despues de la heuristica.

### 4.10 Orquestador principal (`cheque_extractor.py`)

**Proposito:** Coordinar `MontoExtractor`, `FechaEmisionExtractor`, `FechaPagoExtractor` y `LLMValidator` para producir un `DatosCheque` completo.

#### Flujo de `ChequeExtractor.extraer()`

1. **OCR secuencial:** Extrae monto, fecha_emision y fecha_pago via OCR heuristico.
2. **LLM paralelo (opcional):** Si `LLMValidator` esta disponible, lanza hasta 3 llamadas LLM en paralelo con `ThreadPoolExecutor(max_workers=3)`:
   - `infer_fecha` para `fecha_emision` (solo si `fecha_iso` es `None`).
   - `infer_fecha` para `fecha_pago` (solo si `fecha_iso` es `None`).
   - `extract_fields` para `monto` (solo si `monto` es `None`).
3. **Overrides OCR:** Aplica `_apply_fecha_overrides` sobre el resultado LLM para fijar componentes que el OCR confirmo.
4. **Retorna:** `DatosCheque` con todos los campos + scores de confianza.

El paralelismo reduce la latencia total cuando se usan los tres campos con LLM.

### 4.11 Modelo de datos y persistencia (`models.py`)

Se define un `dataclass` llamado `DatosCheque` con los siguientes campos:

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `monto` | `float` o `None` | Monto normalizado como valor numerico (OCR heuristico o LLM) |
| `monto_raw` | `str` | Texto crudo tal como lo leyo el OCR/LLM |
| `monto_score` | `float` | Puntaje de confianza de la deteccion heuristica |
| `monto_llm_confidence` | `float` o `None` | Score de confianza del LLM (0.0-1.0) si disponible |
| `fecha_emision` | `str` o `None` | Fecha de emision en formato ISO (`YYYY-MM-DD`) |
| `fecha_emision_raw` | `str` o `None` | Mismo valor que `fecha_emision` (ISO directo) |
| `fecha_emision_llm_confidence` | `float` o `None` | Score de confianza del LLM para la fecha de emision |
| `fecha_pago` | `str` o `None` | Fecha de pago en formato ISO (`YYYY-MM-DD`) |
| `fecha_pago_raw` | `str` o `None` | Mismo valor que `fecha_pago` (ISO directo) |
| `fecha_pago_llm_confidence` | `float` o `None` | Score de confianza del LLM para la fecha de pago |
| `imagen_path` | `str` | Ruta a la imagen recortada del cheque |
| `pdf_origen` | `str` | Nombre del archivo PDF de origen |
| `pagina` | `int` | Numero de pagina dentro del PDF |
| `indice_en_pagina` | `int` | Indice del cheque dentro de la pagina |

Los datos se persisten en `cheques.json` como un **array de runs**, donde cada run corresponde a una ejecucion del programa:
```json
[
  {
    "fecha_proceso": "2026-04-24T10:00:00",
    "nombre_archivo": "Scan CH1.pdf",
    "cheques": [
      {
        "monto": 2500000.0,
        "monto_raw": "2.500.000",
        "monto_score": 5.35,
        "monto_llm_confidence": null,
        "fecha_emision": "2026-02-15",
        "fecha_emision_raw": "2026-02-15",
        "fecha_emision_llm_confidence": 0.88,
        "fecha_pago": "2026-08-15",
        "fecha_pago_raw": "2026-08-15",
        "fecha_pago_llm_confidence": 0.88,
        "imagen_path": "output\\images\\Scan CH1_p1_ch1.png",
        "pdf_origen": "Scan CH1.pdf",
        "pagina": 1,
        "indice_en_pagina": 1
      }
    ]
  }
]
```

Cada nueva ejecucion **agrega** su run al array existente (no sobreescribe).

### 4.12 Interfaz de linea de comandos (`main.py`)

El programa ofrece tres comandos:

```bash
# Procesar un PDF
python main.py procesar "ruta/al/archivo.pdf"

# Procesar todos los PDFs de un directorio
python main.py procesar "ruta/al/directorio/"

# Activar LLM de texto para campos no parseados
python main.py procesar "ruta/al/archivo.pdf" --con-llm

# Usar TrOCR para la zona de fecha
python main.py procesar "ruta/al/archivo.pdf" --trocr

# Usar Surya como motor OCR principal
python main.py procesar "ruta/al/archivo.pdf" --surya

# Listar cheques procesados
python main.py listar

# Buscar en los cheques procesados
python main.py buscar "termino"
```

**Flags de `procesar`:**

| Flag | Efecto |
|------|--------|
| `--con-llm` | Activa LLM de texto (Ollama) para inferir fechas y monto cuando OCR falla |
| `--llm-model` | Modelo Ollama para LLM de texto (default: `llama3.2`) |
| `--llm-url` | URL del servidor Ollama (default: `http://localhost:11434`) |
| `--trocr` | Usa TrOCR (handwritten) como OCR alternativo |
| `--surya` | Usa Surya como motor OCR principal en lugar de docTR |
| `--debug` | Activa logging DEBUG y guarda imagenes intermedias en `output/debug/` |
| `--vision-llm` | Flag existente en el CLI pero actualmente sin efecto (implementacion deshabilitada) |

## 5. Dependencias

| Biblioteca | Version | Uso |
|------------|---------|-----|
| PyMuPDF | >= 1.24.0 | Conversion PDF a imagen |
| opencv-python | >= 4.8.0 | Deteccion de cheques (Canny, contornos) |
| python-doctr | >= 1.0.1 | OCR principal para monto manuscrito |
| torch | >= 2.6.0 | Backend de docTR |
| Pillow | >= 11.0.0 | Manipulacion de imagenes |
| numpy | >= 1.26.0 | Operaciones con arrays de imagenes |
| tqdm | >= 4.66.0 | Barra de progreso |
| pytesseract | >= 0.3.10 | OCR alternativa (opcional) |
| httpx | >= 0.27.0 | Cliente HTTP para backends LLM (Ollama) |

Todo el procesamiento es **local**. Los modelos de docTR se descargan una sola vez desde HuggingFace al primer uso y quedan cacheados en disco. LLM es opcional y requiere un servidor local (ej. Ollama).

## 6. Resultados actuales

Sobre los 12 cheques de prueba (4 PDFs, 3 cheques por pagina, 6 bancos distintos):

**Extraccion de monto (OCR heuristico):**
- **Deteccion de cheques:** 12/12 (100%)
- **Extraccion de monto exacto:** 8/12 (67%)
- **Extraccion de monto parcial** (orden de magnitud correcto): 10/12 (83%)
- Los 8 aciertos exactos tienen un score de confianza >= 1.3, lo cual permite al usuario filtrar resultados confiables.

**Extraccion de fechas (con LLM):**
- Extraccion exitosa de `fecha_emision` y `fecha_pago` con Ollama + llama3.2 en todos los cheques validados.
- El OCR estructural puede parsear directamente las fechas cuando la escritura es clara.
- Cuando el OCR no puede parsear completamente, el LLM recibe los componentes parciales confirmados y solo infiere los faltantes.

## 7. Limitaciones conocidas

**OCR heuristico (monto):**
1. **Caligrafia muy irregular:** Cuando el monto esta escrito con letra muy desprolija o hay firmas superpuestas, el OCR puede fallar.
2. **Montos sin formato de puntos de miles:** Montos chicos como `$4.215` que no llevan puntos de miles son mas dificiles de distinguir de otros numeros en el cheque.
3. **Decimales separados:** Cuando el OCR detecta los centavos como un texto separado del monto entero, no siempre se concatenan correctamente (ej: `802.470` sin los `,20`).

**OCR estructural (fechas):**
4. **Tokens fusionados:** El OCR a veces funde tokens adyacentes (ej: `"16DE"`, `"EL19DE"`). La funcion `_expandir_tokens_de` maneja los casos mas comunes, pero combinaciones inusuales pueden fallar.
5. **Anclas no detectadas:** Si el scan amplio no detecta ninguna de las anclas (ciudad-coma, DE-cluster, EL), el fallback devuelve tokens genericos que el LLM debe interpretar.

**LLM (fecha, monto):**
6. **Dependencia externa:** El LLM requiere un servidor Ollama corriendo localmente. Sin LLM, las fechas quedan en `None` si el parseo directo no funciona.
7. **Latencia:** Usar LLM agrega tiempo: ~10-30 segundos por cheque en CPU segun el modelo. Con paralelizacion de las 3 llamadas, el tiempo total es el de la llamada mas lenta.
8. **Modelos variables:** La calidad depende del modelo LLM utilizado. Modelos mas pequenos pueden tener precision baja en fechas con escritura muy deteriorada.

**General:**
9. **Velocidad sin GPU:** El procesamiento OCR toma aproximadamente 4-6 segundos por cheque en CPU (sin GPU).
