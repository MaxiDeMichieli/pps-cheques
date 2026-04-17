# PPS Cheques - Procesador de Cheques Escaneados

Herramienta CLI para procesar cheques escaneados en archivos PDF, extrayendo automáticamente campos estructurados (montos y fechas de emisión) mediante una arquitectura híbrida de OCR local (docTR) + validación con LLM (Ollama).

## Requisitos

- Python 3.10 o superior
- Windows, macOS o Linux
- **Ollama** instalado y corriendo localmente (requerido para validación con LLM)
  - Por defecto escucha en `http://localhost:11434`
  - Modelo recomendado: `llama3.2` (descargarse con `ollama pull llama3.2`)

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repo-url>
cd pps-cheques
```

### 2. Crear entorno virtual

**Windows (PowerShell):**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** La primera ejecución descargará los modelos de OCR (~100MB), lo cual puede tomar unos minutos.

## Uso

### Procesar un PDF con cheques (con validación LLM)

```bash
python main.py procesar ruta/al/archivo.pdf
```

### Procesar todos los PDFs en un directorio

```bash
python main.py procesar ruta/al/directorio/
```

### Procesar solo con OCR (sin validación LLM)

```bash
python main.py procesar ruta/al/archivo.pdf --sin-llm
```

### Listar cheques procesados

```bash
python main.py listar
```

### Buscar en cheques procesados

```bash
python main.py buscar "termino"
```

### Opciones avanzadas

| Flag | Default | Descripción |
|---|---|---|
| `-o, --output` | `output` | Directorio de salida |
| `--sin-llm` | — | Desactivar validación LLM, usar solo OCR heurístico |
| `--llm-model` | `llama3.2` | Modelo Ollama a usar |
| `--llm-url` | `http://localhost:11434` | URL del servidor Ollama |
| `--debug` | — | Activar logging DEBUG y guardar imágenes intermedias en `output/debug/` |

**Ejemplos:**

```bash
# Directorio de salida personalizado
python main.py -o resultados procesar cheques/

# Solo OCR, sin LLM
python main.py procesar cheques/ --sin-llm

# Modo debug: ver qué está recortando cada extractor
python main.py procesar cheques/MiCheque.pdf --sin-llm --debug
```

## Reutilización de imágenes

Si el directorio `output/images/` ya contiene las imágenes recortadas de un PDF (archivos con el patrón `{nombre}_p{pag}_ch{idx}.png`), el paso de conversión PDF → detección de cheques se omite por completo y se reutilizan esas imágenes directamente. Solo se re-ejecuta la extracción OCR/LLM.

Esto acelera significativamente el ciclo de prueba cuando se están ajustando parámetros de extracción. Para forzar un reprocesamiento completo, borrar las imágenes correspondientes de `output/images/`.

## Modo debug (`--debug`)

Al pasar `--debug`, se crea un subdirectorio por cheque dentro de `output/debug/`:

```
output/debug/
└── MiCheque_p1_ch1/
    ├── monto_zona_sup.png          # Zona superior-derecha escaneada en busca de $
    ├── monto_dollar_crop_raw.png   # Recorte ajustado alrededor del $ (imagen cruda)
    ├── monto_dollar_crop_otsu.png  # Mismo recorte con umbral Otsu
    ├── monto_dollar_crop_x2otsu.png# Mismo recorte 2× escalado + Otsu
    ├── monto_fixed_z1.png          # Zona fija 1 de fallback para el monto
    ├── monto_fixed_z2.png          # Zona fija 2 de fallback para el monto
    ├── monto_fixed_z3.png          # Zona fija 3 de fallback para el monto
    └── fecha_zona.png              # Zona de extracción de la fecha de emisión
```

Estos archivos permiten verificar visualmente que cada extractor está recortando el área correcta del cheque. Además, el nivel de logging del proceso se eleva a `DEBUG`, mostrando en consola los tokens OCR detectados en cada zona.

## Salida JSON

Los resultados se acumulan en `output/cheques.json` en cada ejecución (no se sobreescribe). El archivo es un array donde cada entrada corresponde a un PDF procesado en una corrida:

```json
[
  {
    "fecha_proceso": "2026-04-17T10:30:00",
    "nombre_archivo": "Scan CH1.pdf",
    "cheques": [
      {
        "monto": 802470.20,
        "monto_raw": "802.470,20",
        "monto_score": 5.15,
        "monto_llm_confidence": 0.95,
        "fecha_emision": "2026-12-12",
        "fecha_emision_raw": "12 DE DICIEMBRE DE 2026",
        "fecha_emision_llm_confidence": 0.98,
        "imagen_path": "output/images/Scan CH1_p1_ch1.png",
        "pdf_origen": "Scan CH1.pdf",
        "pagina": 1,
        "indice_en_pagina": 1
      }
    ]
  }
]
```

Múltiples ejecuciones del mismo PDF generan entradas separadas, lo que permite comparar resultados entre diferentes configuraciones o versiones del modelo.

## Arquitectura

El procesamiento sigue un pipeline de 3 pasos:

```text
PDF → [pdf_processor] → Imágenes de páginas
        ↓
[check_detector] → Imágenes de cheques individuales
        ↓
[ChequeExtractor] → Datos struturados (DatosCheque)
  ├── MontoExtractor → OCR + heurísticas → monto
  ├── FechaEmisionExtractor → OCR → zona de fecha
  └── LLMValidator → Reformateo + validación de confianza (opcional)
```

Para más detalles, consultar [docs/architecture.md](docs/architecture.md).

## Estructura del proyecto

```
pps-cheques/
├── main.py                          # CLI principal
├── requirements.txt                 # Dependencias Python
├── src/
│   ├── __init__.py
│   ├── models.py                    # Modelos de datos (DatosCheque)
│   ├── detection/
│   │   └── check_detector.py        # Detección de cheques en páginas (OpenCV)
│   ├── extractors/
│   │   ├── cheque_extractor.py      # Orquestador de extracción
│   │   ├── monto_extractor.py       # Extracción de montos via OCR
│   │   └── fecha_emision_extractor.py # Extracción de fechas via OCR
│   ├── ocr/
│   │   └── ocr_readers.py           # Interfaz abstracta para OCR (docTR)
│   ├── llm/
│   │   ├── llm_validator.py         # Validación y reformateo con LLM
│   │   └── llm_backends.py          # Backends HTTP para LLMs (Ollama)
│   └── pdf/
│       └── pdf_processor.py         # Conversión PDF → imágenes (PyMuPDF)
├── cheques/                         # Directorio para PDFs de entrada
├── docs/                            # Documentación adicional
│   ├── architecture.md              # Detalles de la arquitectura
│   ├── implementacion.md            # Notas de implementación
│   └── pruebas_modelos.md           # Análisis de modelos OCR
├── output/
│   ├── images/                      # Imágenes de cheques recortados
│   │   ├── archivo_p1_ch1.png
│   │   └── ...
│   ├── debug/                       # Imágenes intermedias (solo con --debug)
│   │   └── archivo_p1_ch1/
│   │       ├── monto_zona_sup.png
│   │       └── ...
│   └── cheques.json                 # Resultados acumulados (array de corridas)
├── notas.md                         # Notas del desarrollo
└── .venv/                           # Entorno virtual (no incluido en git)
```

## Licencia

MIT
