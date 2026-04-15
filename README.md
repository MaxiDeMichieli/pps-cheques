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

- `-o, --output`: Directorio de salida (default: `output`)
- `--sin-llm`: Desactivar validación LLM, usar solo OCR heurístico
- `--llm-model`: Modelo Ollama a usar (default: `llama3.2`)
- `--llm-url`: URL del servidor Ollama (default: `http://localhost:11434`)

**Ejemplos:**

```bash
# Procesar con salida personalizada
python main.py -o resultados procesar cheques/
Arquitectura

El procesamiento sigue un pipeline de 3 pasos:

```
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
│   ├── images/           # Imágenes de cheques individuales
│   │   ├── archivo_p1_ch1.png
│   │   └── ...
│   └── cheques.json      # Datos extraídos en formato JSON
├── notas.md                         # Notas del desarrollo
└── .venv/                           # Entorno virtual (no incluido en git)
```

## Licencia

MIT
