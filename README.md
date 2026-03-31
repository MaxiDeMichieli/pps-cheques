# PPS Cheques - Procesador de Cheques Escaneados

Herramienta CLI para procesar cheques escaneados en archivos PDF, extrayendo automáticamente los montos mediante OCR (docTR).

## Requisitos

- Python 3.10 o superior
- Windows, macOS o Linux

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

### Procesar un PDF con cheques

```bash
python main.py procesar ruta/al/archivo.pdf
```

### Procesar todos los PDFs en un directorio

```bash
python main.py procesar ruta/al/directorio/
```

### Listar cheques procesados

```bash
python main.py listar
```

### Buscar en cheques procesados

```bash
python main.py buscar "termino"
```

### Opciones adicionales

- `-o, --output`: Directorio de salida (default: `output`)

```bash
python main.py -o resultados procesar cheques/
```

## Estructura de salida

```
output/
├── images/           # Imágenes de cheques individuales
│   ├── archivo_p1_ch1.png
│   └── ...
└── cheques.json      # Datos extraídos en formato JSON
```

## Estructura del proyecto

```
pps-cheques/
├── main.py              # CLI principal
├── requirements.txt     # Dependencias Python
├── src/
│   ├── check_detector.py    # Detección de cheques en páginas
│   ├── monto_extractor.py   # Extracción de montos via OCR
│   ├── pdf_processor.py     # Conversión PDF a imágenes
│   └── models.py            # Modelos de datos
├── cheques/             # Directorio para PDFs de entrada
├── docs/                # Documentación adicional
└── .venv/               # Entorno virtual (no incluido en git)
```

## Licencia

MIT
