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
│   │   └── fecha_emision_extractor.py # Extracción de fecha usando OCRReader
│   ├── ocr/
│   │   └── ocr_readers.py          # Abstracción de lectores OCR (docTR, Tesseract, etc.)
│   ├── llm/
│   │   ├── llm_backends.py         # Backends HTTP para LLMs (Ollama, etc.)
│   │   └── llm_validator.py        # Validación de campos con LLM
│   └── pdf/
│       └── pdf_processor.py       # Conversión PDF -> imágenes
├── output/     
│   ├── cheques.json                # Datos extraidos
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
      +---> [4b] Extraccion de fecha (busqueda de tokens)
      |
      v
[5] Validacion/refinamiento con LLM (opcional)
      |
      v
[6] Normalizacion y guardado en JSON
```

**Nota:** Los pasos [4a], [4b] y [5] son coordinados por `ChequeExtractor`.

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

### 4.4 Abstraccion de lectores OCR (`ocr_readers.py`)

**Proposito:** Desacoplar la logica de extraccion de campos de la libreria OCR concreta (docTR, Tesseract, EasyOCR, etc.).

Define una interfaz comun `OCRReader` que permite intercambiar implementaciones sin modificar `MontoExtractor` o `FechaEmisionExtractor`.

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

#### Implementaciones disponibles

1. **DocTRReader**: Usa docTR con arquitectura `db_resnet50 + crnn_vgg16_bn`. Implementacion por defecto, mas robusta.
2. **TesseractReader**: Click alternativa con Tesseract (requiere binarios del sistema).
3. **EasyOCRReader**: Opcion alternativa con EasyOCR.

### 4.5 Extraccion de fecha de emision (`fecha_emision_extractor.py`)

**Proposito:** Extraer la linea de fecha "CIUDAD, DD DE MES DE AAAA" del cheque.

Utiliza un enfoque basado en tokens OCR mas que en procesamiento de imagen:

1. **Escaneo de zona:** Lee tokens OCR de la franja superior-central del cheque (40-45% de alto), excluyendo el logo del banco (izq) y recuadro del monto (der).
2. **Anclaje en "EL":** Busca el token "EL" (que siempre aparece en la linea siguente de la fecha: "EL DD DE MES DE AAAA" para pago). 
3. **Filtrado:** Retorna solo los tokens entre el 30% inferior del header y la linea del "EL", que son los que constituyen la fecha de emision.

El resultado es una lista de `OCRResult` que luego se envia al LLM para normalizacion a formato ISO.

### 4.6 Backends de LLM (`llm_backends.py`)

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

### 4.7 Validacion y extraccion con LLM (`llm_validator.py`)

**Proposito:** Usar un LLM para validar, corregir y extraer campos estructurados a partir de tokens OCR.

Este modulo es **opcional** y mejora la precision cuando se consume bastante tiempo extra.

#### Campos extraidos

1. **monto**: Importe numerico en formato argentino (`"4.000.000"`, `"802.470,20"`).
2. **fecha_emision**: Fecha en formato ISO (`"YYYY-MM-DD"`).

#### Sistema de prompts

El LLM recibe:
- **Prompt del sistema:** Instrucciones detalladas sobre como leer cheques argentinos, formatos, convenciones de confianza.
- **Tokens OCR del cheque:** Texto de todos los tokens (monto + fecha) en orden de lectura.
- **Contexto del lote:** Opcional - montos raw de otros cheques del mismo lote, para ayudar a desambiguar anomalias.

#### Scoring de confianza

El LLM asigna un score de confianza (0.0-1.0) para cada campo:
- **0.95-1.00:** Valor inequivoco, formato estandar reconocible.
- **0.80-0.94:** Legible con algo de ruido OCR, reconstruccion segura.
- **0.60-0.79:** Parcialmente reconstruido con contexto del lote.
- **0.00-0.59:** El LLM esta adivinando, resultado no confiable.

#### Estado de usar/no usar LLM

`LLMValidator` es **opcional**. Si no se proporciona al `ChequeExtractor`, se usa solo la heuristica de OCR.

Si si se proporciona y el LLM devuelve una confianza >= 0.70, se usa su resultado. Sino, se preserva el resultado del OCR heuristico.

### 4.8 Orquestador principal (`cheque_extractor.py`)

**Proposito:** Coordinar `MontoExtractor`, `FechaEmisionExtractor` y `LLMValidator` para producir un `DatosCheque` completo.

#### Flujo de `ChequeExtractor.extraer()`

1. **OCR heuristico**: Extrae monto (heuristica + OCR) y tokens de fecha.
2. **LLM opcional**: Si `LLMValidator` esta disponible, envia todos los tokens al LLM para refinamiento.
3. **Decision de valores finales**:
   - Si LLM confiance >= 0.70, usa resultado del LLM.
   - Sino, preserva resultado del OCR heuristico.
4. **Retorna**: `DatosCheque` con todos los campos + scores de confianza.

Esto permite un flujo hibrido: OCR rapido para velocidad, LLM opcional para precision.

### 4.9 Modelo de datos y persistencia (`models.py`)

Se define un `dataclass` llamado `DatosCheque` con los siguientes campos:

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `monto` | `float` o `None` | Monto normalizado como valor numerico (OCR heuristico o LLM) |
| `monto_raw` | `str` | Texto crudo tal como lo leyo el OCR/LLM |
| `monto_score` | `float` | Puntaje de confianza de la deteccion heuristica |
| `monto_llm_confidence` | `float` o `None` | Score de confianza del LLM (0.0-1.0) si disponible |
| `fecha_emision` | `str` o `None` | Fecha de emision en formato ISO (`YYYY-MM-DD`) si se extrajo |
| `fecha_emision_raw` | `str` o `None` | Texto crudo de la fecha segun el LLM |
| `fecha_emision_llm_confidence` | `float` o `None` | Score de confianza del LLM para la fecha |
| `imagen_path` | `str` | Ruta a la imagen recortada del cheque |
| `pdf_origen` | `str` | Nombre del archivo PDF de origen |
| `pagina` | `int` | Numero de pagina dentro del PDF |
| `indice_en_pagina` | `int` | Indice del cheque dentro de la pagina |

Los datos se persisten en un archivo JSON con la estructura:
```json
{
  "total_cheques": 12,
  "cheques": [
    {
      "monto": 2500000.0,
      "monto_raw": "2.500.000",
      "monto_score": 5.35,
      "monto_llm_confidence": 0.98,
      "fecha_emision": "2026-02-15",
      "fecha_emision_raw": "15 DE Febrero DE 2026",
      "fecha_emision_llm_confidence": 0.95,
      "imagen_path": "output\\images\\Scan CH1_p1_ch1.png",
      "pdf_origen": "Scan CH1.pdf",
      "pagina": 1,
      "indice_en_pagina": 1
    }
  ]
}
```

### 4.10 Interfaz de linea de comandos (`main.py`)

El programa ofrece tres comandos:

```bash
# Procesar un PDF
python main.py procesar "ruta/al/archivo.pdf"

# Procesar todos los PDFs de un directorio
python main.py procesar "ruta/al/directorio/"

# Listar cheques procesados
python main.py listar

# Buscar en los cheques procesados
python main.py buscar "termino"
```

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

**Extraccion de fecha de emision (con LLM):**
- Extraccion exitosa con Ollama + llama3.2 en todos los cheques validados.
- Scores de confianza tipicos: 0.85-0.99 (muy alta precision).

## 7. Limitaciones conocidas

**OCR heuristico (monto):**
1. **Caligrafia muy irregular:** Cuando el monto esta escrito con letra muy desprolija o hay firmas superpuestas, el OCR puede fallar.
2. **Montos sin formato de puntos de miles:** Montos chicos como `$4.215` que no llevan puntos de miles son mas dificiles de distinguir de otros numeros en el cheque.
3. **Decimales separados:** Cuando el OCR detecta los centavos como un texto separado del monto entero, no siempre se concatenan correctamente (ej: `802.470` sin los `,20`).

**LLM (fecha, validacion):**
4. **Dependencia externa:** La extraccion de fecha requiere un servidor LLM externo (ej. Ollama). Sin LLM, solo se extraen montos.
5. **Latencia:** Usar LLM agrega muchisimo tiempo: ~20-30 segundos por cheque en CPU (vs ~4-6 seg con OCR puro). Se recomienda usar LLM solo cuando la precision sea critica.
6. **Modelos variables:** La calidad depends del modelo LLM utilizado. Modelos mas pequenos pueden tener precision baja.

**General:**
7. **Velocidad sin GPU:** El procesamiento OCR toma aproximadamente 4-6 segundos por cheque en CPU (sin GPU).
