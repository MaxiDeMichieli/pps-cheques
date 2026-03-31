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
│   ├── pdf_processor.py       # Conversion PDF -> imagenes
│   ├── check_detector.py      # Deteccion y recorte de cheques
│   ├── monto_extractor.py     # Extraccion del monto con docTR
│   └── models.py              # Modelo de datos y persistencia JSON
├── output/
│   ├── cheques.json           # Datos extraidos
│   └── images/                # Imagenes recortadas de cada cheque
└── docs/
    ├── implementacion.md      # Este documento
    └── pruebas_modelos.md     # Pruebas de modelos OCR
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
[4] Extraccion del monto numerico (docTR)
      |
      v
[5] Normalizacion y guardado en JSON
```

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

**Biblioteca utilizada:** docTR (python-doctr) con arquitectura db_resnet50 + crnn_vgg16_bn

Este es el modulo mas complejo del sistema. Extrae el monto numerico escrito a mano en el recuadro de la esquina superior derecha de cada cheque.

#### Estrategia de extraccion (multi-fallback)

**Paso 1 - Anclaje en el signo $:**
- Se busca el simbolo `$` con docTR en la zona superior del cheque (40% superior, 60% derecho).
- Si se encuentra, se recorta una zona alrededor del `$` (30% de margen vertical, 10% de margen a la izquierda).
- Se ejecuta docTR sobre ese recorte con 3 variantes de preprocesamiento:
  - Imagen cruda (sin procesar)
  - Binarizacion Otsu
  - Escalado x2 + binarizacion Otsu

**Paso 2 - Zonas fijas:**
- Se prueban 3 recortes fijos de la esquina superior derecha con tamaño creciente:
  - Zona chica: 63% derecho, 25% superior
  - Zona media: 58% derecho, 32% superior
  - Zona grande: 50% derecho, 40% superior
- Cada zona se procesa con docTR crudo y con binarizacion Otsu.
- Si algun candidato alcanza un score de 5.0 o mas (formato con puntos de miles), se detiene la busqueda.

#### Seleccion de candidatos (scoring)

De todos los textos detectados por docTR, se filtran los que pueden ser montos:

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

### 4.4 Modelo de datos y persistencia (`models.py`)

Se define un `dataclass` llamado `DatosCheque` con los siguientes campos:

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `monto` | `float` o `None` | Monto normalizado como valor numerico |
| `monto_raw` | `str` | Texto crudo tal como lo leyo el OCR |
| `monto_score` | `float` | Puntaje de confianza de la deteccion |
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
      "imagen_path": "output\\images\\Scan CH1_p1_ch1.png",
      "pdf_origen": "Scan CH1.pdf",
      "pagina": 1,
      "indice_en_pagina": 1
    }
  ]
}
```

### 4.5 Interfaz de linea de comandos (`main.py`)

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

Todo el procesamiento es **local**. Los modelos de docTR se descargan una sola vez desde HuggingFace al primer uso y quedan cacheados en disco.

## 6. Resultados actuales

Sobre 12 cheques de prueba (4 PDFs, 3 cheques por pagina, 6 bancos distintos):

- **Deteccion de cheques:** 12/12 (100%)
- **Extraccion de monto exacto:** 8/12 (67%)
- **Extraccion de monto parcial** (orden de magnitud correcto): 10/12 (83%)

Los 8 aciertos exactos tienen un score de confianza >= 1.3, lo cual permite al usuario filtrar resultados confiables.

## 7. Limitaciones conocidas

1. **Caligrafia muy irregular:** Cuando el monto esta escrito con letra muy desprolija o hay firmas superpuestas, el OCR puede fallar.
2. **Montos sin formato de puntos de miles:** Montos chicos como `$4.215` que no llevan puntos de miles son mas dificiles de distinguir de otros numeros en el cheque.
3. **Decimales separados:** Cuando el OCR detecta los centavos como un texto separado del monto entero, no siempre se concatenan correctamente (ej: `802.470` sin los `,20`).
4. **Velocidad:** El procesamiento toma aproximadamente 4-6 segundos por cheque en CPU (sin GPU).
