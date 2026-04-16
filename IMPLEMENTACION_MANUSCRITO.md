# Implementación: Extractor de Monto Manuscrito en Cheques

## Resumen Ejecutivo

Se ha implementado exitosamente un sistema completo para **extraer, validar y normalizar el monto manuscrito** del campo "LA CANTIDAD DE PESOS" en cheques escaneados. La solución incluye detección robusta del campo, lectura OCR, conversión de palabras a números, y scoring de confianza basado en validación cruzada.

---

## Archivos Implementados

### 1. **src/extractors/manuscrito_extractor.py** (NUEVO)
Módulo principal con la clase `ManuscritoExtractor` (340+ líneas).

**Estructura:**
- **Dataclass `ManuscritoOCRResult`**: Contiene todos los campos de salida requeridos
- **Clase `ManuscritoExtractor`**: Orquesta todo el proceso de extracción

**Métodos principales:**
- `extraer(cheque_img, monto_numerico)`: Orquestador principal
- `_definir_zona_busqueda()`: Define zona centro-izquierda (25%-55% vertical, 0%-50% horizontal)
- `_encontrar_label()`: Busca "LA CANTIDAD DE PESOS" (case-insensitive, fuzzy)
- `_recortar_zona_manuscrito()`: Extrae región de renglones manuscritos alrededor del label
- `_leer_manuscrito()`: OCR con múltiples preprocesados (OTSU, escalado x2)
- `_normalizar_texto_a_numero()`: Convierte palabras en español a números usando `text2number`
- `_calcular_inconsistencia()`: Computa diferencia % entre montos
- `_calcular_confianza()`: Genera score de confianza (0-10) basado en:
  - Si |inconsistencia_pct| ≤ 10%: Score ~9.5 (ALTA CONFIANZA)
  - Si > 10%: Score ~6-8 (CONFIANZA MEDIA)
  - Si ilegible: Score ~2-4 (BAJA CONFIANZA)
- `_imagen_a_base64()`: Serializa imagen para JSON

**Características:**
- Maneja números fragmentados entre renglones (guiones)
- Robusto ante errores tipográficos y baja confianza OCR
- Preprocesamiento adaptativo (3 estrategias de OCR)
- Serializable a JSON

---

### 2. **src/extractors/monto_extractor.py** (MODIFICADO)

**Cambios realizados:**

a) **Actualización de `MontoOCRResult` dataclass**:
   - Agregados campos manuscritos opcionales:
     ```python
     monto_manuscrito: Optional[float] = None
     monto_manuscrito_raw: Optional[str] = None
     monto_manuscrito_score: float = 2.0
     monto_manuscrito_confidence_ocr: float = 0.0
     monto_inconsistencia_pct: Optional[float] = None
     monto_manuscrito_zona_base64: Optional[str] = None
     validacion_alineada: bool = False
     ```

b) **Actualización de `MontoExtractor.__init__`**:
   - Parámetro opcional `manuscrito_extractor` para inyección de dependencia
   - Permite uso independiente o combinado

c) **Nuevo método `extraer_con_manuscrito()`**:
   - Orquesta extracción de monto numérico + manuscrito
   - Combina resultados en un único `MontoOCRResult`
   - Pasa el monto numérico al validador manuscrito para comparación

---

### 3. **src/extractors/__init__.py** (MODIFICADO)

Exporta nuevas clases:
```python
from .manuscrito_extractor import ManuscritoExtractor, ManuscritoOCRResult
```

---

### 4. **requirements.txt** (MODIFICADO)

Agregada dependencia:
```
text2number>=1.1
```

---

## Output JSON

El extractor genera output serializable a JSON con la siguiente estructura:

```json
{
  "monto_manuscrito": 4000000.0,
  "monto_manuscrito_raw": "cuatro millones",
  "monto_manuscrito_score": 9.2,
  "monto_manuscrito_confidence_ocr": 0.87,
  "monto_inconsistencia_pct": -2.5,
  "monto_manuscrito_zona_base64": "iVBOR...",  // PNG codificado en base64
  "validacion_alineada": true
}
```

---

## Funcionalidades Implementadas

### ✓ Detección del Campo
- Búsqueda case-insensitive: "LA CANTIDAD DE PESOS"
- Búsqueda fuzzy: detecta componentes "CANTIDAD" + "PESOS"
- Zona definida: 25%-55% vertical, 0%-50% horizontal

### ✓ Lectura OCR Manuscrita
- Múltiples estrategias de preprocesado:
  1. Sin procesamiento (imagen cruda)
  2. Binarización OTSU
  3. Escalado x2 + OTSU
- Captura confidence score de cada estrategia
- Retorna candidato con mayor confianza

### ✓ Normalización de Palabras a Número
- Biblioteca `text2number` para español
- Convierte: "cuatro millones" → 4000000
- Fallbacks:
  1. Extracción de dígitos puros
  2. Regex para números con decimales (punto o coma)

### ✓ Comparación y Scoring
- Fórmula: `inconsistencia_pct = ((monto_numerico - monto_manuscrito) / monto_manuscrito) * 100`
- Scoring:
  - |inconsistencia_pct| ≤ 10%: Score 9.5 → `validacion_alineada = true`
  - 10% < |inconsistencia_pct| ≤ 20%: Score 6-8 → `validacion_alineada = false`
  - |inconsistencia_pct| > 20%: Score 2-4 → `validacion_alineada = false`
- Boosting: Si OCR confidence > 0.8, +0.5 a score

### ✓ Manejo de Casos Edge
- Manuscrito ilegible: Retorna campos nulos pero incluye `zona_base64` para debugging
- Números incompletos/fragmentados: Manejo robusto con fallbacks
- Errores OCR: Múltiples intentos con diferentes preprocesados
- Sin monto numérico: Score basado solo en OCR confidence

### ✓ Serialización JSON
- Todos los campos son JSON-serializable
- Imagen almacenada en base64 para portabilidad
- Campos opcionales manejados correctamente

---

## Arquitectura de Integración

### Patrón de inyección de dependencias:
```python
from src.ocr.ocr_readers import DocTRReader
from src.extractors import ManuscritoExtractor, MontoExtractor

# Uso independiente
ocr = DocTRReader()
manuscrito = ManuscritoExtractor(ocr)
resultado_manuscrito = manuscrito.extraer(cheque_img, monto_numerico=4000000.0)

# Uso integrado
monto_extractor = MontoExtractor(ocr, manuscrito_extractor=manuscrito)
resultado_completo = monto_extractor.extraer_con_manuscrito(cheque_img)
```

---

## Tecnologías Utilizadas

- **text2number**: Conversión de palabras a números en español
- **OpenCV (cv2)**: Preprocesamiento de imágenes (OTSU, resize)
- **NumPy**: Manipulación de arrays
- **Base64**: Serialización de imágenes
- **Dataclasses**: Definición de estructuras de datos
- **ABC/abstractmethod**: Patrón de interfaces (heredado de OCRReader)

---

## Checklist de Implementación Completado

- [x] Instalación y verificación de `text2number`
- [x] Creación de `ManuscritoExtractor` con método `extraer()`
- [x] Implementación de búsqueda de "LA CANTIDAD DE PESOS" (case-insensitive, robust)
- [x] Preprocesamiento OCR (OTSU, escalado) similar a `monto_extractor.py`
- [x] Normalización texto→número con `text2number` y fallbacks
- [x] Comparación y scoring de confianza
- [x] Actualización de `MontoExtractor` para combinar resultados
- [x] Actualización de `MontoOCRResult` dataclass con campos manuscritos
- [x] Docstrings claros en todos los métodos
- [x] Serialización a JSON soportada
- [x] Exportación en `__init__.py`
- [x] Actualización de `requirements.txt`
- [x] Validación de sintaxis Python

---

## Notas Técnicas Importantes

1. **Robustez**: El margen ±10% es flexible y configurable en el método `_calcular_confianza()`
2. **Debugging**: La zona recortada se guarda en base64 para inspección manual
3. **Performance**: Se usan 3 estrategias OCR que se ejecutan en paralelo conceptualmente pero serialmente en el código actual
4. **Extensibilidad**: El patrón de inyección permite intercambiar OCRReader sin modificar ManuscritoExtractor
5. **CSV/JSON**: Todos los campos son primitivos o strings base64, facilitando exportación a cualquier formato

---

## Próximos Pasos Sugeridos

1. Tuning de zonas de búsqueda con datos reales
2. Ajuste del margen de confianza según metrología real
3. Optimización de performance (parallelización de OCR)
4. Agregación de tests unitarios con cheques reales
