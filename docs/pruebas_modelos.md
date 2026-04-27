# Pruebas de modelos OCR y enfoques descartados

## 1. Contexto

El campo mas critico a extraer de un cheque escaneado es el **monto numerico**, que esta escrito a mano en un recuadro de la esquina superior derecha. El desafio es que se trata de texto **manuscrito** sobre un fondo con marcas de agua y patrones de seguridad, lo cual dificulta la lectura automatica.

Se requirio que todo el procesamiento fuera **local** (sin APIs en la nube), por lo que se evaluaron exclusivamente modelos OCR que pueden ejecutarse offline.

## 2. Conjunto de prueba

Se utilizaron **12 cheques** provenientes de 4 archivos PDF escaneados a 300 DPI, emitidos por 6 bancos distintos (Banco Provincia, Galicia, Banco de la Nacion Argentina, Credicoop, Banco de Corrientes, BBVA). Los montos reales abarcan desde $4.215 hasta $10.000.000, con distintos estilos de caligrafia.

## 3. Modelos evaluados

### 3.1 EasyOCR v1.7.2

**Descripcion:** Motor OCR de proposito general basado en PyTorch. Soporta mas de 80 idiomas incluyendo español. Utiliza una arquitectura CRAFT para deteccion de texto y CRNN para reconocimiento.

**Configuracion de prueba:**
- Idiomas: español + ingles
- GPU: deshabilitada (CPU only)
- Se probo sobre el recorte de la esquina superior derecha del cheque (zona del monto)

**Variantes de preprocesamiento probadas:**
1. Imagen cruda (RGB directo del escaneo)
2. Binarizacion Otsu (escala de grises + threshold automatico)
3. Threshold adaptativo gaussiano (ventana 31, constante 10)
4. CLAHE + Otsu (mejora de contraste adaptativo seguido de binarizacion)
5. Escalado x2 + Otsu (duplicar resolucion antes de binarizar)

**Resultados sobre el monto numerico:**

| Cheque | Monto real | EasyOCR leyo | Correcto |
|--------|-----------|--------------|----------|
| 1 | 2.500.000 | `2 <0.c0` | No |
| 2 | 4.000.000 | `4,000000` / `4.009.000` | Parcial |
| 3 | 4.000.000 | `4000000` | Si |
| 4 | 4.215 | `5+.21s` | No |
| 5 | 850.000 | `850 Ooo` | Parcial |
| 6 | 802.470,20 | `8oz. 47029` | No |
| 7 | 340.000 | `3t0.o0` | No |
| 8 | 134.329 | `J?429` | No |
| 9 | 10.000.000 | (no detecta) | No |
| 10 | 750.000 | `35299` | No |
| 11 | 1.350.000 | `1252` | No |
| 12 | 6.200.000 | `$[6.200.00o` | Parcial |

**Aciertos exactos: 1/12 (8%)**
**Aciertos parciales (numero reconocible): 4/12 (33%)**

**Observaciones:**
- EasyOCR funciona bien para texto impreso (nombres de banco, CTA, CUIT) con confianza >0.8
- Para texto manuscrito la confianza cae por debajo de 0.3 y los resultados son ilegibles
- Confunde sistematicamente digitos manuscritos con letras o simbolos: `5` lo lee como `s`, `0` como `o` o `<`
- Ninguna variante de preprocesamiento mejoro sustancialmente la lectura de manuscrito
- La binarizacion Otsu fue la variante menos mala, pero sigue siendo insuficiente

**Conclusion:** Descartado como motor principal para campos manuscritos. Conservado temporalmente para texto impreso, aunque posteriormente tambien fue reemplazado.

---

### 3.2 TrOCR (Microsoft) - `trocr-base-handwritten`

**Descripcion:** Modelo transformer de tipo Vision Encoder-Decoder (ViT encoder + GPT-2 decoder) diseñado especificamente para reconocimiento de escritura manuscrita. Publicado por Microsoft Research.

**Configuracion de prueba:**
- Modelo: `microsoft/trocr-base-handwritten` (HuggingFace)
- Framework: PyTorch + HuggingFace Transformers
- Se probo tanto con imagen cruda como con binarizacion Otsu

**Resultados:**

| Cheque | Monto real | TrOCR leyo (crudo) | TrOCR leyo (Otsu) |
|--------|-----------|-------------------|-------------------|
| 1 | 2.500.000 | `SEC. 2.000.000` | `a 12.000,000` |
| 2 | 4.000.000 | `4th. Moscow` | `still now .` |
| 3 | 4.000.000 | `978 0785` | `7.000.` |
| 4 | 4.215 | `squeeze .` | `51,sixers are` |
| 5 | 850.000 | `also opposed to` | `also opposed to` |
| 6 | 802.470,20 | `sp.` | `stronger ,` |
| 7 | 340.000 | `2 References` | `stereo were -` |
| 8 | 134.329 | `assembles .` | `952s. August` |
| 9 | 10.000.000 | `spoken . It` | `THE NEW YORK` |
| 10 | 750.000 | `Ethel's Thompson` | `it to be the first...` |
| 11 | 1.350.000 | `also , accept` | `3/1330.000 -000` |
| 12 | 6.200.000 | `4- lb.200,000.` | `S.G.200.000-000` |

**Aciertos exactos: 0/12 (0%)**

**Observaciones:**
- El modelo esta entrenado exclusivamente con el dataset IAM (escritura manuscrita en ingles)
- Genera texto en ingles en lugar de reconocer digitos: produce palabras como "Moscow", "squeeze", "Thompson"
- Captura algunos patrones numericos de forma parcial: en el cheque 12 leyo `lb.200,000` (cercano a `6.200.000`) y en el cheque 1 leyo `2.000.000` (cercano a `2.500.000`)
- La binarizacion no mejora los resultados; en algunos casos los empeora

**Conclusion:** Descartado como motor principal para el monto. Sin embargo, posteriormente se integro como `TrOCRReader` en la abstraccion `OCRReader`, disponible con el flag `--trocr` como motor OCR alternativo para el scan de fechas. No se aplica sobre el monto.

---

### 3.3 PaddleOCR v3.4.0 (PP-OCRv5)

**Descripcion:** Suite OCR de Baidu basada en PaddlePaddle. Incluye modelos de deteccion (PP-OCRv5_server_det), orientacion (PP-LCNet) y reconocimiento (latin_PP-OCRv5_mobile_rec). Soporta multiples idiomas incluyendo español.

**Configuracion de prueba:**
- Framework: PaddlePaddle 3.3.1
- Idioma: español
- Modelo de reconocimiento: latin_PP-OCRv5_mobile_rec

**Resultado:** No se pudo completar la evaluacion.

**Error encontrado:**
```
NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]
(at paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc:118)
```

El error ocurre durante la inferencia del modelo de deteccion de texto, en la capa de optimizacion oneDNN de PaddlePaddle. Se trata de una incompatibilidad entre PaddleOCR 3.4.0 y PaddlePaddle 3.3.1 en Windows con procesadores Intel.

**Intentos de solucion:**
- Se probo con y sin la variable de entorno `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`
- Se actualizo la API de `ocr()` a `predict()` (la API cambio entre versiones)
- Se corrigio el parametro deprecado `use_angle_cls` por `use_textline_orientation`
- Ninguno resolvio el error de runtime de oneDNN

**Conclusion:** Descartado por incompatibilidad de versiones. Podria funcionar en Linux o con versiones anteriores de PaddlePaddle, pero no en el entorno de desarrollo disponible (Windows 11, Python 3.12, PaddlePaddle 3.3.1).

---

### 3.4 docTR v1.0.1 (python-doctr) -- MODELO SELECCIONADO

**Descripcion:** Biblioteca OCR end-to-end para documentos, desarrollada por Mindee. Combina un modelo de deteccion de texto (db_resnet50) con un modelo de reconocimiento (crnn_vgg16_bn). Soporta multiples idiomas y esta optimizada para documentos escaneados.

**Configuracion de prueba:**
- Framework: PyTorch
- Deteccion: db_resnet50
- Reconocimiento: crnn_vgg16_bn
- Modelos pretrained: si (descarga automatica desde HuggingFace)

**Resultados sobre el monto numerico:**

| Cheque | Monto real | docTR leyo | Correcto | Score |
|--------|-----------|------------|----------|-------|
| 1 | 2.500.000 | `2.500.000` | Si | 5.3 |
| 2 | 4.000.000 | `4.000.000` | Si | 5.3 |
| 3 | 4.000.000 | `4000000` | Si | 1.4 |
| 4 | 4.215 | (no detecta) | No | -1.0 |
| 5 | 850.000 | `850.000` | Si | 5.3 |
| 6 | 802.470,20 | `802.470` | Parcial | 5.3 |
| 7 | 340.000 | `30.000` | No | 5.2 |
| 8 | 134.329 | `31719529` | No | 0.6 |
| 9 | 10.000.000 | `10.000.000` | Si | 5.4 |
| 10 | 750.000 | `750000` | Si | 1.3 |
| 11 | 1.350.000 | `1350.000` | Si | 5.3 |
| 12 | 6.200.000 | `6.200.000` | Si | 5.3 |

**Aciertos exactos: 8/12 (67%)**
**Aciertos parciales (orden de magnitud correcto): 10/12 (83%)**

**Observaciones:**
- Claramente superior a los otros modelos para texto manuscrito numerico
- Reconoce correctamente el formato con puntos de miles en la mayoria de los casos
- Los fallos se deben a:
  - Monto muy chico sin formato de puntos (cheque 4: $4.215)
  - Primer digito cortado en el recorte (cheque 7: leyo `30.000` en vez de `340.000`)
  - Confusion con numeros de cheque adyacentes (cheque 8)
  - Decimales separados no concatenados (cheque 6: faltan los `,20`)
- La binarizacion Otsu mejora la lectura en algunos cheques
- El escalado x2 + Otsu permite recuperar montos que estan lejos del centro del recorte

**Conclusion:** Seleccionado como motor OCR principal.

---

## 4. Enfoques de recorte evaluados

Ademas de los modelos OCR, se probaron distintas estrategias para recortar la zona del monto antes de aplicar OCR:

### 4.1 Zona fija de la esquina superior derecha

**Descripcion:** Recortar un porcentaje fijo de la esquina superior derecha del cheque (ej: 60% derecho, 28% superior).

**Resultado:** Funciona en la mayoria de los cheques, pero falla cuando el layout del banco es diferente (el monto esta mas abajo o mas a la izquierda). Se implemento con 3 tamaños de zona crecientes como fallback.

### 4.2 Anclaje en el signo $

**Descripcion:** Buscar primero la posicion del simbolo `$` con docTR, y recortar alrededor de el.

**Resultado:** Es el metodo mas preciso. Funciona bien porque el `$` siempre esta impreso (no manuscrito) y es facilmente detectable por OCR. Se combina con las zonas fijas como fallback.

### 4.3 Deteccion del recuadro del monto por contornos

**Descripcion:** Usar deteccion de bordes (Canny) y contornos (findContours) para encontrar el rectangulo del recuadro donde esta el monto.

**Resultado:** Funciona en algunos cheques (3/12) pero es muy inconsistente porque:
- El recuadro no siempre tiene bordes bien definidos
- Cada banco tiene un diseño de recuadro diferente
- A veces detecta la zona de codigos bancarios en vez del monto

**Conclusion:** Descartado como metodo principal. Se evaluo como paso 3 de fallback pero no mejoraba los resultados.

### 4.4 Deteccion por color verde/turquesa

**Descripcion:** Detectar cheques por su color de fondo verde caracteristico usando espacio de color HSV.

**Resultado:** Solo detecta 5 de 12 cheques. Muchos bancos (Nacion, BBVA, Entre Rios) usan fondos que no caen en el rango HSV definido.

**Conclusion:** Descartado. El metodo de Canny detecta el 100% de los cheques.

## 5. Tabla comparativa final

| Motor OCR | Aciertos exactos | Manuscrito | Impreso | Estado |
|-----------|-----------------|------------|---------|--------|
| EasyOCR 1.7.2 | 1/12 (8%) | Muy pobre | Bueno | Descartado |
| TrOCR base | 0/12 (0%) | Solo ingles | N/A | Disponible via `--trocr` (fechas) |
| PaddleOCR 3.4 | N/A | N/A | N/A | Error de compatibilidad |
| **docTR 1.0.1** | **8/12 (67%)** | **Aceptable** | **Bueno** | **Seleccionado (default)** |
| Surya | No evaluado | — | — | Disponible via `--surya` |

## 6. Variantes de preprocesamiento comparadas

Se evaluaron 5 variantes de preprocesamiento de imagen antes de aplicar OCR. La evaluacion se realizo sobre la zona del monto de los 12 cheques:

| Variante | Descripcion | Mejora vs crudo |
|----------|-------------|-----------------|
| Cruda | Imagen RGB sin procesar | Baseline |
| Otsu | Escala de grises + binarizacion Otsu | Mejora en 2-3 cheques |
| Adaptativa | Threshold adaptativo gaussiano | Similar a Otsu |
| CLAHE + Otsu | Contraste adaptativo + binarizacion | Sin mejora significativa |
| Escalado x2 + Otsu | Duplicar resolucion + binarizar | Recupera 1 cheque adicional |

**Conclusion:** Se implementaron las 3 variantes mas efectivas (cruda, Otsu, x2+Otsu) como pipeline secuencial. El sistema prueba las 3 y se queda con el mejor resultado.

## 7. Lecciones aprendidas

1. **docTR supera ampliamente a EasyOCR para manuscrito**: La diferencia es de 1/12 vs 8/12 aciertos.
2. **TrOCR requiere fine-tuning por idioma**: Un modelo entrenado solo en ingles genera texto en ingles, no numeros.
3. **El anclaje en el `$` es clave**: Buscar primero un elemento impreso facilmente reconocible y recortar alrededor es mas robusto que recortar zonas fijas.
4. **No sobrecomplicar el scoring**: Un sistema de scoring simple (formato de puntos > numero puro > nada) funciona mejor que heuristicas complejas con multiples flags.
5. **Los fallos restantes son del OCR, no del recorte**: Los 4 cheques que fallan tienen caligrafia genuinamente dificil o montos parcialmente cortados por el borde del escaneo.
