from doctr.models import ocr_predictor
import numpy as np
from PIL import Image
import fitz

print("Loading PDF...")
doc = fitz.open('cheques/Scan CH1.pdf')
page = doc[0]
pix = page.get_pixmap(dpi=150)  # Lower DPI for faster processing
img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
test_image = np.array(img)
print(f"Image loaded: {test_image.shape}")

print("Initializing docTR...")
doctr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
print("docTR ready")

print("Testing amount extraction logic...")
h, w = test_image.shape[:2]
region = test_image[0:int(h * 0.2), int(w * 0.6):w]
print(f"Region shape: {region.shape}")

result = doctr_model([region])
print("OCR completed!")

# Simulate the _doctr_read method
textos = []
for page_result in result.pages:
    for block in page_result.blocks:
        for line in block.lines:
            for word in line.words:
                geo = word.geometry
                cx = (geo[0][0] + geo[1][0]) / 2
                cy = (geo[0][1] + geo[1][1]) / 2
                textos.append((word.value, word.confidence, cx, cy))

print(f"OCR results: {len(textos)} items")
for txt, conf, cx, cy in textos[:10]:
    print(f"  '{txt}' (conf={conf:.2f}, pos={cx:.2f},{cy:.2f})")

# Test _extraer_montos logic
def _extraer_montos(textos, cerca_dolar=False):
    """Extract amount candidates from OCR texts."""
    dolar_cy = None
    for txt, conf, cx, cy in textos:
        if '$' in txt:
            dolar_cy = cy
            break

    candidatos = []
    for txt, conf, cx, cy in textos:
        limpio = txt.rstrip('-').rstrip(',').strip()
        # Clean $ or S at start
        limpio = re.sub(r'^[\$Ss]\s*[\[\(]?', '', limpio)
        limpio = limpio.rstrip('-').rstrip(',').strip()
        if not limpio:
            continue

        # Format with thousand dots (X.XXX.XXX)
        if re.match(r'^\d{1,4}(\.\d{3})+(,\d{1,2})?$', limpio):
            candidatos.append(limpio)
            continue

        # Pure number 6+ digits, only if near $
        if re.match(r'^\d{6,}$', limpio):
            if cerca_dolar or (dolar_cy is not None and abs(cy - dolar_cy) < 0.25):
                candidatos.append(limpio)
                continue

        # Amount attached to $
        import re
        m = re.search(r'[\$]\s*[\[\(]?([\d.,]+)', txt)
        if m:
            monto = m.group(1).rstrip('-').rstrip(',')
            digitos = re.sub(r'[^0-9]', '', monto)
            if len(digitos) >= 3:
                candidatos.append(monto)

    return candidatos

import re
candidatos = _extraer_montos(textos, cerca_dolar=False)
print(f"Amount candidates found: {len(candidatos)}")
for cand in candidatos:
    print(f"  '{cand}'")

# Test scoring
def _score(txt, cerca_dolar=False):
    if not txt:
        return -1
    digitos = re.sub(r'[^0-9]', '', txt)
    score = len(digitos) * 0.05

    # Format with thousand dots = high confidence
    if re.match(r'^\d{1,4}(\.\d{3})+(,\d{1,2})?$', txt):
        score += 5.0
    elif re.match(r'^\d{6,}$', txt):
        score += 1.0
        if len(digitos) >= 9:
            score -= 2.0
        elif len(digitos) == 8:
            score -= 0.8  # 8 digits = typical check number
        elif len(digitos) == 7 and not cerca_dolar:
            score -= 0.3
    return score

print("Scoring candidates:")
for cand in candidatos:
    score = _score(cand, False)
    print(f"  '{cand}' -> score {score:.1f}")