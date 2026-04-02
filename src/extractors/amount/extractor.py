"""Amount field extractor."""

import cv2
import numpy as np
import re
from typing import Dict, Any, List, Tuple
from ..base import FieldExtractor
from .parser import AmountParser
from .validators import AmountValidator
from ...ocr.interfaces import OCRResult


class AmountExtractor(FieldExtractor):
    """Extracts monetary amount from checks using the original working logic."""

    def __init__(self, ocr_reader):
        super().__init__(ocr_reader)
        self.parser = AmountParser()
        self.validator = AmountValidator()
        # TEMP: Access the doctr model directly for performance like original
        self.doctr_model = ocr_reader.model if hasattr(ocr_reader, 'model') else None

    @property
    def field_name(self) -> str:
        return "monto"

    def _extract_raw(self, check_image: np.ndarray) -> List[Tuple[str, bool]]:
        """Extract amount candidates using EXACT original MontoExtractor logic."""
        h, w = check_image.shape[:2]
        candidatos = []  # List of (amount_text, near_dollar)

        # ---- Step 1: Find $ and extract region around it ----
        zona_sup = check_image[0:int(h * 0.40), int(w * 0.40):w]
        textos_sup = self._doctr_read(zona_sup)
        dolar_pos = self._encontrar_dolar(textos_sup)

        if dolar_pos:
            cx, cy = dolar_pos
            zh, zw = zona_sup.shape[:2]
            x_abs = int(cx * zw)
            y_abs = int(cy * zh)

            margen_y = int(zh * 0.30)
            margen_x_izq = int(zw * 0.10)
            crop = zona_sup[
                max(0, y_abs - margen_y):min(zh, y_abs + margen_y),
                max(0, x_abs - margen_x_izq):
            ]

            if crop.size > 0:
                # CRITICAL: Use preprocessing like original (_noop, _otsu, _x2_otsu)
                for prep_fn in [self._noop, self._otsu, self._x2_otsu]:
                    img = prep_fn(crop)
                    for txt in self._extraer_montos(self._doctr_read(img), cerca_dolar=True):
                        candidatos.append((txt, True))

        # ---- Step 2: Fixed zones (always complements step 1) ----
        # CRITICAL: Original uses 'if True:' to always run this
        for x_pct, y_fin in [(0.63, 0.25), (0.58, 0.32), (0.50, 0.40)]:
            zona = check_image[0:int(h * y_fin), int(w * x_pct):w]
            # CRITICAL: Use preprocessing like original (_noop, _otsu)
            for prep_fn in [self._noop, self._otsu]:
                img = prep_fn(zona)
                for txt in self._extraer_montos(self._doctr_read(img), cerca_dolar=False):
                    candidatos.append((txt, False))
            # CRITICAL: Early exit if good candidate found
            if any(self._score(t, d) >= 5.0 for t, d in candidatos):
                break

        return candidatos

    def _parse(self, raw_candidates: List[Tuple[str, bool]]) -> Dict[str, Any]:
        """Parse candidates and select best match."""
        if not raw_candidates:
            return {'best_match': None, 'candidates': []}

        # Score all candidates
        scored_candidates = []
        for txt, cerca_dolar in raw_candidates:
            score = self._score(txt, cerca_dolar)
            scored_candidates.append({
                'raw_text': txt,
                'amount_text': txt,
                'score': score,
                'near_dollar': cerca_dolar
            })

        # Sort by score (highest first)
        scored_candidates.sort(key=lambda c: c['score'], reverse=True)

        return {
            'best_match': scored_candidates[0] if scored_candidates else None,
            'candidates': scored_candidates
        }

    def _validate(self, parsed_data: Dict[str, Any]) -> bool:
        """Validate the parsed amount."""
        return self.validator.is_valid(parsed_data)

    def _normalize(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return final normalized result."""
        best_match = validated_data.get('best_match')
        if not best_match:
            return {
                'monto': None,
                'monto_raw': '',
                'monto_score': -1.0
            }

        valor = self._normalizar(best_match['amount_text'])
        return {
            'monto': valor,
            'monto_raw': best_match['raw_text'],
            'monto_score': best_match['score']
        }

    # ---- OCR methods (ported from original) ----

    def _doctr_read(self, img: np.ndarray) -> List[Tuple[str, float, float, float]]:
        """Read text using docTR (returns: text, confidence, cx, cy)."""
        # Use direct doctr model access for performance (like original)
        if self.doctr_model:
            result = self.doctr_model([img])
        else:
            # Fallback to OCR reader
            ocr_results = self.ocr_reader.read(img)
            # Convert back to original format
            result = type('MockResult', (), {'pages': []})()
            mock_page = type('MockPage', (), {'blocks': []})()
            mock_block = type('MockBlock', (), {'lines': []})()
            mock_line = type('MockLine', (), {'words': []})()
            for ocr_result in ocr_results:
                mock_word = type('MockWord', (), {
                    'value': ocr_result.text,
                    'confidence': ocr_result.confidence,
                    'geometry': ocr_result.bbox
                })()
                mock_line.words.append(mock_word)
            mock_block.lines.append(mock_line)
            mock_page.blocks.append(mock_block)
            result.pages.append(mock_page)

        partes = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        geo = word.geometry
                        cx = (geo[0][0] + geo[1][0]) / 2
                        cy = (geo[0][1] + geo[1][1]) / 2
                        partes.append((word.value, word.confidence, cx, cy))
        return partes

    @staticmethod
    def _encontrar_dolar(textos: List[Tuple[str, float, float, float]]) -> Tuple[float, float] | None:
        """Find dollar sign position."""
        for txt, conf, cx, cy in textos:
            if '$' in txt:
                return cx, cy
        return None

    # ---- Preprocessing methods (ported from original) ----

    @staticmethod
    def _noop(img: np.ndarray) -> np.ndarray:
        return img

    @staticmethod
    def _otsu(img: np.ndarray) -> np.ndarray:
        gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _x2_otsu(img: np.ndarray) -> np.ndarray:
        esc = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gris = cv2.cvtColor(esc, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)

    # ---- Candidate extraction (ported from original) ----

    def _extraer_montos(self, textos: List[Tuple[str, float, float, float]], cerca_dolar: bool = False) -> List[str]:
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
            m = re.search(r'[\$]\s*[\[\(]?([\d.,]+)', txt)
            if m:
                monto = m.group(1).rstrip('-').rstrip(',')
                digitos = re.sub(r'[^0-9]', '', monto)
                if len(digitos) >= 3:
                    candidatos.append(monto)

        return candidatos

    # ---- Scoring (ported from original) ----

    @staticmethod
    def _score(txt: str, cerca_dolar: bool = False) -> float:
        """Score amount candidate."""
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

    # ---- Normalization (ported from original) ----

    @staticmethod
    def _normalizar(txt: str) -> float | None:
        """Convert Argentine amount format to float.

        Argentine format: dots = thousands, comma = decimals.
        '2.500.000' -> 2500000.0
        '802.470,20' -> 802470.20
        '4000000' -> 4000000.0
        """
        if not txt:
            return None
        limpio = txt.rstrip('-').rstrip(',').strip()

        if ',' in limpio:
            partes = limpio.split(',')
            entero = partes[0].replace('.', '')
            decimal = partes[1] if len(partes) > 1 else '0'
            return float(f"{entero}.{decimal}")

        return float(limpio.replace('.', ''))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB))

        # 2x upscale + Otsu
        h, w = image.shape[:2]
        upscaled = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        gray_up = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
        _, otsu_up = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images.append(cv2.cvtColor(otsu_up, cv2.COLOR_GRAY2RGB))

        return images

    def _parse(self, raw_data: List[OCRResult]) -> Dict[str, Any]:
        """Parse OCR results using AmountParser."""
        return self.parser.parse(raw_data)

    def _validate(self, parsed_data: Dict[str, Any]) -> bool:
        """Validate using AmountValidator."""
        return self.validator.is_valid(parsed_data)

    def _normalize(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return normalized amount data."""
        best_match = validated_data['best_match']

        # Convert to float
        try:
            monto_float = float(best_match['amount_text'])
        except (ValueError, TypeError):
            monto_float = None

        return {
            'monto': monto_float,
            'monto_raw': best_match['raw_text'],
            'monto_score': best_match['score']
        }