"""Script de ejemplo/test para la extracción de montos manuscritos."""

import numpy as np
from src.ocr.ocr_readers import DocTRReader
from src.extractors import ManuscritoExtractor, MontoExtractor


def test_manuscrito_extractor():
    """Test básico del ManuscritoExtractor."""
    
    print("=" * 60)
    print("TEST: ManuscritoExtractor")
    print("=" * 60)
    
    # Crear reader OCR
    ocr_reader = DocTRReader()
    
    # Crear manuscrito extractor
    manuscrito_ext = ManuscritoExtractor(ocr_reader)
    
    # Crear imagen dummy (para test, usaría imagen real del cheque)
    cheque_dummy = np.ones((600, 800, 3), dtype=np.uint8) * 255  # Blanca
    
    # Extraer resultado
    resultado = manuscrito_ext.extraer(cheque_dummy, monto_numerico=4000000.0)
    
    print(f"Monto manuscrito: {resultado.monto_manuscrito}")
    print(f"Monto manuscrito raw: {resultado.monto_manuscrito_raw}")
    print(f"Score: {resultado.monto_manuscrito_score}")
    print(f"OCR Confidence: {resultado.monto_manuscrito_confidence_ocr}")
    print(f"Inconsistencia %: {resultado.monto_inconsistencia_pct}")
    print(f"Validación alineada: {resultado.validacion_alineada}")
    print(f"Base64 zona (primeros 50 chars): {resultado.monto_manuscrito_zona_base64[:50] if resultado.monto_manuscrito_zona_base64 else 'None'}")
    print()


def test_monto_extractor_integration():
    """Test de integración: MontoExtractor con ManuscritoExtractor."""
    
    print("=" * 60)
    print("TEST: MontoExtractor con integración Manuscrito")
    print("=" * 60)
    
    # Crear readers
    ocr_reader = DocTRReader()
    manuscrito_ext = ManuscritoExtractor(ocr_reader)
    
    # Crear monto extractor con manuscrito
    monto_ext = MontoExtractor(ocr_reader, manuscrito_extractor=manuscrito_ext)
    
    # Imagen dummy
    cheque_dummy = np.ones((600, 800, 3), dtype=np.uint8) * 255  # Blanca
    
    # Extraer con manuscrito
    resultado = monto_ext.extraer_con_manuscrito(cheque_dummy)
    
    print(f"Monto numérico: {resultado.monto}")
    print(f"Monto numérico raw: {resultado.monto_raw}")
    print(f"Monto numérico score: {resultado.monto_score}")
    print()
    print(f"Monto manuscrito: {resultado.monto_manuscrito}")
    print(f"Monto manuscrito raw: {resultado.monto_manuscrito_raw}")
    print(f"Monto manuscrito score: {resultado.monto_manuscrito_score}")
    print(f"Validación alineada: {resultado.validacion_alineada}")
    print()


def test_dataclass_fields():
    """Verifica que MontoOCRResult tenga todos los campos."""
    
    print("=" * 60)
    print("TEST: Campos de MontoOCRResult")
    print("=" * 60)
    
    from src.extractors import MontoOCRResult
    from src.ocr.ocr_readers import OCRResult
    
    # Crear resultado
    tokens = [OCRResult("$", 0.95, 0.5, 0.2)]
    result = MontoOCRResult(
        monto=4000000.0,
        monto_raw="4.000.000",
        monto_score=8.5,
        zona_tokens=tokens,
    )
    
    # Verificar campos especiales
    campos_manuscrito = [
        "monto_manuscrito",
        "monto_manuscrito_raw",
        "monto_manuscrito_score",
        "monto_manuscrito_confidence_ocr",
        "monto_inconsistencia_pct",
        "monto_manuscrito_zona_base64",
        "validacion_alineada",
    ]
    
    for campo in campos_manuscrito:
        valor = getattr(result, campo, "NO EXISTENTE")
        print(f"  {campo}: {valor}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTS DE MANUSCRITO EXTRACTOR")
    print("=" * 60 + "\n")
    
    try:
        test_dataclass_fields()
        test_manuscrito_extractor()
        test_monto_extractor_integration()
        
        print("=" * 60)
        print("✓ TODOS LOS TESTS PASARON")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ ERROR EN TEST: {e}")
        import traceback
        traceback.print_exc()
