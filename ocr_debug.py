"""Script de depuracion para ver que lee docTR de una imagen.

Uso:
    python ocr_debug.py <imagen>          # leer imagen especifica
    python ocr_debug.py                   # leer todas las imagenes en test_images/
    python ocr_debug.py --annotate <img>  # guardar imagen con bounding boxes anotados
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def load_image(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"No se pudo leer: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def annotate_and_save(img_rgb: np.ndarray, tokens, out_path: Path):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    for tok in tokens:
        half_h = tok.height / 2
        x1 = int((tok.cx - tok.height) * w)   # approximate width from height
        x2 = int((tok.cx + tok.height) * w)
        y1 = int((tok.cy - half_h) * h)
        y2 = int((tok.cy + half_h) * h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img_bgr, tok.text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(out_path), img_bgr)
    print(f"  -> Imagen anotada guardada en: {out_path}")


def process(image_path: Path, annotate: bool):
    from src.ocr.ocr_readers import DocTRReader

    print(f"\n{'='*60}")
    print(f"Imagen: {image_path.name}  ({image_path.stat().st_size // 1024} KB)")
    print(f"{'='*60}")

    img = load_image(image_path)
    print(f"Tamaño: {img.shape[1]}x{img.shape[0]} px")
    print("Cargando modelo docTR (primera vez descarga ~100 MB)...")

    reader = DocTRReader()
    tokens = reader.read(img)

    if not tokens:
        print("  [sin texto detectado]")
        return

    tokens_sorted = sorted(tokens, key=lambda t: (round(t.cy, 2), t.cx))

    print(f"\n{len(tokens)} palabras detectadas (orden: arriba→abajo, izq→der):\n")
    print(f"  {'TEXTO':<20} {'CONF':>6}  {'CX':>6}  {'CY':>6}")
    print(f"  {'-'*20} {'-'*6}  {'-'*6}  {'-'*6}")
    for tok in tokens_sorted:
        print(f"  {tok.text:<20} {tok.confidence:>6.3f}  {tok.cx:>6.3f}  {tok.cy:>6.3f}")

    if annotate:
        out = image_path.parent / (image_path.stem + "_annotated" + image_path.suffix)
        annotate_and_save(img, tokens_sorted, out)


def main():
    parser = argparse.ArgumentParser(description="Prueba docTR sobre una imagen")
    parser.add_argument("imagen", nargs="?", help="Ruta a la imagen (omitir para usar test_images/)")
    parser.add_argument("--annotate", action="store_true",
                        help="Guardar imagen anotada con bounding boxes")
    args = parser.parse_args()

    if args.imagen:
        paths = [Path(args.imagen)]
    else:
        folder = Path("test_images")
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            print(f"No hay imagenes en {folder}/. Pone una imagen ahi o pasa la ruta como argumento.")
            sys.exit(1)

    for path in paths:
        process(path, annotate=args.annotate)


if __name__ == "__main__":
    main()
