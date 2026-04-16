"""Conversion de PDF a imagenes de alta resolucion."""

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from pathlib import Path


def pdf_a_imagenes(pdf_path: str, dpi: int = 300) -> list[np.ndarray]:
    """Convierte cada pagina de un PDF a imagen numpy array (RGB).

    Args:
        pdf_path: Ruta al archivo PDF.
        dpi: Resolucion de salida.

    Returns:
        Lista de imagenes como numpy arrays (RGB).
    """
    doc = fitz.open(pdf_path)
    imagenes = []
    zoom = dpi / 72  # 72 es la resolucion base de PDF
    matrix = fitz.Matrix(zoom, zoom)

    for pagina in doc:
        pix = pagina.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        imagenes.append(np.array(img))

    doc.close()
    return imagenes


def guardar_imagen(imagen: np.ndarray, path: str):
    """Guarda un numpy array como imagen PNG."""
    Image.fromarray(imagen).save(path)
