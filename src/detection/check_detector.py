"""Deteccion y recorte de cheques individuales en una imagen escaneada."""

import cv2
import numpy as np


def detectar_cheques(imagen: np.ndarray, min_area_ratio: float = 0.05) -> list[np.ndarray]:
    """Detecta y recorta cheques individuales de una imagen de pagina escaneada.

    Los cheques argentinos tienen fondo verde/turquesa distintivo y son rectangulares.

    Args:
        imagen: Imagen de la pagina completa (RGB numpy array).
        min_area_ratio: Area minima del cheque como proporcion del area total.

    Returns:
        Lista de imagenes recortadas de cada cheque, ordenadas de arriba a abajo.
    """
    alto, ancho = imagen.shape[:2]
    area_total = alto * ancho
    min_area = area_total * min_area_ratio

    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)

    # Aplicar blur para reducir ruido
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # Deteccion de bordes con Canny
    bordes = cv2.Canny(blur, 30, 100)

    # Dilatar bordes para cerrar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilatado = cv2.dilate(bordes, kernel, iterations=3)

    # Encontrar contornos
    contornos, _ = cv2.findContours(dilatado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cheques = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < min_area:
            continue

        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(contorno)

        # Filtrar por relacion de aspecto (cheques son mas anchos que altos)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 1.3 or aspect_ratio > 5.0:
            continue

        # Filtrar por tamaño minimo
        if w < ancho * 0.4 or h < alto * 0.08:
            continue

        # Agregar margen
        margen = 5
        x1 = max(0, x - margen)
        y1 = max(0, y - margen)
        x2 = min(ancho, x + w + margen)
        y2 = min(alto, y + h + margen)

        cheques.append((y1, x1, y2, x2))

    # Ordenar de arriba a abajo
    cheques.sort(key=lambda c: c[0])

    # Recortar imagenes
    imagenes_cheques = []
    for y1, x1, y2, x2 in cheques:
        recorte = imagen[y1:y2, x1:x2].copy()
        imagenes_cheques.append(recorte)

    return imagenes_cheques
