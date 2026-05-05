"""Pipeline reutilizable para procesar un PDF de cheques.

Acepta callbacks opcionales que permiten observar el progreso en vivo,
útil para integraciones GUI (SSE) o testing. La CLI lo usa sin callbacks.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

from .detection.check_detector import detectar_cheques
from .extractors.cheque_extractor import ChequeExtractor
from .models import DatosCheque
from .pdf.pdf_processor import cargar_imagen, guardar_imagen, pdf_a_imagenes

logger = logging.getLogger(__name__)


OnPdfLoaded = Callable[[int], None]
OnChequeDetected = Callable[[int, int, str], None]
OnChequeExtracted = Callable[[int, int, DatosCheque], None]


def _extraer_de_imagen(
    ruta_img: str,
    num_pag: int,
    idx: int,
    pdf_name: str,
    extractor: ChequeExtractor,
    debug_dir: Path | None = None,
) -> DatosCheque:
    cheque_img = cargar_imagen(ruta_img)
    cheque_debug_dir = None
    if debug_dir is not None:
        cheque_debug_dir = debug_dir / f"{Path(pdf_name).stem}_p{num_pag}_ch{idx}"
        cheque_debug_dir.mkdir(exist_ok=True)
    datos = extractor.extraer(cheque_img, debug_dir=cheque_debug_dir)
    datos.imagen_path = ruta_img
    datos.pdf_origen = pdf_name
    datos.pagina = num_pag
    datos.indice_en_pagina = idx
    return datos


def procesar_pdf(
    pdf_path: str,
    extractor: ChequeExtractor,
    output_dir: str = "output",
    debug_dir: Path | None = None,
    on_pdf_loaded: OnPdfLoaded | None = None,
    on_cheque_detected: OnChequeDetected | None = None,
    on_cheque_extracted: OnChequeExtracted | None = None,
) -> list[DatosCheque]:
    """Procesa un PDF con cheques escaneados, reutilizando imágenes si ya existen.

    Los callbacks se invocan sincrónicamente desde el thread del worker:
    - ``on_pdf_loaded(total_paginas)``: una vez, antes de detectar cheques.
    - ``on_cheque_detected(pagina, indice, ruta_img)``: cuando se recorta cada cheque.
    - ``on_cheque_extracted(pagina, indice, datos)``: cuando termina la extracción.
    """
    pdf_path = Path(pdf_path)
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        img_dir.glob(f"{pdf_path.stem}_p*_ch*.png"),
        key=lambda p: tuple(int(x) for x in re.search(r"_p(\d+)_ch(\d+)\.png$", p.name).groups()),
    )

    cheques_datos: list[DatosCheque] = []

    if existing:
        logger.info("Reutilizando %d imagenes existentes para %s", len(existing), pdf_path.name)
        if on_pdf_loaded is not None:
            on_pdf_loaded(0)  # 0 indica "reutilizado, sin paginar"
        for img_path in existing:
            m = re.search(r"_p(\d+)_ch(\d+)\.png$", img_path.name)
            num_pag, idx = int(m.group(1)), int(m.group(2))
            if on_cheque_detected is not None:
                on_cheque_detected(num_pag, idx, str(img_path))
            datos = _extraer_de_imagen(
                str(img_path), num_pag, idx, pdf_path.name, extractor, debug_dir
            )
            if on_cheque_extracted is not None:
                on_cheque_extracted(num_pag, idx, datos)
            cheques_datos.append(datos)
        return cheques_datos

    paginas = pdf_a_imagenes(str(pdf_path), dpi=300)
    if on_pdf_loaded is not None:
        on_pdf_loaded(len(paginas))

    for num_pag, pagina in enumerate(paginas, 1):
        cheques_img = detectar_cheques(pagina)
        logger.info("Pagina %d: %d cheques detectados", num_pag, len(cheques_img))

        for idx, cheque_img in enumerate(cheques_img, 1):
            nombre_img = f"{pdf_path.stem}_p{num_pag}_ch{idx}.png"
            ruta_img = str(img_dir / nombre_img)
            guardar_imagen(cheque_img, ruta_img)
            if on_cheque_detected is not None:
                on_cheque_detected(num_pag, idx, ruta_img)
            datos = _extraer_de_imagen(
                ruta_img, num_pag, idx, pdf_path.name, extractor, debug_dir
            )
            if on_cheque_extracted is not None:
                on_cheque_extracted(num_pag, idx, datos)
            cheques_datos.append(datos)

    return cheques_datos
