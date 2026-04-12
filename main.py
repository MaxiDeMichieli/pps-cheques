"""CLI para procesamiento de cheques escaneados."""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
)

from src.pdf_processor import pdf_a_imagenes, guardar_imagen
from src.check_detector import detectar_cheques
from src.cheque_extractor import ChequeExtractor
from src.ocr_readers import DocTRReader
from src.llm_backends import OllamaBackend
from src.llm_validator import LLMValidator
from src.models import DatosCheque, guardar_cheques_json, cargar_cheques_json


def procesar_pdf(
    pdf_path: str,
    extractor: ChequeExtractor,
    output_dir: str = "output",
) -> list[DatosCheque]:
    """Procesa un PDF con cheques escaneados."""
    pdf_path = Path(pdf_path)
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Procesando: {pdf_path.name}")

    paginas = pdf_a_imagenes(str(pdf_path), dpi=300)
    print(f"  Paginas: {len(paginas)}")

    cheques_datos = []
    batch_montos_raw: list[str] = []

    for num_pag, pagina in enumerate(paginas, 1):
        cheques_img = detectar_cheques(pagina)
        print(f"  Pagina {num_pag}: {len(cheques_img)} cheques detectados")

        for idx, cheque_img in enumerate(cheques_img, 1):
            nombre_img = f"{pdf_path.stem}_p{num_pag}_ch{idx}.png"
            ruta_img = str(img_dir / nombre_img)
            guardar_imagen(cheque_img, ruta_img)

            print(f"    Cheque {idx}...", end=" ", flush=True)

            datos = extractor.extraer(cheque_img, batch_context=batch_montos_raw)
            datos.imagen_path = ruta_img
            datos.pdf_origen = pdf_path.name
            datos.pagina = num_pag
            datos.indice_en_pagina = idx

            batch_montos_raw.append(datos.monto_raw)
            cheques_datos.append(datos)

            monto_str = f"${datos.monto:,.2f}" if datos.monto else "no detectado"
            conf_str = f" llm={datos.monto_llm_confidence:.2f}" if datos.monto_llm_confidence is not None else ""
            fecha_str = f" fecha={datos.fecha_emision}" if datos.fecha_emision else ""
            print(f"Monto: {monto_str} (score={datos.monto_score:.1f}{conf_str}{fecha_str})")

    return cheques_datos


def cmd_procesar(args):
    """Comando: procesar PDF(s)."""
    print("Inicializando OCR (docTR)...")
    ocr_reader = DocTRReader()

    llm = None
    if not args.sin_llm:
        print(f"Inicializando LLM ({args.llm_model} @ {args.llm_url})...")
        backend = OllamaBackend(model=args.llm_model, base_url=args.llm_url)
        llm = LLMValidator(backend=backend)

    extractor = ChequeExtractor(ocr_reader, llm_validator=llm)
    print("Listo.\n")

    ruta = Path(args.entrada)

    if ruta.is_file() and ruta.suffix.lower() == '.pdf':
        pdfs = [ruta]
    elif ruta.is_dir():
        pdfs = sorted(ruta.glob("*.pdf"))
        if not pdfs:
            print(f"No se encontraron PDFs en {ruta}")
            return
        print(f"Encontrados {len(pdfs)} PDFs\n")
    else:
        print(f"Error: {ruta} no es un PDF ni un directorio")
        return

    todos_cheques = []
    for pdf in pdfs:
        cheques = procesar_pdf(str(pdf), extractor, args.output)
        todos_cheques.extend(cheques)
        print()

    json_path = Path(args.output) / "cheques.json"
    guardar_cheques_json(todos_cheques, str(json_path))
    print(f"Resultados guardados en: {json_path}")
    print(f"Total cheques procesados: {len(todos_cheques)}")


def cmd_listar(args):
    """Comando: listar cheques del JSON."""
    json_path = Path(args.output) / "cheques.json"
    if not json_path.exists():
        print(f"No se encontro {json_path}. Procese algun PDF primero.")
        return

    cheques = cargar_cheques_json(str(json_path))
    print(f"Total cheques: {len(cheques)}\n")

    for i, ch in enumerate(cheques, 1):
        monto = ch.get('monto')
        monto_str = f"${monto:,.2f}" if monto else "no detectado"
        score = ch.get('monto_score', 0)
        pdf = ch.get('pdf_origen', '?')
        print(f"  {i:3d}. {monto_str:>16s}  score={score:.1f}  ({pdf})")


def cmd_buscar(args):
    """Comando: buscar en cheques."""
    json_path = Path(args.output) / "cheques.json"
    if not json_path.exists():
        print(f"No se encontro {json_path}. Procese algun PDF primero.")
        return

    cheques = cargar_cheques_json(str(json_path))
    termino = args.termino.lower()
    encontrados = []

    for ch in cheques:
        for valor in ch.values():
            if isinstance(valor, str) and termino in valor.lower():
                encontrados.append(ch)
                break

    print(f"Encontrados: {len(encontrados)} cheques con '{args.termino}'\n")
    for i, ch in enumerate(encontrados, 1):
        monto = ch.get('monto')
        monto_str = f"${monto:,.2f}" if monto else "no detectado"
        print(f"  {i}. {monto_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Procesador de cheques escaneados - OCR local"
    )
    parser.add_argument("--output", "-o", default="output",
                        help="Directorio de salida (default: output)")

    subparsers = parser.add_subparsers(dest="comando", help="Comando a ejecutar")

    p_proc = subparsers.add_parser("procesar", help="Procesar PDF(s) con cheques")
    p_proc.add_argument("entrada", help="Archivo PDF o directorio con PDFs")
    p_proc.add_argument("--sin-llm", action="store_true",
                        help="Desactivar validacion LLM (solo OCR heuristico)")
    p_proc.add_argument("--llm-model", default="llama3.2",
                        help="Modelo Ollama a usar (default: llama3.2)")
    p_proc.add_argument("--llm-url", default="http://localhost:11434",
                        help="URL del servidor Ollama (default: http://localhost:11434)")
    p_proc.set_defaults(func=cmd_procesar)

    p_list = subparsers.add_parser("listar", help="Listar cheques procesados")
    p_list.set_defaults(func=cmd_listar)

    p_buscar = subparsers.add_parser("buscar", help="Buscar en cheques procesados")
    p_buscar.add_argument("termino", help="Termino de busqueda")
    p_buscar.set_defaults(func=cmd_buscar)

    args = parser.parse_args()

    if not args.comando:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
