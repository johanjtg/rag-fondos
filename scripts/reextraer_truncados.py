"""
Re-extracción quirúrgica de los fondos cuyo DFI superaba el límite anterior
de MAX_PDF_CHARS (12.000 caracteres).

Contexto: una prueba con 10 fondos confirmó que el truncado le ocultaba a
Gemini secciones enteras del documento (comisiones, calidad crediticia,
etc.) y en al menos un caso (ES0158306003) producía un valor incorrecto,
no solo ausente (comisión de suscripción: 0% en vez del 5% real). Por eso
MAX_PDF_CHARS subió de forma permanente a 30.000 en extraction/pdf_extractor.py.

Este script fuerza la re-extracción SOLO de los fondos de
--candidatos (por defecto, el listado ya calculado de ~923 fondos con texto
>12.000 caracteres, excluyendo los 10 ya reprocesados en la prueba previa).
Llama a process_pdf() directamente, sin pasar por el skip-logic de
run_pipeline (que los saltaría por ya tener fila en funds.db con datos
truncados). Los fondos que ya estaban bajo el límite de 12.000 NO se tocan
aquí — no se benefician del cambio y consumirían cuota de Gemini sin motivo.

Uso:
  python -m scripts.reextraer_truncados
  python -m scripts.reextraer_truncados --workers 4
  python -m scripts.reextraer_truncados --candidatos /ruta/a/candidatos.json
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from extraction.pdf_extractor import (
    CHROMA_PATH,
    DB_PATH,
    GEMINI_MODEL,
    MAX_PDF_CHARS,
    _build_llm,
    process_pdf,
)

log = logging.getLogger(__name__)

CANDIDATOS_PATH_DEFAULT = Path("evaluation/candidatos_reextraccion.json")
PDFS_DIR = Path("data/dfi_pdfs")


def cargar_candidatos(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def reextraer(candidatos: list[dict], pdfs_dir: Path, db_path: Path, chroma_path: Path, workers: int) -> tuple[int, int]:
    ok = errores = 0

    def _task(candidato: dict) -> bool:
        hilo_llm = _build_llm()
        pdf_path = pdfs_dir / candidato["pdf_origen"]
        return process_pdf(pdf_path, hilo_llm, db_path, chroma_path, use_docling=False) is not None

    if workers == 1:
        llm = _build_llm()
        for c in candidatos:
            pdf_path = pdfs_dir / c["pdf_origen"]
            resultado = process_pdf(pdf_path, llm, db_path, chroma_path, use_docling=False)
            ok += 1 if resultado else 0
            errores += 0 if resultado else 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futuros = {executor.submit(_task, c): c for c in candidatos}
            for futuro in as_completed(futuros):
                if futuro.result():
                    ok += 1
                else:
                    errores += 1

    return ok, errores


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-extrae con Gemini solo los fondos cuyo PDF superaba el límite anterior de MAX_PDF_CHARS."
    )
    parser.add_argument("--candidatos", type=Path, default=CANDIDATOS_PATH_DEFAULT)
    parser.add_argument("--pdfs", type=Path, default=PDFS_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--chroma", type=Path, default=CHROMA_PATH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    candidatos = cargar_candidatos(args.candidatos)
    log.info(
        "Re-extracción forzada: %d fondos | MAX_PDF_CHARS=%d | workers=%d | modelo=%s",
        len(candidatos), MAX_PDF_CHARS, args.workers, GEMINI_MODEL,
    )

    ok, errores = reextraer(candidatos, args.pdfs, args.db, args.chroma, args.workers)

    log.info("Re-extracción finalizada. OK: %d | Errores: %d | Total: %d", ok, errores, len(candidatos))


if __name__ == "__main__":
    main()
