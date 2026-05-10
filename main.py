"""
Punto de entrada principal del sistema de análisis de fondos de inversión españoles.

Modos de ejecución:
  scrape   → Descarga DFIs del portal CNMV a data/dfi_pdfs/
  extract  → Extrae datos estructurados de los PDFs y los almacena en SQLite + ChromaDB
  chat     → Lanza el chatbot conversacional de recomendación de fondos

Ejemplos:
  python main.py --mode scrape
  python main.py --mode scrape --max 50 --resume
  python main.py --mode extract
  python main.py --mode extract --workers 4
  python main.py --mode chat
  python main.py --mode chat --top 3
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Defaults compartidos ──────────────────────────────────────────────────────

DEFAULT_PDF_DIR    = Path("data/dfi_pdfs")
DEFAULT_MANIFEST   = Path("data/manifest.csv")
DEFAULT_DB         = Path("database/funds.db")
DEFAULT_CHROMA     = Path("database/chroma")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Sistema RAG de análisis y recomendación de fondos de inversión españoles.\n"
            "Fuente de datos: portal CNMV (Documentos de Datos Fundamentales).\n\n"
            "Flujo completo recomendado:\n"
            "  1. python main.py --mode scrape   → descarga PDFs del CNMV\n"
            "  2. python main.py --mode extract  → extrae datos con Gemini\n"
            "  3. python main.py --mode chat     → lanza el asesor conversacional"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode", "-m",
        required=True,
        choices=["scrape", "extract", "chat"],
        metavar="MODO",
        help=(
            "Modo de ejecución:\n"
            "  scrape   Descarga DFIs del portal CNMV\n"
            "  extract  Extrae datos estructurados de los PDFs\n"
            "  chat     Lanza el chatbot asesor de fondos"
        ),
    )

    # ── Opciones de scrape ────────────────────────────────────────────────────
    scrape = parser.add_argument_group("opciones de scrape")
    scrape.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Número máximo de fondos a descargar (por defecto: todos).",
    )
    scrape.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_PDF_DIR,
        metavar="DIR",
        help=f"Directorio de destino para los PDFs (por defecto: {DEFAULT_PDF_DIR}).",
    )
    scrape.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        metavar="CSV",
        help=f"Ruta del manifest CSV (por defecto: {DEFAULT_MANIFEST}).",
    )
    scrape.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar una descarga previa saltando fondos ya presentes en el manifest.",
    )

    # ── Opciones de extract ───────────────────────────────────────────────────
    extract = parser.add_argument_group("opciones de extract")
    extract.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_PDF_DIR,
        metavar="PATH",
        help=f"Ruta a un PDF o directorio de PDFs (por defecto: {DEFAULT_PDF_DIR}).",
    )
    extract.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        metavar="PATH",
        help=f"Ruta de la base de datos SQLite (por defecto: {DEFAULT_DB}).",
    )
    extract.add_argument(
        "--chroma",
        type=Path,
        default=DEFAULT_CHROMA,
        metavar="DIR",
        help=f"Directorio de ChromaDB (por defecto: {DEFAULT_CHROMA}).",
    )
    extract.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Hilos paralelos para llamadas al LLM (por defecto: 1).",
    )

    # ── Opciones de chat ──────────────────────────────────────────────────────
    chat = parser.add_argument_group("opciones de chat")
    chat.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Número de fondos a recomendar al finalizar el perfilado (por defecto: 5).",
    )

    # ── Global ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activar logging detallado (nivel DEBUG).",
    )

    return parser


# ── Modos ─────────────────────────────────────────────────────────────────────

def _mode_scrape(args: argparse.Namespace) -> None:
    from scraper.cnmv_scraper import run_scraper
    run_scraper(
        max_funds    = args.max,
        output_dir   = args.out,
        manifest_path= args.manifest,
        resume       = args.resume,
    )


def _mode_extract(args: argparse.Namespace) -> None:
    from extraction.pdf_extractor import run_pipeline
    run_pipeline(
        input_path  = args.input,
        db_path     = args.db,
        chroma_path = args.chroma,
        workers     = args.workers,
    )


def _mode_chat(args: argparse.Namespace) -> None:
    from chatbot.conversation import run_cli
    run_cli(
        top_n   = args.top,
        db_path = args.db,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    dispatch = {
        "scrape":  _mode_scrape,
        "extract": _mode_extract,
        "chat":    _mode_chat,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
