"""
Limpieza de PDFs huérfanos en data/dfi_pdfs/ que no están referenciados por
ningún pdf_path válido en el manifest.

Un huérfano se clasifica en dos categorías:
  - duplicado_obsoleto: el nombre de archivo es el NIF de un fondo cuyo
    manifest ya tiene una fila con el mismo NIF pero un pdf_path DISTINTO
    (normalmente el ISIN correcto, resuelto en una pasada posterior del
    scraper). El archivo bajo el ISIN correcto existe en disco → el huérfano
    es un resto seguro de borrar.
  - doc_invalido_sin_reemplazo: el manifest tiene una fila para ese
    ISIN/NIF con pdf_path VACÍO (el fondo fue descartado por
    _validar_pdf_es_dfi, p.ej. "Reglamento de gestión" en vez de DFI), pero
    el archivo no se llegó a borrar del disco.

Cualquier huérfano que no encaje en ninguna de las dos categorías anteriores
se deja fuera de la lista de borrado (revisión manual).

Para los huérfanos "doc_invalido_sin_reemplazo", si el PDF llegó a
procesarse por error en una extracción anterior (antes de este script),
también puede haber una fila contaminada en funds.db y un documento
indexado en ChromaDB con datos extraídos del documento equivocado. Con
--execute se eliminan también esas filas, no solo el PDF.

Por defecto solo LISTA lo que borraría. Hay que pasar --execute para borrar.

Uso:
  python scripts/cleanup_orphan_pdfs.py                # dry-run, solo lista
  python scripts/cleanup_orphan_pdfs.py --execute       # borra tras listar
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path

MANIFEST_PATH = Path("data/manifest.csv")
PDFS_DIR = Path("data/dfi_pdfs")
DB_PATH = Path("database/funds.db")
CHROMA_PATH = Path("database/chroma")
CHROMA_COLLECTION = "politica_inversion"

SEP = "─" * 90


@dataclass
class Huerfano:
    path: Path
    razon: str
    reemplazo: str | None
    isin: str | None = None


def _cargar_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _clasificar_huerfanos(filas: list[dict], pdfs_dir: Path) -> tuple[list[Huerfano], list[Path]]:
    """
    Compara los PDFs en disco contra los pdf_path del manifest y clasifica
    cada huérfano. Devuelve (huerfanos_clasificados, huerfanos_sin_explicar).
    """
    nombres_validos = {Path(r["pdf_path"]).name for r in filas if r["pdf_path"]}
    por_nif = {r["nif"]: r for r in filas}
    por_isin = {r["isin"]: r for r in filas if r["isin"]}

    disco = {p.name: p for p in pdfs_dir.glob("*.pdf")}
    nombres_huerfanos = sorted(set(disco) - nombres_validos)

    clasificados: list[Huerfano] = []
    sin_explicar: list[Path] = []

    for nombre in nombres_huerfanos:
        stem = Path(nombre).stem
        fila = por_nif.get(stem) or por_isin.get(stem)

        if fila and fila["pdf_path"]:
            reemplazo = Path(fila["pdf_path"]).name
            if reemplazo != nombre and (pdfs_dir / reemplazo).exists():
                clasificados.append(Huerfano(disco[nombre], "duplicado_obsoleto", reemplazo))
                continue

        if fila and not fila["pdf_path"]:
            isin_fila = fila["isin"] or fila["nif"]
            clasificados.append(Huerfano(disco[nombre], "doc_invalido_sin_reemplazo", None, isin_fila))
            continue

        sin_explicar.append(disco[nombre])

    return clasificados, sin_explicar


def _isins_contaminados_en_db(inv: list[Huerfano], db_path: Path) -> set[str]:
    """De los huérfanos 'doc_invalido_sin_reemplazo', cuáles llegaron a tener fila en funds.db."""
    if not inv or not db_path.exists():
        return set()
    isins = [h.isin for h in inv if h.isin]
    if not isins:
        return set()
    with sqlite3.connect(db_path) as con:
        placeholders = ",".join("?" * len(isins))
        filas = con.execute(
            f"SELECT isin FROM funds WHERE isin IN ({placeholders})", isins
        ).fetchall()
    return {isin for (isin,) in filas}


def _imprimir_lista(clasificados: list[Huerfano], sin_explicar: list[Path], contaminados: set[str]) -> None:
    print(SEP)
    print(f"  HUÉRFANOS DETECTADOS: {len(clasificados) + len(sin_explicar)}")
    print(SEP)

    dup = [h for h in clasificados if h.razon == "duplicado_obsoleto"]
    inv = [h for h in clasificados if h.razon == "doc_invalido_sin_reemplazo"]

    print(f"\n  Duplicados obsoletos (reemplazados por un archivo con ISIN correcto): {len(dup)}")
    print(SEP)
    for h in dup:
        print(f"  {h.path.name:22} → reemplazado por {h.reemplazo}")

    print(f"\n  Documentos inválidos sin reemplazo (descartados en manifest, nunca borrados): {len(inv)}")
    print(SEP)
    for h in inv:
        marca = "  [fila contaminada en funds.db + ChromaDB → se borrará también]" if h.isin in contaminados else ""
        print(f"  {h.path.name}{marca}")

    if sin_explicar:
        print(f"\n  SIN EXPLICACIÓN CLARA (no se tocan, requieren revisión manual): {len(sin_explicar)}")
        print(SEP)
        for p in sin_explicar:
            print(f"  {p.name}")

    print(f"\n  Total PDFs a borrar si se ejecuta --execute: {len(clasificados)}")
    if contaminados:
        print(f"  Filas de funds.db + ChromaDB a borrar también: {len(contaminados)} ({sorted(contaminados)})")
    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lista (y opcionalmente borra) los PDFs huérfanos de data/dfi_pdfs/."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Borra los huérfanos clasificados (duplicados + inválidos). Sin este flag solo se listan.",
    )
    args = parser.parse_args()

    filas = _cargar_manifest(MANIFEST_PATH)
    clasificados, sin_explicar = _clasificar_huerfanos(filas, PDFS_DIR)
    inv = [h for h in clasificados if h.razon == "doc_invalido_sin_reemplazo"]
    contaminados = _isins_contaminados_en_db(inv, DB_PATH)
    _imprimir_lista(clasificados, sin_explicar, contaminados)

    if not args.execute:
        print("\n  (dry-run: no se ha borrado nada. Ejecuta con --execute para borrar.)")
        return

    borrados = 0
    for h in clasificados:
        h.path.unlink()
        borrados += 1
    print(f"\n  Borrados {borrados} archivos de {PDFS_DIR}.")

    if contaminados and DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as con:
            placeholders = ",".join("?" * len(contaminados))
            con.execute(f"DELETE FROM funds WHERE isin IN ({placeholders})", list(contaminados))
        print(f"  Borradas {len(contaminados)} filas de funds.db: {sorted(contaminados)}")

    if contaminados and CHROMA_PATH.exists():
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        ef = SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            coleccion = client.get_collection(CHROMA_COLLECTION, embedding_function=ef)
            coleccion.delete(ids=list(contaminados))
            print(f"  Borrados {len(contaminados)} documentos de ChromaDB.")
        except Exception as e:
            print(f"  Aviso: no se pudo limpiar ChromaDB ({e}).")


if __name__ == "__main__":
    main()
