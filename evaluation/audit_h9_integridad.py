"""
Auditoría de integridad H9: ¿el PDF guardado es realmente el DFI/KID?

Contexto
--------
Al completar el golden dataset de extracción se detectó que el fondo
V86467461 (ABACO RENTA FIJA MIXTA GLOBAL, FI) tenía en `pdf_origen` un
documento que en realidad era el Reglamento de Gestión del fondo, no el
DFI. Es el mismo problema ya documentado para fondos multi-clase o con
compartimentos: CNMV publica el DFI en la vista de clases/compartimentos
(`vista=2` o `vista=3`), no en la vista por defecto (`vista=0`), y el
scraper puede terminar enlazando el documento equivocado si cae al
fallback posicional.

Este script comprueba, para TODOS los fondos ya presentes en
`database/funds.db`, si el PDF referenciado en `pdf_origen` pasa la misma
validación que ya usa el scraper (`_validar_pdf_es_dfi`, en
`scraper/cnmv_scraper.py`) para descartar los falsos positivos conocidos
(Reglamento de Gestión, anexo SFDR Art. 8 ocupando el lugar del DFI).

Importante: esta validación es negativa, no positiva. Solo descarta los
documentos que coinciden con los marcadores ya observados
(`DOC_INVALIDO_MARCADORES`); no garantiza que un PDF que la supera sea
efectivamente un DFI bien formado, solo que no es uno de los dos
documentos equivocados detectados hasta ahora.

Uso
---
    python evaluation/audit_h9_integridad.py

Requiere que los PDFs estén descargados en `data/dfi_pdfs/` (ese
directorio está en `.gitignore`, así que hay que descargarlos localmente
antes de correr el script — ver `data/manifest.csv` para las URLs de
CNMV de cada fondo).

Última ejecución registrada
----------------------------
Fecha: 2026-07-04
Resultado: 99/99 fondos con PDF válido (0 documentos equivocados
detectados, 0 PDFs faltantes, 0 errores de lectura). El caso V86467461
fue corregido en un rescrape posterior de la BD (pasó a registrarse por
ISIN de clase, ES0140072002, con el DFI real).
"""

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scraper.cnmv_scraper import _validar_pdf_es_dfi  # noqa: E402

DB_PATH = REPO_ROOT / "database" / "funds.db"
PDF_DIR = REPO_ROOT / "data" / "dfi_pdfs"


def audit() -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT isin, nombre_fondo, pdf_origen FROM funds ORDER BY isin")
    rows = cur.fetchall()

    missing, invalid, error = [], [], []
    valid = 0

    for isin, nombre, pdf_origen in rows:
        pdf_path = PDF_DIR / pdf_origen
        if not pdf_path.exists():
            missing.append((isin, nombre, pdf_origen))
            continue
        try:
            es_valido, motivo = _validar_pdf_es_dfi(pdf_path)
        except Exception as exc:
            error.append((isin, nombre, pdf_origen, str(exc)))
            continue
        if es_valido:
            valid += 1
        else:
            invalid.append((isin, nombre, pdf_origen, motivo))

    print(f"Total fondos en BD: {len(rows)}")
    print(f"PDFs válidos (no coinciden con marcadores conocidos de doc inválido): {valid}")
    print(f"PDFs FALTANTES en disco: {len(missing)}")
    print(f"PDFs INVÁLIDOS (marcador de doc equivocado detectado): {len(invalid)}")
    print(f"Errores al leer PDF: {len(error)}")
    print()

    if invalid:
        print("=== FONDOS CON DOCUMENTO EQUIVOCADO ===")
        for isin, nombre, pdf_origen, motivo in invalid:
            print(f"  {isin} | {nombre} | {pdf_origen} | motivo: {motivo}")
        print()

    if missing:
        print("=== FONDOS CON PDF FALTANTE EN DISCO ===")
        for isin, nombre, pdf_origen in missing:
            print(f"  {isin} | {nombre} | {pdf_origen}")
        print()

    if error:
        print("=== ERRORES AL LEER PDF ===")
        for isin, nombre, pdf_origen, exc in error:
            print(f"  {isin} | {nombre} | {pdf_origen} | error: {exc}")

    return len(invalid)


if __name__ == "__main__":
    n_invalid = audit()
    sys.exit(1 if n_invalid else 0)
