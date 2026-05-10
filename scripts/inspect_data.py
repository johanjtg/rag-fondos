"""
Herramienta de inspección de datos extraídos.

Permite validar que la extracción de PDFs ha funcionado correctamente,
consultando tanto SQLite como ChromaDB.

Uso:
  python scripts/inspect_data.py                  # resumen general
  python scripts/inspect_data.py --isin ES0156873004   # fondo concreto
  python scripts/inspect_data.py --search "renta fija"  # búsqueda por nombre
  python scripts/inspect_data.py --nulls           # fondos con campos vacíos
  python scripts/inspect_data.py --chroma "tecnología" # test semántico ChromaDB
"""

import argparse
import json
import sqlite3
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_PATH    = Path("database/funds.db")
CHROMA_PATH = Path("database/chroma")
CHROMA_COLLECTION = "politica_inversion"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

SEP  = "─" * 80
SEP2 = "═" * 80


# ── Helpers ───────────────────────────────────────────────────────────────────

def _con() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Base de datos no encontrada: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _truncate(text: str, n: int = 120) -> str:
    if not text:
        return "—"
    return text[:n] + "…" if len(text) > n else text


# ── Comandos ──────────────────────────────────────────────────────────────────

def cmd_resumen():
    """Muestra estadísticas generales de la base de datos."""
    with _con() as con:
        total = con.execute("SELECT COUNT(*) FROM funds").fetchone()[0]
        con_riesgo = con.execute("SELECT COUNT(*) FROM funds WHERE nivel_riesgo IS NOT NULL").fetchone()[0]
        con_horizonte = con.execute("SELECT COUNT(*) FROM funds WHERE horizonte_recomendado_anios IS NOT NULL").fetchone()[0]
        con_politica = con.execute("SELECT COUNT(*) FROM funds WHERE politica_inversion IS NOT NULL AND politica_inversion != ''").fetchone()[0]
        con_esg = con.execute("SELECT COUNT(*) FROM funds WHERE esg = 1").fetchone()[0]
        con_minimo = con.execute("SELECT COUNT(*) FROM funds WHERE importe_minimo_inversion IS NOT NULL").fetchone()[0]

        riesgo_dist = con.execute(
            "SELECT nivel_riesgo, COUNT(*) as n FROM funds GROUP BY nivel_riesgo ORDER BY nivel_riesgo"
        ).fetchall()

        gestora_top = con.execute(
            "SELECT gestora, COUNT(*) as n FROM funds GROUP BY gestora ORDER BY n DESC LIMIT 10"
        ).fetchall()

    print(SEP2)
    print("  RESUMEN DE DATOS EXTRAÍDOS")
    print(SEP2)
    print(f"  Total fondos en SQLite : {total}")
    print(f"  Con nivel_riesgo       : {con_riesgo}  ({con_riesgo/total*100:.0f}%)")
    print(f"  Con horizonte temporal : {con_horizonte}  ({con_horizonte/total*100:.0f}%)")
    print(f"  Con política inversión : {con_politica}  ({con_politica/total*100:.0f}%)")
    print(f"  Con importe mínimo     : {con_minimo}  ({con_minimo/total*100:.0f}%)")
    print(f"  Fondos ESG             : {con_esg}  ({con_esg/total*100:.0f}%)")

    print(f"\n  Distribución por nivel de riesgo:")
    for row in riesgo_dist:
        nivel = row["nivel_riesgo"] or "null"
        barra = "█" * row["n"]
        print(f"    Nivel {nivel:>4} : {row['n']:>4}  {barra[:40]}")

    print(f"\n  Top 10 gestoras:")
    for row in gestora_top:
        print(f"    {row['gestora'][:45]:<45} {row['n']:>4} fondos")

    # ChromaDB
    try:
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        col = client.get_collection(CHROMA_COLLECTION, embedding_function=ef)
        print(f"\n  Documentos en ChromaDB : {col.count()}")
    except Exception as e:
        print(f"\n  ChromaDB no disponible : {e}")

    print(SEP2)


def cmd_fondo(isin: str):
    """Muestra todos los campos extraídos de un fondo concreto."""
    with _con() as con:
        row = con.execute("SELECT * FROM funds WHERE isin = ?", (isin,)).fetchone()

    if not row:
        print(f"No se encontró ningún fondo con ISIN: {isin}")
        return

    d = dict(row)
    print(SEP2)
    print(f"  FONDO: {d['nombre_fondo']}")
    print(SEP2)

    campos_basicos = [
        ("ISIN",               "isin"),
        ("Nombre",             "nombre_fondo"),
        ("Gestora",            "gestora"),
        ("Nº Registro",        "numero_registro"),
        ("Categoría",          "categoria"),
        ("Tipo gestión",       "tipo_gestion"),
        ("Índice referencia",  "indice_referencia"),
    ]
    campos_riesgo = [
        ("Nivel riesgo (1-7)", "nivel_riesgo"),
        ("Perfil riesgo",      "perfil_riesgo"),
        ("Perfil inversor",    "perfil_inversor"),
        ("Volatilidad",        "volatilidad"),
        ("Horizonte (años)",   "horizonte_recomendado_anios"),
        ("Restricc. liquidez", "restricciones_liquidez"),
    ]
    campos_costes = [
        ("Importe mínimo (€)", "importe_minimo_inversion"),
        ("Comis. suscripción", "comision_suscripcion"),
        ("Comis. reembolso",   "comision_reembolso"),
        ("Comis. gestión",     "comision_gestion"),
        ("Comis. éxito",       "comision_exito"),
        ("Comis. depósito",    "comision_deposito"),
    ]
    campos_extra = [
        ("ESG",                "esg"),
        ("Divisa cobertura",   "divisa_cobertura"),
        ("Política dividendos","politica_dividendos"),
        ("RV %",               "distribucion_renta_variable_pct"),
        ("RF %",               "distribucion_renta_fija_pct"),
        ("Calidad crediticia", "calidad_crediticia"),
        ("PDF origen",         "pdf_origen"),
    ]

    for titulo, campos in [
        ("IDENTIFICACIÓN",  campos_basicos),
        ("RIESGO Y PLAZO",  campos_riesgo),
        ("COSTES",          campos_costes),
        ("OTROS",           campos_extra),
    ]:
        print(f"\n  {titulo}")
        print(SEP)
        for label, key in campos:
            print(f"  {label:<22} {_fmt(d.get(key))}")

    # JSON fields
    for key in ("universo_activos", "distribucion_sectorial", "distribucion_geografica"):
        val = d.get(key)
        if val:
            print(f"\n  {key.upper().replace('_', ' ')}")
            print(SEP)
            try:
                parsed = json.loads(val)
                print(f"  {json.dumps(parsed, ensure_ascii=False, indent=4)}")
            except Exception:
                print(f"  {val}")

    # Política de inversión
    print(f"\n  POLÍTICA DE INVERSIÓN")
    print(SEP)
    politica = d.get("politica_inversion") or "—"
    # Imprimir en bloques de 80 chars
    for i in range(0, len(politica), 80):
        print(f"  {politica[i:i+80]}")

    print(SEP2)


def cmd_search(texto: str):
    """Busca fondos cuyo nombre o gestora contenga el texto dado."""
    with _con() as con:
        rows = con.execute(
            """SELECT isin, nombre_fondo, gestora, nivel_riesgo,
                      horizonte_recomendado_anios, importe_minimo_inversion, esg
               FROM funds
               WHERE nombre_fondo LIKE ? OR gestora LIKE ?
               ORDER BY nombre_fondo
               LIMIT 30""",
            (f"%{texto}%", f"%{texto}%"),
        ).fetchall()

    print(SEP2)
    print(f"  BÚSQUEDA: '{texto}'  →  {len(rows)} resultados")
    print(SEP2)
    print(f"  {'ISIN':14}  {'Fondo':45}  {'Riesgo':7}  {'Horizonte':10}  {'Mínimo':10}  ESG")
    print(SEP)
    for r in rows:
        print(
            f"  {r['isin'] or '—':14}  "
            f"{(r['nombre_fondo'] or '')[:45]:45}  "
            f"{_fmt(r['nivel_riesgo']):7}  "
            f"{_fmt(r['horizonte_recomendado_anios']):10}  "
            f"{_fmt(r['importe_minimo_inversion']):10}  "
            f"{_fmt(r['esg'])}"
        )
    print(SEP2)


def cmd_nulls():
    """Muestra fondos con campos críticos vacíos para detectar extracciones fallidas."""
    with _con() as con:
        rows = con.execute(
            """SELECT isin, nombre_fondo, pdf_origen,
                      nivel_riesgo, horizonte_recomendado_anios,
                      politica_inversion, importe_minimo_inversion
               FROM funds
               WHERE nivel_riesgo IS NULL
                  OR horizonte_recomendado_anios IS NULL
                  OR politica_inversion IS NULL
                  OR politica_inversion = ''
               ORDER BY nombre_fondo
               LIMIT 50"""
        ).fetchall()

    print(SEP2)
    print(f"  FONDOS CON CAMPOS CRÍTICOS VACÍOS  →  {len(rows)} encontrados")
    print(SEP2)
    print(f"  {'PDF':30}  {'Riesgo':7}  {'Horizonte':10}  {'Política':8}  Nombre")
    print(SEP)
    for r in rows:
        tiene_politica = "✓" if r["politica_inversion"] else "✗"
        print(
            f"  {(r['pdf_origen'] or '—')[:30]:30}  "
            f"{_fmt(r['nivel_riesgo']):7}  "
            f"{_fmt(r['horizonte_recomendado_anios']):10}  "
            f"{tiene_politica:8}  "
            f"{(r['nombre_fondo'] or '')[:40]}"
        )
    print(SEP2)


def cmd_chroma(query: str):
    """Hace una búsqueda semántica en ChromaDB y muestra los resultados."""
    try:
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        col = client.get_collection(CHROMA_COLLECTION, embedding_function=ef)
    except Exception as e:
        print(f"ChromaDB no disponible: {e}")
        return

    results = col.query(
        query_texts=[query],
        n_results=min(10, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    print(SEP2)
    print(f"  BÚSQUEDA SEMÁNTICA EN CHROMADB: '{query}'")
    print(f"  Total documentos indexados: {col.count()}")
    print(SEP2)
    print(f"  {'Similitud':10}  {'ISIN':14}  {'Riesgo':7}  {'Fondo':40}")
    print(SEP)

    for isin, dist, meta, doc in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
        results["documents"][0],
    ):
        similitud = round(1.0 - dist / 2.0, 4)
        print(
            f"  {similitud:.4f}      "
            f"{isin:14}  "
            f"{meta.get('nivel_riesgo', '—'):7}  "
            f"{meta.get('nombre_fondo', '')[:40]}"
        )
        print(f"    Política: {_truncate(doc, 100)}")
        print()

    print(SEP2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspecciona los datos extraídos en SQLite y ChromaDB."
    )
    parser.add_argument("--isin",   help="Muestra todos los campos de un fondo por ISIN")
    parser.add_argument("--search", help="Busca fondos por nombre o gestora")
    parser.add_argument("--nulls",  action="store_true", help="Fondos con campos críticos vacíos")
    parser.add_argument("--chroma", help="Búsqueda semántica en ChromaDB")
    args = parser.parse_args()

    if args.isin:
        cmd_fondo(args.isin)
    elif args.search:
        cmd_search(args.search)
    elif args.nulls:
        cmd_nulls()
    elif args.chroma:
        cmd_chroma(args.chroma)
    else:
        cmd_resumen()


if __name__ == "__main__":
    main()
