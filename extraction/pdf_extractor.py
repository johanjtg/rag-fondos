"""
Pipeline de extracción estructurada de DFIs (Documentos de Datos Fundamentales).

Flujo por PDF:
  1. Extrae texto con pypdf (por defecto) o Docling (--docling).
  2. Llama a Gemini 2.5 vía LangChain con salida estructurada → FundModel.
  3. Persiste el fondo en SQLite (database/funds.db).
  4. Indexa politica_inversion en ChromaDB para búsqueda semántica RAG.

Uso:
  python extraction/pdf_extractor.py --input data/dfi_pdfs/
  python extraction/pdf_extractor.py --input data/dfi_pdfs/ES0123456789.pdf
  python extraction/pdf_extractor.py --input data/dfi_pdfs/ --workers 4
  python extraction/pdf_extractor.py --input data/dfi_pdfs/ --docling
  python extraction/pdf_extractor.py --input data/dfi_pdfs/ES0123456789.pdf --docling
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import re
import unicodedata

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

from extraction.fund_model import FundModel

load_dotenv()

# ── Constantes ────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash"   # cambia a gemini-2.5-pro si se necesita más precisión
DB_PATH = Path("database/funds.db")
CHROMA_PATH = Path("database/chroma")
CHROMA_COLLECTION = "politica_inversion"
MAX_PDF_CHARS = 30_000   # cubre el DFI más largo observado en el corpus (28.347 caracteres);
                          # antes en 12.000, lo que truncaba el 87% de los PDFs (ver evaluation/)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Limpieza de texto ─────────────────────────────────────────────────────────

def clean_text(texto: str, is_markdown: bool = False) -> str:
    """
    Limpia el texto extraído de un PDF para reducir tokens enviados al LLM.

    Pasos aplicados:
      1. Normaliza unicode (ej: caracteres latinos con acento en forma NFC).
      2. Reemplaza tabuladores y caracteres de control por espacios.
      3. Elimina espacios al final de cada línea.
      4. Colapsa espacios múltiples en uno (excepto dentro de tablas Markdown).
      5. Colapsa 3+ líneas en blanco consecutivas → máximo 2.
      6. Elimina líneas que solo contienen caracteres de relleno (─, ═, •, …).
      7. Strip final del texto completo.
    """
    # 1. Normalización unicode
    texto = unicodedata.normalize("NFC", texto)

    # 2. Tabuladores y otros caracteres de control → espacio
    texto = re.sub(r"[\t\r\x0c\x0b]", " ", texto)

    # 3. Espacios al final de línea
    texto = re.sub(r" +$", "", texto, flags=re.MULTILINE)

    # 4. Espacios múltiples en una línea
    #    En Markdown, las líneas de tabla empiezan con '|' — no tocar su interior
    if is_markdown:
        lineas = []
        for linea in texto.splitlines():
            if linea.startswith("|"):
                lineas.append(linea)          # tabla: conservar tal cual
            else:
                lineas.append(re.sub(r" {2,}", " ", linea))
        texto = "\n".join(lineas)
    else:
        texto = re.sub(r" {2,}", " ", texto)

    # 5. Más de 2 líneas en blanco consecutivas → 2
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    # 6. Líneas de solo caracteres de relleno (separadores visuales de PDF)
    texto = re.sub(r"^[\s─═━—–·•▪▸►●\-=_*]{3,}\s*$", "", texto, flags=re.MULTILINE)

    # 7. Strip
    return texto.strip()


# ── Extracción de texto del PDF ───────────────────────────────────────────────

def extract_text_pypdf(pdf_path: Path) -> str:
    """Extrae texto plano con pypdf. Rápido pero pierde estructura de tablas."""
    reader = PdfReader(str(pdf_path))
    partes = [page.extract_text() for page in reader.pages if page.extract_text()]
    texto_completo = clean_text("\n".join(partes), is_markdown=False)
    if len(texto_completo) > MAX_PDF_CHARS:
        log.debug("%s recortado de %d a %d caracteres", pdf_path.name, len(texto_completo), MAX_PDF_CHARS)
        return texto_completo[:MAX_PDF_CHARS]
    return texto_completo


def extract_text_docling(pdf_path: Path) -> str:
    """
    Extrae texto estructurado con Docling, exportando a Markdown.
    Preserva tablas con formato | col | col |, lo que permite a Gemini
    identificar el número marcado en la escala de riesgo SRRI (1-7).
    Más lento que pypdf pero mejora significativamente la extracción
    de fondos cuyo indicador de riesgo es una tabla visual.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "Docling no está instalado. Instálalo con: pip install docling"
        )

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    texto_completo = clean_text(result.document.export_to_markdown(), is_markdown=True)

    if len(texto_completo) > MAX_PDF_CHARS:
        log.debug(
            "%s (docling) recortado de %d a %d caracteres",
            pdf_path.name, len(texto_completo), MAX_PDF_CHARS,
        )
        return texto_completo[:MAX_PDF_CHARS]
    return texto_completo


def extract_text(pdf_path: Path, use_docling: bool = False) -> str:
    """
    Extrae el texto de un PDF.
    Si use_docling=True usa Docling (Markdown estructurado con tablas).
    Si use_docling=False usa pypdf (texto plano, más rápido).
    """
    if use_docling:
        log.debug("Usando Docling para: %s", pdf_path.name)
        return extract_text_docling(pdf_path)
    return extract_text_pypdf(pdf_path)


# ── Llamada al LLM con salida estructurada ────────────────────────────────────

SYSTEM_PROMPT = """\
Eres un analista financiero experto en fondos de inversión españoles.
Se te proporciona el texto de un Documento de Datos Fundamentales (DFI/KID).
Extrae TODOS los campos que puedas encontrar con precisión.
Si un campo no aparece en el documento, devuelve null para ese campo.
No inventes valores: solo extrae lo que está explícitamente en el texto.
Los porcentajes deben expresarse como números decimales (ej: 1.5 para 1,5%).

Para el campo nivel_riesgo (escala 1-7):
- Busca el indicador SRRI/SRI: el número del cuadro marcado en la barra 1-7.
- También puede aparecer como: "Indicador de riesgo: X", "Clase X", o simplemente
  el número subrayado/marcado en la secuencia 1 2 3 4 5 6 7.
- En el texto extraído del PDF, el cuadro marcado suele aparecer como el único número
  que aparece dos veces seguidas (ej: "1 2 3 4 4 5 6 7" → nivel 4) o con marcas
  de énfasis. Infiere el número marcado a partir del contexto.
"""

def _build_llm() -> ChatGoogleGenerativeAI:
    """Inicializa el modelo Gemini con salida estructurada vinculada a FundModel."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY no está definida en el entorno.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0,
        timeout=120,   # evita cuelgues indefinidos (p.ej. tras dormir el equipo con la llamada en vuelo)
    )


def extract_fund_data(texto: str, pdf_path: Path, llm: ChatGoogleGenerativeAI) -> FundModel:
    """
    Llama a Gemini con salida estructurada y devuelve un FundModel validado.
    Inyecta pdf_origen con el nombre del archivo fuente.
    """
    structured_llm = llm.with_structured_output(FundModel)

    mensajes = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Texto del DFI:\n\n{texto}"),
    ]

    fondo: FundModel = structured_llm.invoke(mensajes)
    fondo.pdf_origen = pdf_path.name

    # Si el LLM devuelve la cadena literal "null" o un ISIN vacío,
    # usar el stem del archivo PDF como identificador de reserva.
    if not fondo.isin or fondo.isin.lower() in ("null", "none", "n/a", ""):
        fondo.isin = pdf_path.stem
        log.warning("ISIN no encontrado en %s — usando nombre de archivo: %s", pdf_path.name, fondo.isin)

    return fondo


# ── Persistencia en SQLite ────────────────────────────────────────────────────

DDL = """\
CREATE TABLE IF NOT EXISTS funds (
    isin                        TEXT PRIMARY KEY,
    nombre_fondo                TEXT NOT NULL,
    numero_registro             TEXT,
    gestora                     TEXT NOT NULL,
    categoria                   TEXT,
    tipo_gestion                TEXT,
    indice_referencia           TEXT,
    universo_activos            TEXT,   -- JSON array
    politica_inversion          TEXT,
    nivel_riesgo                INTEGER,
    perfil_riesgo               TEXT,
    perfil_inversor             TEXT,
    volatilidad                 REAL,
    horizonte_recomendado_anios INTEGER,
    restricciones_liquidez      TEXT,
    importe_minimo_inversion    REAL,
    comision_suscripcion        REAL,
    comision_reembolso          REAL,
    comision_gestion            REAL,
    comision_exito              REAL,
    comision_deposito           REAL,
    distribucion_sectorial      TEXT,   -- JSON dict
    distribucion_geografica     TEXT,   -- JSON dict
    distribucion_renta_variable_pct REAL,
    distribucion_renta_fija_pct     REAL,
    calidad_crediticia          TEXT,
    divisa_cobertura            INTEGER,  -- 0/1
    politica_dividendos         TEXT,
    esg                         INTEGER,  -- 0/1
    pdf_origen                  TEXT
)
"""

UPSERT = """\
INSERT INTO funds VALUES (
    :isin, :nombre_fondo, :numero_registro, :gestora,
    :categoria, :tipo_gestion, :indice_referencia, :universo_activos,
    :politica_inversion, :nivel_riesgo, :perfil_riesgo, :perfil_inversor,
    :volatilidad, :horizonte_recomendado_anios, :restricciones_liquidez,
    :importe_minimo_inversion, :comision_suscripcion, :comision_reembolso,
    :comision_gestion, :comision_exito, :comision_deposito,
    :distribucion_sectorial, :distribucion_geografica,
    :distribucion_renta_variable_pct, :distribucion_renta_fija_pct,
    :calidad_crediticia, :divisa_cobertura, :politica_dividendos,
    :esg, :pdf_origen
)
ON CONFLICT(isin) DO UPDATE SET
    nombre_fondo                = excluded.nombre_fondo,
    numero_registro             = excluded.numero_registro,
    gestora                     = excluded.gestora,
    categoria                   = excluded.categoria,
    tipo_gestion                = excluded.tipo_gestion,
    indice_referencia           = excluded.indice_referencia,
    universo_activos            = excluded.universo_activos,
    politica_inversion          = excluded.politica_inversion,
    nivel_riesgo                = excluded.nivel_riesgo,
    perfil_riesgo               = excluded.perfil_riesgo,
    perfil_inversor             = excluded.perfil_inversor,
    volatilidad                 = excluded.volatilidad,
    horizonte_recomendado_anios = excluded.horizonte_recomendado_anios,
    restricciones_liquidez      = excluded.restricciones_liquidez,
    importe_minimo_inversion    = excluded.importe_minimo_inversion,
    comision_suscripcion        = excluded.comision_suscripcion,
    comision_reembolso          = excluded.comision_reembolso,
    comision_gestion            = excluded.comision_gestion,
    comision_exito              = excluded.comision_exito,
    comision_deposito           = excluded.comision_deposito,
    distribucion_sectorial      = excluded.distribucion_sectorial,
    distribucion_geografica     = excluded.distribucion_geografica,
    distribucion_renta_variable_pct = excluded.distribucion_renta_variable_pct,
    distribucion_renta_fija_pct     = excluded.distribucion_renta_fija_pct,
    calidad_crediticia          = excluded.calidad_crediticia,
    divisa_cobertura            = excluded.divisa_cobertura,
    politica_dividendos         = excluded.politica_dividendos,
    esg                         = excluded.esg,
    pdf_origen                  = excluded.pdf_origen
"""


def _fund_to_row(fondo: FundModel) -> dict:
    """Serializa un FundModel a un dict plano apto para SQLite."""
    d = fondo.model_dump()
    # Serializar tipos compuestos a JSON
    for campo in ("universo_activos", "distribucion_sectorial", "distribucion_geografica"):
        if d[campo] is not None:
            d[campo] = json.dumps(d[campo], ensure_ascii=False)
    # Convertir bool → int para SQLite
    for campo in ("divisa_cobertura", "esg"):
        if d[campo] is not None:
            d[campo] = int(d[campo])
    return d


def save_to_sqlite(fondo: FundModel, db_path: Path) -> None:
    """Inserta o actualiza un fondo en la base de datos SQLite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute(DDL)
        con.execute(UPSERT, _fund_to_row(fondo))
    log.debug("SQLite: upsert completado para ISIN %s", fondo.isin)


def _load_existing_identifiers(db_path: Path) -> tuple[set[str], set[str]]:
    """
    Carga los identificadores de los fondos ya presentes en funds.db, para
    poder omitirlos sin volver a llamar a Gemini.

    Devuelve (isins, nombres_pdf): el ISIN suele coincidir con el stem del
    PDF (el scraper nombra el archivo "{ISIN}.pdf", o "{NIF}.pdf" cuando no
    hay ISIN — ver extract_fund_data), y pdf_origen guarda el nombre de
    archivo exacto usado en la extracción anterior.
    """
    if not db_path.exists():
        return set(), set()
    with sqlite3.connect(db_path) as con:
        existe_tabla = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='funds'"
        ).fetchone()
        if existe_tabla is None:
            return set(), set()
        filas = con.execute("SELECT isin, pdf_origen FROM funds").fetchall()
    isins = {isin for isin, _ in filas if isin}
    nombres_pdf = {pdf_origen for _, pdf_origen in filas if pdf_origen}
    return isins, nombres_pdf


def _ya_extraido(pdf_path: Path, isins_existentes: set[str], nombres_existentes: set[str]) -> bool:
    """True si el PDF ya tiene fila en funds.db (por ISIN/NIF del nombre de archivo o por pdf_origen)."""
    return pdf_path.stem in isins_existentes or pdf_path.name in nombres_existentes


# ── Persistencia en ChromaDB ──────────────────────────────────────────────────

def _get_chroma_collection(chroma_path: Path) -> chromadb.Collection:
    """Devuelve (o crea) la colección ChromaDB para politica_inversion."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def save_to_chroma(fondo: FundModel, chroma_path: Path) -> None:
    """
    Indexa el campo politica_inversion en ChromaDB.
    ChromaDB genera el embedding internamente usando su modelo por defecto.
    El ISIN actúa como ID único del documento.
    """
    if not fondo.politica_inversion:
        log.debug("ChromaDB: sin politica_inversion para %s, omitiendo.", fondo.isin)
        return

    coleccion = _get_chroma_collection(chroma_path)
    coleccion.upsert(
        ids=[fondo.isin],
        documents=[fondo.politica_inversion],
        metadatas=[{
            "nombre_fondo": fondo.nombre_fondo,
            "gestora": fondo.gestora,
            "categoria": fondo.categoria or "",
            "nivel_riesgo": str(fondo.nivel_riesgo or ""),
            "esg": str(fondo.esg or ""),
        }],
    )
    log.debug("ChromaDB: documento indexado para ISIN %s", fondo.isin)


# ── Procesamiento de un solo PDF ─────────────────────────────────────────────

def process_pdf(
    pdf_path: Path,
    llm: ChatGoogleGenerativeAI,
    db_path: Path,
    chroma_path: Path,
    use_docling: bool = False,
) -> FundModel | None:
    """
    Procesa un único PDF: extracción de texto → LLM → SQLite → ChromaDB.
    Devuelve el FundModel resultante o None si ocurre un error irrecuperable.
    """
    log.info("Procesando: %s  [extractor=%s]", pdf_path.name, "docling" if use_docling else "pypdf")
    try:
        texto = extract_text(pdf_path, use_docling=use_docling)
        if not texto.strip():
            log.warning("%s: PDF sin texto extraíble (¿escaneado sin OCR?)", pdf_path.name)
            return None
        log.debug("%s: %d caracteres tras limpieza", pdf_path.name, len(texto))

        fondo = extract_fund_data(texto, pdf_path, llm)
        save_to_sqlite(fondo, db_path)
        save_to_chroma(fondo, chroma_path)

        log.info(
            "OK  %s | %s | riesgo=%s | ESG=%s",
            fondo.isin,
            fondo.nombre_fondo[:50],
            fondo.nivel_riesgo,
            fondo.esg,
        )
        return fondo

    except Exception as exc:
        log.error("Error procesando %s: %s", pdf_path.name, exc, exc_info=True)
        return None


# ── Orquestador ───────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: Path,
    db_path: Path,
    chroma_path: Path,
    workers: int,
    use_docling: bool = False,
) -> None:
    """
    Procesa un archivo individual o todos los PDFs de un directorio.
    Con workers > 1 usa un ThreadPoolExecutor (I/O-bound: esperas de red al LLM).
    """
    if input_path.is_file():
        pdfs = [input_path]
    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
    else:
        raise FileNotFoundError(f"No existe la ruta: {input_path}")

    if not pdfs:
        log.warning("No se encontraron PDFs en %s", input_path)
        return

    isins_existentes, nombres_existentes = _load_existing_identifiers(db_path)
    omitidos = [p for p in pdfs if _ya_extraido(p, isins_existentes, nombres_existentes)]
    pdfs = [p for p in pdfs if not _ya_extraido(p, isins_existentes, nombres_existentes)]
    for pdf in omitidos:
        log.info("Ya existe en BD, omitido: %s", pdf.name)

    if not pdfs:
        log.info("Todos los PDFs ya están en BD (%d omitidos). Nada que procesar.", len(omitidos))
        return

    log.info(
        "PDFs a procesar: %d (omitidos por ya existir en BD: %d) | workers: %d | modelo: %s | extractor: %s",
        len(pdfs), len(omitidos), workers, GEMINI_MODEL, "docling" if use_docling else "pypdf",
    )

    llm = _build_llm()
    ok = errores = 0

    if workers == 1:
        for pdf in pdfs:
            resultado = process_pdf(pdf, llm, db_path, chroma_path, use_docling=use_docling)
            if resultado:
                ok += 1
            else:
                errores += 1
    else:
        # Cada hilo necesita su propia instancia de LLM para evitar condiciones de carrera
        def _task(pdf: Path) -> bool:
            hilo_llm = _build_llm()
            return process_pdf(pdf, hilo_llm, db_path, chroma_path, use_docling=use_docling) is not None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futuros = {executor.submit(_task, pdf): pdf for pdf in pdfs}
            for futuro in as_completed(futuros):
                if futuro.result():
                    ok += 1
                else:
                    errores += 1

    log.info(
        "Pipeline finalizado. OK: %d | Errores: %d | Omitidos: %d | Total: %d",
        ok, errores, len(omitidos), len(pdfs) + len(omitidos),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae datos estructurados de DFIs y los almacena en SQLite + ChromaDB."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/dfi_pdfs"),
        metavar="PATH",
        help="Ruta a un PDF o a un directorio de PDFs (por defecto: data/dfi_pdfs).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        metavar="PATH",
        help=f"Ruta de la base de datos SQLite (por defecto: {DB_PATH}).",
    )
    parser.add_argument(
        "--chroma",
        type=Path,
        default=CHROMA_PATH,
        metavar="DIR",
        help=f"Directorio de ChromaDB (por defecto: {CHROMA_PATH}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Número de hilos paralelos para llamadas al LLM (por defecto: 1).",
    )
    parser.add_argument(
        "--docling",
        action="store_true",
        help=(
            "Usar Docling en lugar de pypdf para extraer texto. "
            "Genera Markdown estructurado con tablas, mejora la extracción "
            "del indicador de riesgo SRRI (1-7). Requiere: pip install docling"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activar logging en nivel DEBUG.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        input_path=args.input,
        db_path=args.db,
        chroma_path=args.chroma,
        workers=args.workers,
        use_docling=args.docling,
    )
