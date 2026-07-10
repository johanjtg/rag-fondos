"""
Scraper del portal CNMV para descargar DFIs (Documentos de Datos Fundamentales).

Fuente: https://www.cnmv.es/portal/consultas/mostrarlistados?id=3&lang=es
Uso:    python scraper/cnmv_scraper.py [--max N] [--out DIR] [--resume] [--random [--seed N]]
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ── Constantes ───────────────────────────────────────────────────────────────

BASE_URL = "https://www.cnmv.es"
LIST_URL = (
    "https://www.cnmv.es/portal/consultas/mostrarlistados.aspx"
    "?id=3&lang=es&page={page}"
)
FUND_URL = "https://www.cnmv.es/portal/consultas/iic/fondo.aspx?nif={nif}"

# Documentos que se han observado ocupando por error el hueco del DFI/KID
# (p.ej. cuando el fallback posicional cogía el único enlace disponible).
DOC_INVALIDO_MARCADORES: list[tuple[str, str]] = [
    ("INFORMACIÓN PRECONTRACTUAL DE LOS PRODUCTOS FINANCIEROS", "anexo SFDR Art. 8, no es el DFI"),
    ("REGLAMENTO DE GESTIÓN DE", "reglamento de gestión del fondo, no es el DFI"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.cnmv.es/portal/consultas/mostrarlistados?id=3&lang=es",
}

ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b")
RATE_LIMIT_SECS = 1.0      # pausa mínima entre peticiones
MAX_RETRIES = 3
BACKOFF_BASE = 2           # segundos de espera × 2^intento en caso de error

MANIFEST_FIELDS = ["fund_name", "nif", "isin", "pdf_url", "pdf_path", "download_date"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Sesión HTTP ──────────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """Crea una sesión HTTP con cabeceras de navegador y cookies iniciales."""
    session = requests.Session()
    session.headers.update(HEADERS)
    # Primera petición al portal para obtener cookies de sesión ASP.NET
    try:
        session.get(
            "https://www.cnmv.es/portal/consultas/mostrarlistados?id=3&lang=es",
            timeout=15,
        )
    except requests.RequestException as exc:
        log.warning("No se pudieron obtener cookies iniciales: %s", exc)
    return session


def _get(session: requests.Session, url: str) -> requests.Response | None:
    """Realiza una petición GET con reintentos y backoff exponencial."""
    for intento in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                return resp
            log.warning("HTTP %s en %s (intento %d/%d)", resp.status_code, url, intento, MAX_RETRIES)
        except requests.RequestException as exc:
            log.warning("Error de red en %s (intento %d/%d): %s", url, intento, MAX_RETRIES, exc)
        if intento < MAX_RETRIES:
            espera = BACKOFF_BASE ** intento
            log.debug("Reintentando en %ds…", espera)
            time.sleep(espera)
    return None


# ── Parseo de la página de listado ───────────────────────────────────────────

def _parse_total_pages(soup: BeautifulSoup) -> int:
    """Extrae el número total de páginas del indicador 'Página X de Y'."""
    texto = soup.get_text(" ", strip=True)
    match = re.search(r"P[áa]gina\s+\d+\s+de\s+(\d+)", texto, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fallback: contar enlaces de paginación numéricos
    pag_links = soup.select("a[href*='page=']")
    nums = [
        int(m.group(1))
        for a in pag_links
        if (m := re.search(r"page=(\d+)", a.get("href", "")))
    ]
    return max(nums, default=0) + 1  # page es 0-indexado


def _parse_fund_list(soup: BeautifulSoup) -> list[dict]:
    """
    Extrae los fondos listados en una página del índice CNMV.

    Devuelve lista de dicts con claves: fund_name, nif, registro.

    Nota: el HTML renderiza cada fondo dos veces (vista mobile + desktop).
    Se deduplica por NIF manteniendo el orden de aparición.
    """
    fondos: list[dict] = []
    vistos: set[str] = set()

    for li in soup.select("ul li"):
        enlace = li.find("a", href=re.compile(r"iic/fondo", re.I))
        if not enlace:
            continue

        href = enlace.get("href", "")
        nif_match = re.search(r"nif=([^&\s]+)", href, re.I)
        if not nif_match:
            continue

        nif = nif_match.group(1).strip()
        if nif in vistos:
            continue
        vistos.add(nif)

        nombre = enlace.get_text(strip=True)

        # Número de registro (si aparece en el <li>)
        texto_li = li.get_text(" ", strip=True)
        reg_match = re.search(r"registro oficial[:\s]+([\d]+)", texto_li, re.I)
        registro = reg_match.group(1) if reg_match else None

        fondos.append({"fund_name": nombre, "nif": nif, "registro": registro})

    return fondos


# ── Parseo de la página de detalle del fondo ─────────────────────────────────

def _find_classes_view_url(soup: BeautifulSoup, nif: str) -> str | None:
    """
    Localiza, en la barra de navegación de la página de detalle, el enlace a
    la vista por unidad de inversión: "Clases de participaciones sin
    compartimentos" o "Compartimentos", según la estructura del fondo. Ahí es
    donde CNMV publica el ISIN y el DFI (*) fila a fila cuando el fondo tiene
    varias unidades.
    """
    nav = soup.find("div", class_="NavegApdo")
    if not nav:
        return None
    enlace = nav.find("a", id=re.compile(r"hlParticipaciones|hlCompartimentos", re.I))
    if enlace and enlace.get("href"):
        # El href es relativo al directorio de fondo.aspx, no al dominio raíz.
        return urljoin(FUND_URL.format(nif=nif), enlace["href"])
    return None


def _parse_dfi_from_classes_view(session: requests.Session, soup: BeautifulSoup, nif: str) -> tuple[str | None, str | None]:
    """
    Busca ISIN + DFI en la vista por unidad de inversión del fondo (clases de
    participación o compartimentos). Los fondos con varias unidades no llevan
    estas columnas en la vista por defecto; CNMV las publica una fila por
    unidad en esta vista aparte.

    Devuelve (isin, pdf_url) de la primera unidad con enlace a DFI, o (None, None).
    """
    url_clases = _find_classes_view_url(soup, nif)
    if url_clases is None:
        return None, None

    time.sleep(RATE_LIMIT_SECS)
    resp = _get(session, url_clases)
    if resp is None:
        return None, None

    soup_clases = BeautifulSoup(resp.text, "html.parser")
    for fila in soup_clases.find_all("tr"):
        celda_dfi = fila.find("td", attrs={"data-th": re.compile(r"DFI", re.I)})
        if not celda_dfi:
            continue
        enlace_dfi = celda_dfi.find("a", href=True)
        if not enlace_dfi:
            continue
        celda_isin = fila.find("td", attrs={"data-th": "ISIN"})
        isin = celda_isin.get_text(strip=True) if celda_isin else None
        return isin, urljoin(BASE_URL, enlace_dfi["href"])

    return None, None


def _parse_fund_detail(session: requests.Session, soup: BeautifulSoup, nif: str) -> dict:
    """
    Extrae el ISIN y la URL del PDF DFI desde la página de detalle del fondo.

    La página renderiza una tabla con id 'gridDatos' cuyas celdas llevan el
    atributo data-th con el nombre de la columna:
      data-th="ISIN"    → enlace con el código ISIN
      data-th="DFI (*)" → enlace al PDF del Documento de Datos Fundamentales

    En fondos con varias clases o compartimentos estas columnas no están en
    la vista por defecto, así que se consulta la vista de clases/compartimentos
    (ver `_parse_dfi_from_classes_view`) antes de recurrir al fallback
    posicional.
    """
    isin: str | None = None
    pdf_url: str | None = None

    # ── Estrategia principal: tabla con data-th ───────────────────────────────
    celda_isin = soup.find("td", attrs={"data-th": "ISIN"})
    if celda_isin:
        enlace_isin = celda_isin.find("a")
        if enlace_isin:
            isin = enlace_isin.get_text(strip=True)

    celda_dfi = soup.find("td", attrs={"data-th": re.compile(r"DFI", re.I)})
    if celda_dfi:
        enlace_dfi = celda_dfi.find("a", href=True)
        if enlace_dfi:
            pdf_url = urljoin(BASE_URL, enlace_dfi["href"])

    # ── Fallback 1: buscar celda con texto "DFI" o "KID" en su columna ────────
    if not pdf_url:
        for td in soup.find_all("td"):
            td_text = (td.get("data-th", "") + " " + td.get_text(" ")).lower()
            if "dfi" in td_text or "kid" in td_text:
                enlace = td.find("a", href=True)
                if enlace and "verdocumento" in enlace["href"]:
                    pdf_url = urljoin(BASE_URL, enlace["href"])
                    break

    # ── Fallback 2: buscar enlace cuyo texto visible sea "DFI" o "KID" ─────────
    if not pdf_url:
        for a in soup.find_all("a", href=True):
            if "verdocumento" in a["href"] and re.search(r"\bDFI\b|\bKID\b", a.get_text(), re.I):
                pdf_url = urljoin(BASE_URL, a["href"])
                break

    # ── Fallback 3: fondo multi-clase/compartimentos — el DFI vive en esa vista ─
    if not pdf_url:
        isin_clase, pdf_url_clase = _parse_dfi_from_classes_view(session, soup, nif)
        if pdf_url_clase:
            pdf_url = pdf_url_clase
            isin = isin_clase or isin

    # ── Fallback 4: posicional — último recurso, sin garantía de acierto ──────
    if not pdf_url:
        doc_links = [
            a["href"] for a in soup.find_all("a", href=True)
            if "verdocumento" in a["href"]
        ]
        if len(doc_links) >= 3:
            log.warning(
                "NIF %s: no se pudo confirmar el DFI por columnas ni por vista "
                "de clases — usando fallback posicional (doc_links[1]), revisar manualmente",
                nif,
            )
            pdf_url = urljoin(BASE_URL, doc_links[1])   # posición DFI en orden habitual
        elif len(doc_links) == 2:
            log.warning("NIF %s: solo 2 documentos, probablemente sin DFI — omitiendo", nif)
            # No asignar pdf_url: es más probable que sean Folleto + Reglamento
        elif doc_links:
            log.warning(
                "NIF %s: no se pudo confirmar el DFI por columnas ni por vista "
                "de clases — usando único documento disponible como fallback posicional, revisar manualmente",
                nif,
            )
            pdf_url = urljoin(BASE_URL, doc_links[0])

    if not pdf_url:
        log.warning("NIF %s: no se encontró ningún enlace a documento", nif)

    # ── Fallback ISIN: regex sobre texto completo ─────────────────────────────
    if not isin:
        isins = ISIN_RE.findall(soup.get_text(" "))
        isin = isins[0] if isins else None

    return {"isin": isin, "pdf_url": pdf_url}


# ── Descarga del PDF ──────────────────────────────────────────────────────────

def _download_pdf(
    session: requests.Session,
    pdf_url: str,
    dest_dir: Path,
    filename: str,
) -> Path | None:
    """
    Descarga un PDF desde pdf_url y lo guarda en dest_dir/filename.
    Devuelve la ruta guardada o None si falla.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists():
        log.debug("PDF ya existe, omitiendo: %s", dest_path.name)
        return dest_path

    resp = _get(session, pdf_url)
    if resp is None:
        return None

    content_type = resp.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not pdf_url.lower().endswith(".pdf"):
        log.warning("Respuesta no parece PDF (Content-Type: %s): %s", content_type, pdf_url)

    dest_path.write_bytes(resp.content)
    log.info("PDF descargado: %s (%.1f KB)", dest_path.name, len(resp.content) / 1024)
    return dest_path


def _validar_pdf_es_dfi(pdf_path: Path) -> tuple[bool, str | None]:
    """
    Comprueba, tras la descarga, que el PDF no sea uno de los documentos que
    se han observado ocupando por error el hueco del DFI (anexo SFDR Art. 8,
    Reglamento de gestión). No confirma que SEA el DFI, solo descarta esos
    falsos positivos ya detectados.
    """
    try:
        texto = (PdfReader(str(pdf_path)).pages[0].extract_text() or "").upper()
    except Exception as exc:
        log.warning("No se pudo validar el contenido de %s: %s", pdf_path.name, exc)
        return True, None

    for marcador, motivo in DOC_INVALIDO_MARCADORES:
        if marcador in texto:
            return False, motivo
    return True, None


# ── Manifest CSV ──────────────────────────────────────────────────────────────

def _load_manifest(manifest_path: Path) -> set[str]:
    """Carga los NIFs ya descargados desde el manifest para permitir --resume."""
    if not manifest_path.exists():
        return set()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["nif"] for row in reader if row.get("nif")}


def _append_manifest(manifest_path: Path, row: dict) -> None:
    """Añade una fila al manifest CSV (crea la cabecera si es archivo nuevo)."""
    needs_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


# ── Orquestador principal ─────────────────────────────────────────────────────

def _procesar_fondos_pagina(
    session: requests.Session,
    fondos_pagina: list[dict],
    output_dir: Path,
    manifest_path: Path,
    resume: bool,
    ya_descargados: set[str],
    max_funds: int | None,
    total_descargados: int,
) -> int:
    """
    Descarga el DFI de cada fondo de `fondos_pagina` y anota el manifest.

    Para cada fondo: visita el detalle → extrae ISIN + URL del PDF → descarga
    → valida que no sea un falso positivo → registra en el manifest CSV.
    Devuelve el `total_descargados` actualizado.
    """
    for fondo in fondos_pagina:
        if max_funds and total_descargados >= max_funds:
            break

        nif = fondo["nif"]
        nombre = fondo["fund_name"]

        if resume and nif in ya_descargados:
            log.debug("Omitiendo (ya descargado): %s", nombre)
            continue

        time.sleep(RATE_LIMIT_SECS)

        # ── Detalle del fondo ─────────────────────────────────────────
        url_detalle = FUND_URL.format(nif=nif)
        resp_detalle = _get(session, url_detalle)

        isin = None
        pdf_url = None
        pdf_path_str = ""

        if resp_detalle:
            detalle = BeautifulSoup(resp_detalle.text, "html.parser")
            info = _parse_fund_detail(session, detalle, nif)
            isin = info["isin"]
            pdf_url = info["pdf_url"]
        else:
            log.warning("No se pudo obtener detalle de %s (%s)", nombre, nif)

        # ── Descarga PDF ──────────────────────────────────────────────
        if pdf_url:
            time.sleep(RATE_LIMIT_SECS)
            # Nombre de archivo: ISIN si disponible, si no usar NIF
            safe_id = isin if isin else nif
            filename = f"{safe_id}.pdf"
            descargado = _download_pdf(session, pdf_url, output_dir, filename)
            if descargado:
                es_valido, motivo = _validar_pdf_es_dfi(descargado)
                if not es_valido:
                    log.warning(
                        "PDF descartado para %s (%s): %s — %s",
                        nombre, nif, descargado.name, motivo,
                    )
                    descargado.unlink(missing_ok=True)
                    descargado = None
            pdf_path_str = str(descargado) if descargado else ""
        else:
            log.warning("Sin enlace PDF para %s (%s)", nombre, nif)

        # ── Manifest ──────────────────────────────────────────────────
        _append_manifest(
            manifest_path,
            {
                "fund_name": nombre,
                "nif": nif,
                "isin": isin or "",
                "pdf_url": pdf_url or "",
                "pdf_path": pdf_path_str,
                "download_date": date.today().isoformat(),
            },
        )

        total_descargados += 1
        log.info(
            "[%d] %s | ISIN: %s | PDF: %s",
            total_descargados,
            nombre,
            isin or "—",
            "OK" if pdf_path_str else "NO",
        )

    return total_descargados


def _run_scraper_secuencial(
    session: requests.Session,
    max_funds: int | None,
    output_dir: Path,
    manifest_path: Path,
    resume: bool,
    ya_descargados: set[str],
) -> int:
    """Recorre las páginas del listado CNMV en orden estricto (1, 2, 3…)."""
    total_descargados = 0
    pagina = 0

    while True:
        if max_funds and total_descargados >= max_funds:
            break

        url_pagina = LIST_URL.format(page=pagina)
        log.info("── Página %d ──────────────────────", pagina + 1)
        resp = _get(session, url_pagina)
        if resp is None:
            log.error("No se pudo obtener la página %d. Abortando.", pagina)
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Detectar número total de páginas en la primera iteración
        if pagina == 0:
            total_paginas = _parse_total_pages(soup)
            log.info("Total de páginas detectadas: %d", total_paginas)

        fondos_pagina = _parse_fund_list(soup)
        if not fondos_pagina:
            log.info("Sin fondos en página %d. Fin del listado.", pagina)
            break

        total_descargados = _procesar_fondos_pagina(
            session, fondos_pagina, output_dir, manifest_path,
            resume, ya_descargados, max_funds, total_descargados,
        )

        pagina += 1

        # Salida limpia si hemos llegado al final de las páginas
        if pagina >= total_paginas:
            log.info("Listado completo procesado (%d páginas).", total_paginas)
            break

    return total_descargados


def _run_scraper_aleatorio(
    session: requests.Session,
    max_funds: int | None,
    output_dir: Path,
    manifest_path: Path,
    resume: bool,
    ya_descargados: set[str],
    seed: int | None,
) -> int:
    """
    Recorre páginas del listado CNMV en orden aleatorio hasta alcanzar
    `max_funds` (o agotar todas las páginas si no se indica límite).

    Pide primero la página 0 para conocer el total de páginas, baraja el
    rango completo [0, total_paginas) y va pidiendo páginas por ese orden
    (salto directo vía `page=N`, sin recorrer las intermedias).
    """
    url_pagina0 = LIST_URL.format(page=0)
    resp0 = _get(session, url_pagina0)
    if resp0 is None:
        log.error("No se pudo obtener la página 0. Abortando.")
        return 0

    soup0 = BeautifulSoup(resp0.text, "html.parser")
    total_paginas = _parse_total_pages(soup0)
    log.info("Total de páginas detectadas: %d", total_paginas)

    orden_paginas = list(range(total_paginas))
    rng = random.Random(seed)
    rng.shuffle(orden_paginas)
    log.info(
        "Modo aleatorio activado (seed=%s): %d páginas en orden aleatorio",
        seed, len(orden_paginas),
    )

    soups_cacheadas = {0: soup0}
    total_descargados = 0

    for i, pagina in enumerate(orden_paginas, start=1):
        if max_funds and total_descargados >= max_funds:
            break

        log.info(
            "── Página %d (aleatoria, %d/%d) ──────────────────────",
            pagina + 1, i, len(orden_paginas),
        )

        soup = soups_cacheadas.get(pagina)
        if soup is None:
            resp = _get(session, LIST_URL.format(page=pagina))
            if resp is None:
                log.warning("No se pudo obtener la página %d. Saltando.", pagina)
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

        fondos_pagina = _parse_fund_list(soup)
        if not fondos_pagina:
            log.info("Sin fondos en página %d. Saltando.", pagina)
            continue

        total_descargados = _procesar_fondos_pagina(
            session, fondos_pagina, output_dir, manifest_path,
            resume, ya_descargados, max_funds, total_descargados,
        )

    return total_descargados


def run_scraper(
    max_funds: int | None,
    output_dir: Path,
    manifest_path: Path,
    resume: bool,
    random_order: bool = False,
    seed: int | None = None,
) -> None:
    """
    Bucle principal del scraper:
      1. Itera páginas del listado CNMV (secuencial o aleatorio según `random_order`).
      2. Para cada fondo visita la página de detalle → extrae ISIN + URL del PDF.
      3. Descarga el PDF.
      4. Guarda entrada en el manifest CSV.
    """
    session = _build_session()
    ya_descargados = _load_manifest(manifest_path) if resume else set()
    if ya_descargados:
        log.info("Reanudando: %d fondos ya descargados", len(ya_descargados))

    if random_order:
        total_descargados = _run_scraper_aleatorio(
            session, max_funds, output_dir, manifest_path, resume, ya_descargados, seed,
        )
    else:
        total_descargados = _run_scraper_secuencial(
            session, max_funds, output_dir, manifest_path, resume, ya_descargados,
        )

    log.info("Scraper finalizado. Fondos procesados: %d", total_descargados)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga DFIs de fondos de inversión desde el portal CNMV."
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Número máximo de fondos a descargar (por defecto: todos).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/dfi_pdfs"),
        metavar="DIR",
        help="Directorio de destino para los PDFs (por defecto: data/dfi_pdfs).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        metavar="CSV",
        help="Ruta del manifest CSV (por defecto: data/manifest.csv).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Saltar fondos ya presentes en el manifest (reanuda una descarga previa).",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help=(
            "Seleccionar fondos recorriendo las páginas del listado en orden "
            "aleatorio (en vez de secuencial 1, 2, 3…) hasta alcanzar --max."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Semilla para el orden aleatorio de páginas (solo con --random; por defecto: no determinista).",
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

    run_scraper(
        max_funds=args.max,
        output_dir=args.out,
        manifest_path=args.manifest,
        resume=args.resume,
        random_order=args.random,
        seed=args.seed,
    )
