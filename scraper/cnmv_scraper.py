"""
Scraper del portal CNMV para descargar DFIs (Documentos de Datos Fundamentales).

Fuente: https://www.cnmv.es/portal/consultas/mostrarlistados?id=3&lang=es
Uso:    python scraper/cnmv_scraper.py [--max N] [--out DIR] [--resume]
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ── Constantes ───────────────────────────────────────────────────────────────

BASE_URL = "https://www.cnmv.es"
LIST_URL = (
    "https://www.cnmv.es/portal/consultas/mostrarlistados.aspx"
    "?id=3&lang=es&page={page}"
)
FUND_URL = "https://www.cnmv.es/portal/consultas/iic/fondo.aspx?nif={nif}"

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

def _parse_fund_detail(soup: BeautifulSoup, nif: str) -> dict:
    """
    Extrae el ISIN y la URL del PDF DFI desde la página de detalle del fondo.

    La página renderiza una tabla con id 'gridDatos' cuyas celdas llevan el
    atributo data-th con el nombre de la columna:
      data-th="ISIN"    → enlace con el código ISIN
      data-th="DFI (*)" → enlace al PDF del Documento de Datos Fundamentales

    Fallback: si la tabla no está, busca el primer enlace a verdocumento
    que no sea Folleto ni Reglamento (posición 1 de 3).
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

    # ── Fallback 1: buscar celda con texto "DFI" o "KID" en su columna ──────────
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

    # ── Fallback 3: posicional (Folleto/DFI/Reglamento) — última opción ────────
    if not pdf_url:
        doc_links = [
            a["href"] for a in soup.find_all("a", href=True)
            if "verdocumento" in a["href"]
        ]
        if len(doc_links) >= 3:
            pdf_url = urljoin(BASE_URL, doc_links[1])   # posición DFI en orden habitual
        elif len(doc_links) == 2:
            log.warning("NIF %s: solo 2 documentos, probablemente sin DFI — omitiendo", nif)
            # No asignar pdf_url: es más probable que sean Folleto + Reglamento
        elif doc_links:
            log.warning("NIF %s: sin columna DFI — usando único documento disponible", nif)
            pdf_url = urljoin(BASE_URL, doc_links[0])

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

def run_scraper(
    max_funds: int | None,
    output_dir: Path,
    manifest_path: Path,
    resume: bool,
) -> None:
    """
    Bucle principal del scraper:
      1. Itera páginas del listado CNMV.
      2. Para cada fondo visita la página de detalle → extrae ISIN + URL del PDF.
      3. Descarga el PDF.
      4. Guarda entrada en el manifest CSV.
    """
    session = _build_session()
    ya_descargados = _load_manifest(manifest_path) if resume else set()
    if ya_descargados:
        log.info("Reanudando: %d fondos ya descargados", len(ya_descargados))

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
                info = _parse_fund_detail(detalle, nif)
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

        pagina += 1

        # Salida limpia si hemos llegado al final de las páginas
        if pagina >= total_paginas:
            log.info("Listado completo procesado (%d páginas).", total_paginas)
            break

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
    )
