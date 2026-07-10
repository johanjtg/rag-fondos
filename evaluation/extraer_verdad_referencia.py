"""
Extracción de "verdad de referencia" para el golden dataset ampliado,
usando patrones de texto sobre el PDF original — SIN Gemini ni ningún LLM.

La idea es tener una fuente de verdad independiente del modelo que se
está evaluando (evaluation/evaluador_extraccion.py compara los datos de
Gemini en funds.db contra este golden dataset).

Campos que intenta extraer, y el patrón DFI/KID en el que se basan:
  - nivel_riesgo: "Hemos clasificado este producto en la clase de riesgo
    X en una escala de 7" (Indicador Resumido de Riesgo, sección
    obligatoria en todos los DFI).
  - horizonte_recomendado_anios: "Período/Periodo de mantenimiento
    recomendado: X años" (o "es de X años/meses"). Se descarta cuando el
    valor no es un número directo (p.ej. fondos a vencimiento que solo
    dan una fecha) para no inventar una cifra.
  - comision_gestion: fila "Comisiones de gestión y otros costes
    administrativos o de funcionamiento" de la tabla de costes corrientes.
  - comision_suscripcion / comision_reembolso: frases explícitas tipo "Los
    costes de entrada/salida son del X%" o "No cobramos comisión de
    entrada/salida" (→ 0).
  - esg: boilerplate SFDR — "no tienen en cuenta los criterios de la UE
    para las actividades económicas medioambientalmente sostenibles"
    (→ False) vs. "promueve características medioambientales" / mención
    de art. 8 o 9 del Reglamento (UE) 2019/2088 (→ True).
  - tipo_gestion: declaración explícita "fondo de gestión activa/pasiva",
    o en su defecto "replica/reproduce un índice" (→ pasiva) / "maximizar
    la rentabilidad", "superar al mercado" (→ activa).

Si ningún patrón encaja con confianza razonable, el campo queda en null
— no se inventan valores.

Uso:
  python -m evaluation.extraer_verdad_referencia
  python -m evaluation.extraer_verdad_referencia --muestra evaluation/muestra_golden_100.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from pypdf import PdfReader

from extraction.pdf_extractor import clean_text

MUESTRA_PATH = Path("evaluation/muestra_golden_100.json")
OUTPUT_PATH = Path("evaluation/golden_dataset_100_extraccion.json")
PDFS_DIR = Path("data/dfi_pdfs")

PCT_VERIFICACION_MANUAL = 0.18   # ~18% cae en el rango 15-20% pedido para 100 fondos
SEED_VERIFICACION = 42

CAMPOS = [
    "nivel_riesgo",
    "horizonte_recomendado_anios",
    "comision_gestion",
    "comision_suscripcion",
    "comision_reembolso",
    "esg",
    "tipo_gestion",
]


# ── Extracción de texto completo (sin el recorte de MAX_PDF_CHARS del pipeline) ─

def extraer_texto_completo(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    partes = [page.extract_text() for page in reader.pages if page.extract_text()]
    return clean_text("\n".join(partes), is_markdown=False)


# ── Patrones ──────────────────────────────────────────────────────────────────

RE_NIVEL_RIESGO = re.compile(
    r"(?:clase|nivel)\s+de\s+riesgo\s*[\[\(]?\s*(\d)\s*[\]\)]?\s*en\s*(?:una\s*)?escala\s+de\s+7",
    re.IGNORECASE,
)

RE_HORIZONTE_EXPLICITO = re.compile(
    r"per[íi]odo\s+de\s+manteni\w*\s*recomendado\s*es\s+de\s*(\d+(?:[.,]\d+)?)\s*(años?|anos?|mes(?:es)?)",
    re.IGNORECASE,
)
RE_HORIZONTE_ETIQUETA = re.compile(
    r"per[íi]odo\s+de\s+manteni\w*\s*recomendado[\s:]*(\d+(?:[.,]\d+)?)\s*(años?|anos?|mes(?:es)?)",
    re.IGNORECASE,
)

RE_GESTION_PCT = re.compile(
    r"gesti[óo]n\s+y\s+otros[\s\S]{0,120}?(\d+(?:[.,]\d+)?)\s*%",
    re.IGNORECASE,
)

RE_ENTRADA_CERO = re.compile(
    r"no\s+(?:se\s+)?cobra(?:mos)?\s+(?:una\s+)?comisi[óo]n\s+de\s+entrada"
    r"|no\s+hay\s+comisi[óo]n\s+de\s+entrada"
    r"|no\s+se\s+aplican\s+costes\s+de\s+entrada"
    r"|los\s+costes\s+de\s+entrada\s+son\s+del?\s*0\s*%",
    re.IGNORECASE,
)
RE_ENTRADA_PCT = re.compile(
    r"costes?\s+de\s+entrada[^%]{0,120}?(\d+(?:[.,]\d+)?)\s*%",
    re.IGNORECASE,
)

RE_SALIDA_CERO = re.compile(
    r"no\s+(?:se\s+)?(?:cobra(?:mos)?|aplica)\s+(?:una\s+)?comisi[óo]n\s+de\s+(?:salida|reembolso)"
    r"|no\s+hay\s+comisi[óo]n\s+de\s+salida"
    r"|no\s+se\s+aplican\s+costes\s+de\s+salida"
    r"|los\s+costes\s+de\s+salida\s+son\s+del?\s*0\s*%",
    re.IGNORECASE,
)
RE_SALIDA_PCT = re.compile(
    r"(?:costes?\s+de\s+salida|comisi[óo]n\s+de\s+reembolso)[^%]{0,120}?(\d+(?:[.,]\d+)?)\s*%",
    re.IGNORECASE,
)

RE_ESG_NEGATIVO = re.compile(
    r"(no\s+)tiene[n]?\s+en\s+cuenta\s+los\s+criterios\s+de\s+la\s+ue",
    re.IGNORECASE,
)
RE_ESG_POSITIVO = re.compile(
    r"promueve\s+caracter[íi]sticas\s+medioambientales"
    r"|art[íi]culo?\.?\s*8\s+reglamento|art[íi]culo?\.?\s*9\s+reglamento"
    r"|art\.?\s*8\s+reglamento|art\.?\s*9\s+reglamento",
    re.IGNORECASE,
)

RE_GESTION_EXPLICITA_ACTIVA = re.compile(
    r"(?:fondo|compartimento|se\s+trata)\s+de\s+gesti[óo]n\s+activa|es\s+de\s+gesti[óo]n\s+activa",
    re.IGNORECASE,
)
RE_GESTION_EXPLICITA_PASIVA = re.compile(
    r"(?:fondo|compartimento|se\s+trata)\s+de\s+gesti[óo]n\s+pasiva|es\s+de\s+gesti[óo]n\s+pasiva",
    re.IGNORECASE,
)
RE_REPLICA_INDICE = re.compile(
    r"replicar?\s+(?:la\s+rentabilidad\s+de\s+)?(?:un|el)\s+[íi]ndice"
    r"|reproduc\w+\s+(?:la\s+rentabilidad\s+de\s+)?(?:un|el)\s+[íi]ndice",
    re.IGNORECASE,
)
RE_MAXIMIZAR = re.compile(
    r"maximizar\s+la\s+rentabilidad|superar\s+(?:a\s+)?(?:el\s+|la\s+)?(?:mercado|[íi]ndice|benchmark)"
    r"|obtener\s+una\s+rentabilidad\s+superior",
    re.IGNORECASE,
)


def _a_float(texto: str) -> float:
    return float(texto.replace(",", "."))


# ── Extractores por campo ────────────────────────────────────────────────────

def extraer_nivel_riesgo(texto: str) -> int | None:
    m = RE_NIVEL_RIESGO.search(texto)
    return int(m.group(1)) if m else None


def extraer_horizonte(texto: str) -> float | int | None:
    m = RE_HORIZONTE_EXPLICITO.search(texto) or RE_HORIZONTE_ETIQUETA.search(texto)
    if not m:
        return None
    valor = _a_float(m.group(1))
    unidad = m.group(2).lower()
    anios = valor / 12 if unidad.startswith("mes") else valor
    return int(anios) if anios == int(anios) else round(anios, 2)


def extraer_comision_gestion(texto: str) -> float | None:
    m = RE_GESTION_PCT.search(texto)
    if m and "entrada" not in m.group(0).lower() and "salida" not in m.group(0).lower():
        return _a_float(m.group(1))
    return None


def _parece_comision_anual(fragmento: str) -> bool:
    """Detecta si el % capturado en realidad pertenece a la fila de comisión de
    gestión (recurrente) que se coló en la ventana, no a un coste de entrada/salida
    (que es un cobro único, sin la anotación 'al año' / 'cada año')."""
    f = fragmento.lower()
    return "al año" in f or "cada año" in f or "por año" in f


def extraer_comision_suscripcion(texto: str) -> float | None:
    if RE_ENTRADA_CERO.search(texto):
        return 0.0
    m = RE_ENTRADA_PCT.search(texto)
    if m and "salida" not in m.group(0).lower() and not _parece_comision_anual(m.group(0)):
        return _a_float(m.group(1))
    return None


def extraer_comision_reembolso(texto: str) -> float | None:
    if RE_SALIDA_CERO.search(texto):
        return 0.0
    m = RE_SALIDA_PCT.search(texto)
    if m and not _parece_comision_anual(m.group(0)):
        return _a_float(m.group(1))
    return None


def extraer_esg(texto: str) -> bool | None:
    neg = RE_ESG_NEGATIVO.search(texto)
    if neg and neg.group(1):
        return False
    if RE_ESG_POSITIVO.search(texto):
        return True
    return None


def extraer_tipo_gestion(texto: str) -> str | None:
    if RE_GESTION_EXPLICITA_ACTIVA.search(texto):
        return "activa"
    if RE_GESTION_EXPLICITA_PASIVA.search(texto):
        return "pasiva"
    if RE_REPLICA_INDICE.search(texto):
        return "pasiva"
    if RE_MAXIMIZAR.search(texto):
        return "activa"
    return None


def extraer_campos(texto: str) -> dict:
    return {
        "nivel_riesgo": extraer_nivel_riesgo(texto),
        "horizonte_recomendado_anios": extraer_horizonte(texto),
        "comision_gestion": extraer_comision_gestion(texto),
        "comision_suscripcion": extraer_comision_suscripcion(texto),
        "comision_reembolso": extraer_comision_reembolso(texto),
        "esg": extraer_esg(texto),
        "tipo_gestion": extraer_tipo_gestion(texto),
    }


def _descripcion_caso(campos: dict) -> str:
    encontrados = [c for c in CAMPOS if campos.get(c) is not None]
    faltantes = [c for c in CAMPOS if campos.get(c) is None]
    partes = ["Verdad de referencia extraída por patrones de texto (sin LLM)."]
    if encontrados:
        partes.append(f"Campos encontrados: {', '.join(encontrados)}.")
    if faltantes:
        partes.append(f"Sin patrón (null): {', '.join(faltantes)}.")
    return " ".join(partes)


# ── Orquestador ───────────────────────────────────────────────────────────────

def construir_golden_dataset(muestra: list[dict], pdfs_dir: Path) -> list[dict]:
    resultados = []
    for fondo in muestra:
        pdf_path = pdfs_dir / fondo["pdf_origen"]
        if not pdf_path.exists():
            print(f"  AVISO: no existe el PDF {pdf_path}, se omite {fondo['isin']}")
            continue

        texto = extraer_texto_completo(pdf_path)
        campos = extraer_campos(texto)

        resultados.append({
            "isin": fondo["isin"],
            "nombre_fondo": fondo["nombre_fondo"],
            "descripcion_caso": _descripcion_caso(campos),
            "esperado": campos,
            "para_verificar_manualmente": False,
        })
    return resultados


def marcar_verificacion_manual(dataset: list[dict], pct: float, seed: int) -> list[dict]:
    n = round(len(dataset) * pct)
    n = max(15, min(20, n))   # forzar rango 15-20 pedido, aunque cambie el tamaño de la muestra
    rng = random.Random(seed)
    elegidos = rng.sample(range(len(dataset)), min(n, len(dataset)))
    for i in elegidos:
        dataset[i]["para_verificar_manualmente"] = True
    return dataset


# ── Resumen ───────────────────────────────────────────────────────────────────

def imprimir_resumen(dataset: list[dict]) -> None:
    sep = "─" * 72
    print("\n" + "═" * 72)
    print("  RESUMEN — GOLDEN DATASET AMPLIADO (100 fondos, patrones de texto)")
    print("═" * 72)

    print(f"\n  Fondos procesados: {len(dataset)}")
    print(f"\n  Relleno por campo (no-null / total):")
    print(sep)
    for campo in CAMPOS:
        rellenos = sum(1 for d in dataset if d["esperado"].get(campo) is not None)
        pct = rellenos / len(dataset) * 100 if dataset else 0
        barra = "█" * int(pct / 5)
        print(f"  {campo:<32} {rellenos:>3}/{len(dataset)}  ({pct:>5.1f}%)  {barra}")

    verificar = [d for d in dataset if d["para_verificar_manualmente"]]
    print(f"\n  Marcados para verificación manual: {len(verificar)}")
    print(sep)
    for d in sorted(verificar, key=lambda x: x["isin"]):
        print(f"  {d['isin']:<14} {d['nombre_fondo']}")
    print("═" * 72)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae verdad de referencia por patrones de texto (sin LLM) para el golden dataset ampliado."
    )
    parser.add_argument("--muestra", type=Path, default=MUESTRA_PATH)
    parser.add_argument("--pdfs", type=Path, default=PDFS_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--pct-verificacion", type=float, default=PCT_VERIFICACION_MANUAL)
    parser.add_argument("--seed-verificacion", type=int, default=SEED_VERIFICACION)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with args.muestra.open(encoding="utf-8") as f:
        muestra = json.load(f)

    dataset = construir_golden_dataset(muestra, args.pdfs)
    dataset = marcar_verificacion_manual(dataset, args.pct_verificacion, args.seed_verificacion)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {args.out}")

    imprimir_resumen(dataset)


if __name__ == "__main__":
    main()
