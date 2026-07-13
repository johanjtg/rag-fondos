"""
Motor de puntuación de fondos de inversión.

Implementa SCORING_FORMULA de chatbot/questions.py:

  score_final =
      0.60 * cosine_similarity(vector_usuario, vector_fondo)
    + 0.40 * semantic_similarity(tematica_usuario, politica_inversion_fondo)
    + 0.10 * esg_boost  (si sensibilidad_esg > 0.6 y fondo.esg == True)

Filtros duros aplicados ANTES del scoring:
  - fondo.importe_minimo_inversion > usuario.capital_disponible  → EXCLUIR
  - fondo.nivel_riesgo > usuario.nivel_riesgo_max                → EXCLUIR
  - fondo.horizonte_recomendado_anios > usuario.horizonte_anios  → EXCLUIR
  - fondo.restricciones_liquidez != None y necesidad_liquidez > 0.7 → EXCLUIR
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from extraction.fund_model import FundModel
from scoring.fund_vectorizer import vectorize
from scoring.user_profiler import UserProfile

log = logging.getLogger(__name__)

# ── Pesos de la fórmula ───────────────────────────────────────────────────────
W_COSINE   = 0.60
W_SEMANTIC = 0.40
W_ESG      = 0.10
ESG_THRESHOLD = 0.6          # sensibilidad_esg mínima para aplicar boost
LIQUIDEZ_UMBRAL = 0.7        # necesidad_liquidez a partir de la cual se filtra

DB_PATH     = Path("database/funds.db")
CHROMA_PATH = Path("database/chroma")
CHROMA_COLLECTION = "politica_inversion"


# ── Resultado de scoring ──────────────────────────────────────────────────────

@dataclass
class FundScore:
    """Fondo puntuado con desglose de componentes."""
    isin:             str
    nombre_fondo:     str
    gestora:          str
    nivel_riesgo:     int | None
    esg:              bool | None
    tipo_gestion:     str | None
    horizonte_anios:  int | None
    politica_inversion: str | None

    score_total:      float
    score_coseno:     float
    score_semantico:  float
    esg_boost:        float

    def __str__(self) -> str:
        return (
            f"{self.nombre_fondo} ({self.isin})\n"
            f"  Score: {self.score_total:.3f}  "
            f"[coseno={self.score_coseno:.3f}  "
            f"semántico={self.score_semantico:.3f}  "
            f"esg_boost={self.esg_boost:.2f}]\n"
            f"  Riesgo: {self.nivel_riesgo}/7 | "
            f"Gestión: {self.tipo_gestion or '—'} | "
            f"ESG: {self.esg} | "
            f"Horizonte: {self.horizonte_anios} años"
        )


# ── Carga de fondos desde SQLite ──────────────────────────────────────────────

def _row_to_fundmodel(row: sqlite3.Row) -> FundModel:
    """Convierte una fila SQLite en un FundModel deserializando los campos JSON."""
    d = dict(row)
    for campo in ("universo_activos", "distribucion_sectorial", "distribucion_geografica"):
        if d.get(campo):
            d[campo] = json.loads(d[campo])
    for campo in ("divisa_cobertura", "esg"):
        if d.get(campo) is not None:
            d[campo] = bool(d[campo])
    return FundModel(**d)


def load_all_funds(db_path: Path = DB_PATH) -> list[FundModel]:
    """Carga todos los fondos almacenados en SQLite."""
    if not db_path.exists():
        raise FileNotFoundError(f"Base de datos no encontrada: {db_path}")
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM funds").fetchall()
    log.info("Fondos cargados desde SQLite: %d", len(rows))
    return [_row_to_fundmodel(r) for r in rows]


# ── Filtros duros ─────────────────────────────────────────────────────────────

CAPITAL_MINIMO_DEFAULT = 30.0   # € — umbral preventivo cuando el fondo no informa su mínimo

def _pasa_filtros(fondo: FundModel, perfil: UserProfile) -> tuple[bool, str]:
    """
    Evalúa los filtros de exclusión duros.
    Devuelve (True, "") si el fondo pasa, o (False, motivo) si es excluido.

    Política de datos ausentes:
    - importe_minimo_inversion nulo → se aplica CAPITAL_MINIMO_DEFAULT (30 €).
      Si el capital del usuario es inferior a ese umbral, el fondo se excluye
      preventivamente en lugar de omitir la comprobación.
    - horizonte_recomendado_anios nulo → se excluye preventivamente cuando el
      usuario declara un horizonte temporal definido (> 0), ya que no es posible
      verificar la compatibilidad.
    """
    # Capital mínimo
    if perfil.capital_disponible > 0:
        minimo = (
            fondo.importe_minimo_inversion
            if fondo.importe_minimo_inversion is not None
            else CAPITAL_MINIMO_DEFAULT
        )
        if perfil.capital_disponible < minimo:
            return False, (
                f"importe_minimo ({minimo:.0f}€) "
                f"> capital ({perfil.capital_disponible:.0f}€)"
            )

    # Nivel de riesgo
    if (
        fondo.nivel_riesgo is not None
        and fondo.nivel_riesgo > perfil.nivel_riesgo_max
    ):
        return False, f"nivel_riesgo ({fondo.nivel_riesgo}) > max ({perfil.nivel_riesgo_max})"

    # Horizonte temporal
    if perfil.horizonte_anios > 0:
        if fondo.horizonte_recomendado_anios is None:
            return False, "horizonte_fondo desconocido — excluido preventivamente"
        if fondo.horizonte_recomendado_anios > perfil.horizonte_anios:
            return False, (
                f"horizonte_fondo ({fondo.horizonte_recomendado_anios}a) "
                f"> horizonte_usuario ({perfil.horizonte_anios:.1f}a)"
            )

    # Restricciones de liquidez
    if (
        fondo.restricciones_liquidez
        and fondo.restricciones_liquidez.strip()
        and perfil.necesidad_liquidez > LIQUIDEZ_UMBRAL
    ):
        return False, "fondo con restricciones de liquidez pero usuario necesita liquidez alta"

    return True, ""


# ── Similitud semántica vía ChromaDB ─────────────────────────────────────────

def _semantic_scores(
    query: str,
    isins: list[str],
    chroma_path: Path = CHROMA_PATH,
) -> dict[str, float]:
    """
    Consulta ChromaDB con el texto temático del usuario y devuelve
    un dict {isin: similitud_coseno} para los fondos solicitados.

    ChromaDB devuelve distancias coseno [0, 2]; convertimos a similitud [0, 1]
    con: similitud = 1 - distancia / 2.
    """
    if not query.strip() or not isins:
        return {isin: 0.0 for isin in isins}

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        coleccion = client.get_collection(CHROMA_COLLECTION, embedding_function=embedding_fn)
    except Exception:
        log.warning("Colección ChromaDB '%s' no encontrada. Similitud semántica = 0.", CHROMA_COLLECTION)
        return {isin: 0.0 for isin in isins}

    resultados = coleccion.query(
        query_texts=[query],
        n_results=min(len(isins), coleccion.count() or 1),
        include=["distances"],
    )

    scores: dict[str, float] = {isin: 0.0 for isin in isins}
    for isin_resultado, distancia in zip(
        resultados["ids"][0], resultados["distances"][0]
    ):
        if isin_resultado in scores:
            scores[isin_resultado] = round(1.0 - distancia / 2.0, 4)

    return scores


# ── Motor de puntuación ───────────────────────────────────────────────────────

def score_funds(
    perfil: UserProfile,
    fondos: list[FundModel],
    top_n: int = 5,
    chroma_path: Path = CHROMA_PATH,
) -> list[FundScore]:
    """
    Aplica filtros duros y calcula el score final para cada fondo.

    Pasos:
      1. Filtrar fondos que no pasan los filtros duros.
      2. Calcular similitud coseno entre el vector del usuario y el de cada fondo.
      3. Calcular similitud semántica RAG desde ChromaDB.
      4. Combinar con SCORING_FORMULA y ordenar.
      5. Devolver los top_n mejores.
    """
    vector_usuario = perfil.to_vector().reshape(1, -1)

    # ── 1. Filtros duros ──────────────────────────────────────────────────────
    candidatos: list[FundModel] = []
    excluidos = 0
    for fondo in fondos:
        pasa, motivo = _pasa_filtros(fondo, perfil)
        if pasa:
            candidatos.append(fondo)
        else:
            log.debug("EXCLUIDO %s: %s", fondo.isin, motivo)
            excluidos += 1

    log.info(
        "Filtros duros: %d fondos pasan de %d (excluidos: %d)",
        len(candidatos), len(fondos), excluidos,
    )

    if not candidatos:
        log.warning("Ningún fondo supera los filtros duros con el perfil dado.")
        return []

    # ── 2. Similitud coseno ───────────────────────────────────────────────────
    vectores_fondos = np.array([vectorize(f) for f in candidatos])
    cosenos = cosine_similarity(vector_usuario, vectores_fondos)[0]  # shape (n,)

    # ── 3. Similitud semántica (RAG) ──────────────────────────────────────────
    geo     = perfil.preferencia_geografica.strip()
    sector  = perfil.preferencia_sectorial.strip()
    if sector and geo:
        query_semantica = (
            f"fondo que invierte en el sector {sector} "
            f"con exposición geográfica a {geo}"
        )
    elif sector:
        query_semantica = f"fondo especializado en el sector {sector}"
    elif geo:
        query_semantica = f"fondo que invierte en {geo}"
    else:
        query_semantica = ""
    isins_candidatos = [f.isin for f in candidatos]
    semanticos = _semantic_scores(query_semantica, isins_candidatos, chroma_path)

    # ── 4. Scoring final ──────────────────────────────────────────────────────
    resultados: list[FundScore] = []
    for i, fondo in enumerate(candidatos):
        coseno = float(cosenos[i])
        semantico = semanticos.get(fondo.isin, 0.0)

        esg_boost = 0.0
        if perfil.sensibilidad_esg > ESG_THRESHOLD and fondo.esg is True:
            esg_boost = W_ESG

        score_total = W_COSINE * coseno + W_SEMANTIC * semantico + esg_boost
        score_total = round(min(score_total, 1.0), 4)

        resultados.append(FundScore(
            isin=fondo.isin,
            nombre_fondo=fondo.nombre_fondo,
            gestora=fondo.gestora,
            nivel_riesgo=fondo.nivel_riesgo,
            esg=fondo.esg,
            tipo_gestion=fondo.tipo_gestion,
            horizonte_anios=fondo.horizonte_recomendado_anios,
            politica_inversion=fondo.politica_inversion,
            score_total=score_total,
            score_coseno=coseno,
            score_semantico=semantico,
            esg_boost=esg_boost,
        ))

    resultados.sort(key=lambda r: r.score_total, reverse=True)
    return resultados[:top_n]


# ── API pública de alto nivel ─────────────────────────────────────────────────

def get_top_funds(
    perfil: UserProfile,
    top_n: int = 5,
    db_path: Path = DB_PATH,
    chroma_path: Path = CHROMA_PATH,
) -> list[FundScore]:
    """
    Punto de entrada principal para el chatbot.
    Carga fondos de SQLite, puntúa contra el perfil y devuelve los top_n.
    """
    fondos = load_all_funds(db_path)
    return score_funds(perfil, fondos, top_n=top_n, chroma_path=chroma_path)
