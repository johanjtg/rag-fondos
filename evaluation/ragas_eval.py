"""
Evaluación RAGAS del sistema RAG de fondos de inversión.

Compara dos modos para las mismas 10 preguntas:
  - RAG:      pregunta + contexto recuperado de ChromaDB → Gemini genera respuesta
  - Base LLM: pregunta sin contexto                     → Gemini genera respuesta

Métricas evaluadas (RAGAS 0.4):
  - faithfulness       ¿Está la respuesta fundamentada en el contexto?
  - answer_relevancy   ¿Responde la respuesta a la pregunta?
  - context_precision  ¿Es el contexto recuperado relevante para la pregunta?

Salida:
  - Tabla por consola
  - evaluation/results.csv  con una fila por (pregunta × modo)

Uso:
  python evaluation/ragas_eval.py
  python evaluation/ragas_eval.py --db database/funds.db --out evaluation/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

# aiohttp 3.9+ removed ClientConnectorDNSError; google-genai SDK still references it
if not hasattr(aiohttp, "ClientConnectorDNSError"):
    aiohttp.ClientConnectorDNSError = aiohttp.ClientConnectorError

import chromadb
from dotenv import load_dotenv
from google import genai as google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import GoogleEmbeddings
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

load_dotenv()
log = logging.getLogger(__name__)


class _GoogleEmbeddingsAdapter(BaseRagasEmbeddings):
    """
    Adapts ragas.embeddings.GoogleEmbeddings to the BaseRagasEmbeddings interface.

    The legacy ragas answer_relevancy metric calls embed_query / embed_documents
    (LangChain-style), but GoogleEmbeddings (BaseRagasEmbedding) only exposes
    embed_text / embed_texts. This adapter bridges the gap.
    """

    def __init__(self, google_emb: GoogleEmbeddings):
        super().__init__()
        self._emb = google_emb
        self.set_run_config(RunConfig())

    def embed_query(self, text: str) -> list:
        return self._emb.embed_text(text)

    def embed_documents(self, texts: list) -> list:
        return self._emb.embed_texts(texts)

    async def aembed_query(self, text: str) -> list:
        return await self._emb.aembed_text(text)

    async def aembed_documents(self, texts: list) -> list:
        return await self._emb.aembed_texts(texts)

# ── Constantes ────────────────────────────────────────────────────────────────

GEMINI_MODEL     = "gemini-2.5-flash"
EMBED_MODEL      = "gemini-embedding-001"
DB_PATH          = Path("database/funds.db")
CHROMA_PATH      = Path("database/chroma")
CHROMA_COLLECTION = "politica_inversion"
OUT_CSV          = Path("evaluation/results.csv")
N_CHROMA_RESULTS = 3     # fragmentos recuperados por pregunta
RATE_LIMIT_SECS  = 12.0  # pausa entre generaciones para respetar 10 RPM free-tier

# ── Set de evaluación: 10 preguntas con ground truth verificable en los PDFs ──
#
# Categorías:
#   factual  → respuesta exacta extraíble de campos estructurados del DFI
#   semantic → respuesta requiere comprensión de la política de inversión
#
# ground_truth está redactado a partir del texto literal de los PDFs descargados.

TEST_SET: list[dict] = [

    # ── A&G RENTA FIJA CORTO PLAZO (ES0156873004) ─────────────────────────────
    {
        "id": "q01",
        "tipo": "factual",
        "fondo": "A&G RENTA FIJA CORTO PLAZO. FI",
        "pregunta": "¿Cuál es el nivel de riesgo del fondo A&G Renta Fija Corto Plazo?",
        "ground_truth": "El nivel de riesgo del fondo A&G Renta Fija Corto Plazo es 2 sobre una escala de 1 a 7.",
    },
    {
        "id": "q02",
        "tipo": "factual",
        "fondo": "A&G RENTA FIJA CORTO PLAZO. FI",
        "pregunta": "¿Cuál es el horizonte temporal mínimo recomendado para el fondo A&G Renta Fija Corto Plazo?",
        "ground_truth": "El horizonte temporal mínimo recomendado es de 1 año. El fondo puede no ser adecuado para inversores que prevean retirar su dinero en un plazo inferior a 1 año.",
    },
    {
        "id": "q03",
        "tipo": "factual",
        "fondo": "A&G RENTA FIJA CORTO PLAZO. FI",
        "pregunta": "¿Cuál es el importe mínimo de inversión del fondo A&G Renta Fija Corto Plazo?",
        "ground_truth": "El importe mínimo de inversión del fondo A&G Renta Fija Corto Plazo es de 10.000 euros.",
    },
    {
        "id": "q04",
        "tipo": "factual",
        "fondo": "A&G RENTA FIJA CORTO PLAZO. FI",
        "pregunta": "¿Qué índice de referencia utiliza el fondo A&G Renta Fija Corto Plazo?",
        "ground_truth": "La gestión toma como referencia la rentabilidad del índice €STR Euro Short-Term Rate (€STR Index), utilizado a efectos meramente comparativos.",
    },
    {
        "id": "q05",
        "tipo": "semantic",
        "fondo": "A&G RENTA FIJA CORTO PLAZO. FI",
        "pregunta": "¿En qué tipo de activos invierte el fondo A&G Renta Fija Corto Plazo y qué restricciones de calidad crediticia aplica?",
        "ground_truth": (
            "El fondo invierte el 100% de la exposición total en activos de renta fija pública y/o privada, "
            "incluyendo depósitos e instrumentos del mercado monetario. Las emisiones deben tener al menos "
            "mediana calidad crediticia (rating mínimo BBB- o equivalentes). Se puede invertir hasta un 15% "
            "en emisiones con baja calidad crediticia (inferior a BBB-). La duración media de la cartera será "
            "igual o inferior a 1 año."
        ),
    },

    # ── A&P LIFESCIENCE FUND (V06985584) ──────────────────────────────────────
    {
        "id": "q06",
        "tipo": "factual",
        "fondo": "A&P LIFESCIENCE FUND, FI",
        "pregunta": "¿Cuál es el horizonte temporal mínimo recomendado para el fondo A&P Lifescience Fund?",
        "ground_truth": "El horizonte temporal mínimo recomendado para el fondo A&P Lifescience Fund es de 8 años. El fondo puede no ser adecuado para inversores que prevean retirar su dinero en un plazo inferior a 8 años.",
    },
    {
        "id": "q07",
        "tipo": "factual",
        "fondo": "A&P LIFESCIENCE FUND, FI",
        "pregunta": "¿Qué índice de referencia sigue el fondo A&P Lifescience Fund?",
        "ground_truth": "La gestión toma como referencia la rentabilidad del índice NASDAQ Biotechnology Total Return Index, utilizado a efectos meramente informativos y/o comparativos.",
    },
    {
        "id": "q08",
        "tipo": "semantic",
        "fondo": "A&P LIFESCIENCE FUND, FI",
        "pregunta": "¿En qué sectores y geografías invierte el fondo A&P Lifescience Fund?",
        "ground_truth": (
            "El fondo invierte más del 75% de la exposición total en renta variable, principalmente en el sector "
            "biotecnológico y de la salud (pequeñas empresas de fármacos innovadores, empresas de genéricos, "
            "grandes farmacéuticas, dispositivos médicos, servicios hospitalarios y aseguradoras). "
            "Las inversiones se centran en emisores de países OCDE, pudiendo invertir hasta un 30% en "
            "emisores de mercados no OCDE, incluyendo emergentes."
        ),
    },

    # ── ABACO GLOBAL VALUE OPPORTUNITIES (V87152088) ──────────────────────────
    {
        "id": "q09",
        "tipo": "factual",
        "fondo": "ABACO GLOBAL VALUE OPPORTUNITIES FI",
        "pregunta": "¿Cuál es el horizonte temporal mínimo recomendado para el fondo Ábaco Global Value Opportunities?",
        "ground_truth": "El horizonte temporal mínimo recomendado para el fondo Ábaco Global Value Opportunities es de 3 años. El fondo puede no ser adecuado para inversores que prevean retirar su dinero en un plazo inferior a 3 años.",
    },
    {
        "id": "q10",
        "tipo": "semantic",
        "fondo": "ABACO GLOBAL VALUE OPPORTUNITIES FI",
        "pregunta": "¿Qué filosofía de inversión aplica el fondo Ábaco Global Value Opportunities y qué restricciones geográficas tiene?",
        "ground_truth": (
            "El fondo aplica una filosofía Value Investing, analizando empresas para buscar activos infravalorados "
            "respecto a su precio de mercado. No tiene predeterminación por tipo de emisor, rating, capitalización "
            "bursátil, divisa, sector económico ni países. Los emisores y mercados pueden ser de países OCDE o "
            "emergentes, sin limitación, pudiendo existir puntualmente concentración geográfica o sectorial."
        ),
    },
]


# ── Clientes LLM y embeddings ─────────────────────────────────────────────────

def _build_ragas_llm() -> LangchainLLMWrapper:
    """Construye el wrapper LangChain para las métricas legacy de RAGAS 0.4."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY no está definida.")
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0)
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings() -> _GoogleEmbeddingsAdapter:
    """Construye embeddings Google para RAGAS AnswerRelevancy (google-genai SDK)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY no está definida.")
    client = google_genai.Client(api_key=api_key)
    google_emb = GoogleEmbeddings(client=client, model=EMBED_MODEL)
    return _GoogleEmbeddingsAdapter(google_emb)


def _build_generation_llm() -> ChatGoogleGenerativeAI:
    """LLM de LangChain para generar respuestas RAG y base."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )


# ── Recuperación de contexto (ChromaDB) ──────────────────────────────────────

def _retrieve_contexts(pregunta: str, chroma_path: Path) -> list[str]:
    """
    Recupera los N_CHROMA_RESULTS fragmentos de politica_inversion más similares
    a la pregunta desde ChromaDB.
    Devuelve lista vacía si la colección no existe o está vacía.
    """
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.get_collection(CHROMA_COLLECTION)
        n = min(N_CHROMA_RESULTS, col.count())
        if n == 0:
            return []
        result = col.query(query_texts=[pregunta], n_results=n, include=["documents"])
        return result["documents"][0]
    except Exception as exc:
        log.warning("ChromaDB no disponible: %s", exc)
        return []


# ── Generación de respuestas ──────────────────────────────────────────────────

RAG_SYSTEM = (
    "Eres un asistente experto en fondos de inversión españoles. "
    "Responde ÚNICAMENTE usando la información del contexto proporcionado. "
    "Si la información no está en el contexto, indícalo explícitamente. "
    "Responde en español, de forma concisa y precisa."
)

BASE_SYSTEM = (
    "Eres un asistente experto en fondos de inversión españoles. "
    "Responde a la pregunta con tu conocimiento general. "
    "Responde en español, de forma concisa y precisa."
)


def _generate_rag_answer(pregunta: str, contextos: list[str], llm: ChatGoogleGenerativeAI) -> str:
    """Genera una respuesta usando el contexto recuperado (modo RAG)."""
    if not contextos:
        return "No se encontró contexto relevante en la base de datos."
    contexto_texto = "\n\n---\n\n".join(contextos)
    messages = [
        ("system", RAG_SYSTEM),
        ("human", f"Contexto:\n{contexto_texto}\n\nPregunta: {pregunta}"),
    ]
    return llm.invoke(messages).content


def _generate_base_answer(pregunta: str, llm: ChatGoogleGenerativeAI) -> str:
    """Genera una respuesta sin contexto (modo base LLM)."""
    messages = [
        ("system", BASE_SYSTEM),
        ("human", pregunta),
    ]
    return llm.invoke(messages).content


# ── Construcción del dataset RAGAS ────────────────────────────────────────────

@dataclass
class EvalRow:
    """Fila de resultado para una pregunta en un modo de evaluación."""
    question_id:     str
    tipo:            str
    fondo:           str
    modo:            str       # "rag" o "base_llm"
    pregunta:        str
    respuesta:       str
    contextos:       list[str]
    ground_truth:    str
    faithfulness:    float = field(default=float("nan"))
    answer_relevancy: float = field(default=float("nan"))
    context_precision: float = field(default=float("nan"))
    score_medio:     float = field(default=float("nan"))


RESPONSES_CACHE = Path("evaluation/responses_cache.json")


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            data = json.load(f)
        log.info("Caché cargada: %d entradas (%s)", len(data), cache_path)
        return data
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def build_eval_rows(
    test_set: list[dict],
    chroma_path: Path,
    gen_llm: ChatGoogleGenerativeAI,
    refresh_cache: bool = False,
    cache_path: Path = RESPONSES_CACHE,
) -> tuple[list[EvalRow], list[EvalRow]]:
    """
    Para cada pregunta del test set genera:
      - Una fila RAG (con contexto de ChromaDB)
      - Una fila Base LLM (sin contexto)

    Las respuestas se cachean en `cache_path` (JSON) para evitar re-consumir
    cuota de API en ejecuciones sucesivas. Usa --refresh para regenerarlas.

    Devuelve (filas_rag, filas_base).
    """
    cache = {} if refresh_cache else _load_cache(cache_path)
    filas_rag: list[EvalRow] = []
    filas_base: list[EvalRow] = []

    for i, item in enumerate(test_set, 1):
        qid = item["id"]
        cache_key_rag  = f"{qid}_rag"
        cache_key_base = f"{qid}_base"

        contextos = _retrieve_contexts(item["pregunta"], chroma_path)

        # ── Modo RAG ──────────────────────────────────────────────────────────
        if cache_key_rag in cache:
            log.info("[%d/%d] %s (RAG) — desde caché", i, len(test_set), qid)
            resp_rag = cache[cache_key_rag]["respuesta"]
            contextos = cache[cache_key_rag].get("contextos", contextos)
        else:
            log.info("[%d/%d] %s (RAG) — llamando LLM…", i, len(test_set), qid)
            resp_rag = _generate_rag_answer(item["pregunta"], contextos, gen_llm)
            cache[cache_key_rag] = {"respuesta": resp_rag, "contextos": contextos}
            _save_cache(cache, cache_path)
            time.sleep(RATE_LIMIT_SECS)

        # ── Modo Base LLM ─────────────────────────────────────────────────────
        if cache_key_base in cache:
            log.info("[%d/%d] %s (Base) — desde caché", i, len(test_set), qid)
            resp_base = cache[cache_key_base]["respuesta"]
        else:
            log.info("[%d/%d] %s (Base) — llamando LLM…", i, len(test_set), qid)
            resp_base = _generate_base_answer(item["pregunta"], gen_llm)
            cache[cache_key_base] = {"respuesta": resp_base}
            _save_cache(cache, cache_path)
            time.sleep(RATE_LIMIT_SECS)

        filas_rag.append(EvalRow(
            question_id=qid,
            tipo=item["tipo"],
            fondo=item["fondo"],
            modo="rag",
            pregunta=item["pregunta"],
            respuesta=resp_rag,
            contextos=contextos if contextos else [""],
            ground_truth=item["ground_truth"],
        ))
        filas_base.append(EvalRow(
            question_id=qid,
            tipo=item["tipo"],
            fondo=item["fondo"],
            modo="base_llm",
            pregunta=item["pregunta"],
            respuesta=resp_base,
            contextos=[""],
            ground_truth=item["ground_truth"],
        ))

    return filas_rag, filas_base


# ── Evaluación RAGAS ──────────────────────────────────────────────────────────

def run_ragas(
    filas: list[EvalRow],
    ragas_llm: LangchainLLMWrapper,
    ragas_emb: _GoogleEmbeddingsAdapter,
    label: str,
) -> list[EvalRow]:
    """
    Ejecuta las tres métricas RAGAS sobre el conjunto de filas.
    Actualiza faithfulness, answer_relevancy y context_precision en cada fila.
    """
    samples = [
        SingleTurnSample(
            user_input=f.pregunta,
            response=f.respuesta,
            retrieved_contexts=f.contextos,
            reference=f.ground_truth,
        )
        for f in filas
    ]
    dataset = EvaluationDataset(samples=samples)

    # Asignar LLM/embeddings a los singletons de métricas legacy (compatibles con evaluate())
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_precision.llm = ragas_llm
    metrics = [faithfulness, answer_relevancy, context_precision]

    # RunConfig con timeout y reintentos ajustados al free-tier (10 RPM)
    run_cfg = RunConfig(timeout=120, max_retries=5, max_wait=70, max_workers=4)

    log.info("Evaluando %d muestras [%s] con RAGAS…", len(filas), label)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_cfg,
        raise_exceptions=False,
        show_progress=True,
        batch_size=1,
    )

    df = result.to_pandas()

    for i, fila in enumerate(filas):
        fila.faithfulness      = float(df.loc[i, "faithfulness"])      if "faithfulness"      in df.columns else float("nan")
        fila.answer_relevancy  = float(df.loc[i, "answer_relevancy"])  if "answer_relevancy"  in df.columns else float("nan")
        fila.context_precision = float(df.loc[i, "context_precision"]) if "context_precision" in df.columns else float("nan")

        vals = [v for v in [fila.faithfulness, fila.answer_relevancy, fila.context_precision] if v == v]
        fila.score_medio = sum(vals) / len(vals) if vals else float("nan")

    return filas


# ── Presentación de resultados ────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:.3f}" if v == v else "  — "


def print_table(filas_rag: list[EvalRow], filas_base: list[EvalRow]) -> None:
    """Imprime una tabla comparativa RAG vs Base LLM por pregunta."""
    sep = "─" * 112
    header = (
        f"{'ID':>4}  {'Tipo':8}  {'Fondo':35}  {'Modo':8}  "
        f"{'Faith':>6}  {'Relev':>6}  {'Prec':>6}  {'Media':>6}"
    )
    print("\n" + "═" * 112)
    print("  RESULTADOS RAGAS — RAG vs Base LLM")
    print("═" * 112)
    print(header)
    print(sep)

    for rag, base in zip(filas_rag, filas_base):
        for fila in (rag, base):
            print(
                f"{fila.question_id:>4}  {fila.tipo:8}  {fila.fondo[:35]:35}  "
                f"{fila.modo:8}  "
                f"{_fmt(fila.faithfulness):>6}  "
                f"{_fmt(fila.answer_relevancy):>6}  "
                f"{_fmt(fila.context_precision):>6}  "
                f"{_fmt(fila.score_medio):>6}"
            )
        print(sep)

    # Promedios globales
    def _media(filas: list[EvalRow], attr: str) -> float:
        vals = [getattr(f, attr) for f in filas if getattr(f, attr) == getattr(f, attr)]
        return sum(vals) / len(vals) if vals else float("nan")

    print()
    print(f"  {'PROMEDIO RAG':50}  "
          f"{_media(filas_rag, 'faithfulness'):>6.3f}  "
          f"{_media(filas_rag, 'answer_relevancy'):>6.3f}  "
          f"{_media(filas_rag, 'context_precision'):>6.3f}  "
          f"{_media(filas_rag, 'score_medio'):>6.3f}")
    print(f"  {'PROMEDIO BASE LLM':50}  "
          f"{_media(filas_base, 'faithfulness'):>6.3f}  "
          f"{_media(filas_base, 'answer_relevancy'):>6.3f}  "
          f"{_media(filas_base, 'context_precision'):>6.3f}  "
          f"{_media(filas_base, 'score_medio'):>6.3f}")
    print("═" * 112)


def save_csv(filas: list[EvalRow], out_path: Path) -> None:
    """Guarda todos los resultados en un CSV (una fila por pregunta × modo)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id", "tipo", "fondo", "modo", "pregunta",
        "faithfulness", "answer_relevancy", "context_precision", "score_medio",
        "respuesta", "ground_truth", "contextos",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for fila in filas:
            row = fila.__dict__.copy()
            row["contextos"] = " | ".join(row["contextos"])[:500]
            writer.writerow(row)
    log.info("Resultados guardados en %s", out_path)


# ── Orquestador ───────────────────────────────────────────────────────────────

def run_evaluation(
    db_path: Path = DB_PATH,
    chroma_path: Path = CHROMA_PATH,
    out_csv: Path = OUT_CSV,
    refresh_cache: bool = False,
) -> None:
    """Pipeline completo: generación (con caché) → evaluación RAGAS → tabla → CSV."""

    log.info("Construyendo clientes LLM y embeddings…")
    ragas_llm  = _build_ragas_llm()
    ragas_emb  = _build_ragas_embeddings()
    gen_llm    = _build_generation_llm()

    log.info("Generando respuestas para %d preguntas × 2 modos…", len(TEST_SET))
    filas_rag, filas_base = build_eval_rows(
        TEST_SET, chroma_path, gen_llm, refresh_cache=refresh_cache
    )

    filas_rag  = run_ragas(filas_rag,  ragas_llm, ragas_emb, label="RAG")
    filas_base = run_ragas(filas_base, ragas_llm, ragas_emb, label="Base LLM")

    print_table(filas_rag, filas_base)
    save_csv(filas_rag + filas_base, out_csv)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación RAGAS del sistema RAG de fondos de inversión."
    )
    parser.add_argument("--db",     type=Path, default=DB_PATH,    help="Ruta a funds.db")
    parser.add_argument("--chroma", type=Path, default=CHROMA_PATH,help="Directorio ChromaDB")
    parser.add_argument("--out",    type=Path, default=OUT_CSV,    help="Ruta del CSV de salida")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Ignorar la caché y regenerar todas las respuestas LLM.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_evaluation(
        db_path=args.db,
        chroma_path=args.chroma,
        out_csv=args.out,
        refresh_cache=args.refresh,
    )
