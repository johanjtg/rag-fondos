"""
Evaluador 3 — Comparativa de modelos de embedding.

Para cada modelo de embedding candidato:
  1. Re-indexa la politica_inversion de todos los fondos en una colección temporal.
  2. Corre el mismo golden dataset del Evaluador 2.
  3. Compara Precision@K, Hit@1 y MRR entre modelos.

Objetivo: justificar empíricamente la elección de
paraphrase-multilingual-MiniLM-L12-v2 frente a alternativas.

Modelos evaluados:
  - paraphrase-multilingual-MiniLM-L12-v2  (actual, multilingüe)
  - all-MiniLM-L6-v2                       (más rápido, solo inglés — baseline)
  - distiluse-base-multilingual-cased-v2   (multilingüe alternativo)

Uso:
  python -m evaluation.evaluador_embeddings
  python -m evaluation.evaluador_embeddings --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sklearn.metrics.pairwise import cosine_similarity

from scoring.scorer import load_all_funds, _pasa_filtros
from scoring.fund_vectorizer import vectorize
from scoring.user_profiler import UserProfile

log = logging.getLogger(__name__)

GOLDEN_PATH = Path("evaluation/golden_dataset_recomendacion.json")
DB_PATH     = Path("database/funds.db")
OUTPUT_PATH = Path("evaluation/resultados_embeddings.json")

# Pesos fijos — los mismos del sistema actual
W_COSENO   = 0.60
W_SEMANTIC = 0.40
W_ESG      = 0.10
ESG_THRESHOLD = 0.6
TOP_N = 5

MODELOS_EMBEDDING = [
    {
        "nombre": "paraphrase-multilingual-MiniLM-L12-v2 (actual)",
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "descripcion": "Multilingüe, 384 dims — modelo actual del sistema",
    },
    {
        "nombre": "all-MiniLM-L6-v2 (baseline inglés)",
        "model_name": "all-MiniLM-L6-v2",
        "descripcion": "Solo inglés, 384 dims — baseline rápido",
    },
    {
        "nombre": "distiluse-base-multilingual-cased-v2",
        "model_name": "distiluse-base-multilingual-cased-v2",
        "descripcion": "Multilingüe alternativo, 512 dims",
    },
]


# ── Indexación temporal en ChromaDB ──────────────────────────────────────────

def _indexar_fondos(fondos, model_name: str) -> chromadb.Collection:
    """
    Crea una colección ChromaDB en memoria con el modelo dado
    e indexa la politica_inversion de todos los fondos.
    Cada modelo usa un nombre de colección único para evitar
    conflictos de dimensión entre modelos.
    """
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.EphemeralClient()
    # Nombre único por modelo para evitar conflictos de dimensión
    nombre_coleccion = "eval_" + model_name.replace("/", "_").replace("-", "_")[:40]
    coleccion = client.get_or_create_collection(
        name=nombre_coleccion,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = [], [], []
    for fondo in fondos:
        if fondo.politica_inversion and fondo.politica_inversion.strip():
            ids.append(fondo.isin)
            docs.append(fondo.politica_inversion)
            metas.append({"nombre_fondo": fondo.nombre_fondo})

    if ids:
        coleccion.upsert(ids=ids, documents=docs, metadatas=metas)

    return coleccion


def _semantic_scores_coleccion(
    query: str,
    isins: list[str],
    coleccion: chromadb.Collection,
) -> dict[str, float]:
    if not query.strip() or not isins:
        return {isin: 0.0 for isin in isins}

    n = min(len(isins), coleccion.count() or 1)
    resultados = coleccion.query(
        query_texts=[query],
        n_results=n,
        include=["distances"],
    )

    scores = {isin: 0.0 for isin in isins}
    for isin_r, dist in zip(resultados["ids"][0], resultados["distances"][0]):
        if isin_r in scores:
            scores[isin_r] = round(1.0 - dist / 2.0, 4)
    return scores


# ── Scoring con colección personalizada ──────────────────────────────────────

def _recomendar(perfil: UserProfile, fondos, coleccion: chromadb.Collection) -> list[str]:
    vector_usuario = perfil.to_vector().reshape(1, -1)
    candidatos = [f for f in fondos if _pasa_filtros(f, perfil)[0]]
    if not candidatos:
        return []

    vectores = np.array([vectorize(f) for f in candidatos])
    cosenos  = cosine_similarity(vector_usuario, vectores)[0]

    query = " ".join(filter(None, [
        perfil.preferencia_geografica,
        perfil.preferencia_sectorial,
    ]))
    isins      = [f.isin for f in candidatos]
    semanticos = _semantic_scores_coleccion(query, isins, coleccion)

    scores = []
    for i, fondo in enumerate(candidatos):
        esg_boost = W_ESG if (perfil.sensibilidad_esg > ESG_THRESHOLD and fondo.esg) else 0.0
        score = W_COSENO * float(cosenos[i]) + W_SEMANTIC * semanticos.get(fondo.isin, 0.0) + esg_boost
        scores.append((fondo.isin, min(score, 1.0)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [isin for isin, _ in scores[:TOP_N]]


# ── Métricas ──────────────────────────────────────────────────────────────────

def _precision_at_k(esperados: list[str], obtenidos: list[str]) -> float:
    if not esperados:
        return 0.0
    return sum(1 for e in esperados if e in obtenidos) / len(esperados)

def _mrr(esperados: list[str], obtenidos: list[str]) -> float:
    for rank, isin in enumerate(obtenidos, start=1):
        if isin in esperados:
            return 1.0 / rank
    return 0.0

def _hit_at_1(top1: str, obtenidos: list[str]) -> bool:
    return bool(obtenidos) and obtenidos[0] == top1


# ── Evaluación ────────────────────────────────────────────────────────────────

@dataclass
class ResultadoModelo:
    nombre:      str
    model_name:  str
    descripcion: str
    tiempo_s:    float
    perfiles:    list[dict] = field(default_factory=list)

    @property
    def precision_media(self) -> float:
        return sum(p["precision_at_k"] for p in self.perfiles) / len(self.perfiles) if self.perfiles else 0.0

    @property
    def mrr_medio(self) -> float:
        return sum(p["mrr"] for p in self.perfiles) / len(self.perfiles) if self.perfiles else 0.0

    @property
    def hit_at_1_ratio(self) -> float:
        return sum(1 for p in self.perfiles if p["hit_at_1"]) / len(self.perfiles) if self.perfiles else 0.0


def evaluar(
    golden_path: Path = GOLDEN_PATH,
    db_path: Path = DB_PATH,
    verbose: bool = False,
) -> list[ResultadoModelo]:

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    fondos = load_all_funds(db_path)
    resultados: list[ResultadoModelo] = []

    for config in MODELOS_EMBEDDING:
        print(f"\nIndexando con: {config['model_name']}...")
        t0 = time.time()

        coleccion = _indexar_fondos(fondos, config["model_name"])
        tiempo_indexacion = time.time() - t0

        rm = ResultadoModelo(
            nombre=config["nombre"],
            model_name=config["model_name"],
            descripcion=config["descripcion"],
            tiempo_s=round(tiempo_indexacion, 2),
        )

        for entrada in golden:
            perfil = UserProfile()
            for campo, respuesta in entrada["respuestas"].items():
                perfil.update(campo, respuesta)

            obtenidos = _recomendar(perfil, fondos, coleccion)
            esperados = entrada["fondos_esperados"]
            top1      = entrada["fondo_top1_esperado"]

            rm.perfiles.append({
                "id":            entrada["id"],
                "descripcion":   entrada["descripcion"],
                "esperados":     esperados,
                "obtenidos":     obtenidos,
                "precision_at_k": _precision_at_k(esperados, obtenidos),
                "mrr":           _mrr(esperados, obtenidos),
                "hit_at_1":      _hit_at_1(top1, obtenidos),
            })

        resultados.append(rm)

    return resultados


# ── Informe ───────────────────────────────────────────────────────────────────

def imprimir_informe(resultados: list[ResultadoModelo], verbose: bool = False) -> None:
    print("\n" + "═" * 75)
    print("  EVALUADOR 3 — COMPARATIVA DE MODELOS DE EMBEDDING")
    print("═" * 75)
    print(f"\n{'Modelo':<45} {'P@K':>6} {'Hit@1':>7} {'MRR':>7} {'t(s)':>6}")
    print("─" * 75)
    for r in resultados:
        marca = " ◄ ACTUAL" if "actual" in r.nombre else ""
        print(
            f"{r.nombre:<45} {r.precision_media:>5.0%} "
            f"{r.hit_at_1_ratio:>6.0%} {r.mrr_medio:>6.3f} "
            f"{r.tiempo_s:>5.1f}s{marca}"
        )

    if verbose:
        for r in resultados:
            print(f"\n── {r.nombre} ──")
            for p in r.perfiles:
                estado = "✓" if p["precision_at_k"] > 0 else "✗"
                print(f"  {estado} {p['id']:<35} P@K={p['precision_at_k']:.0%}  MRR={p['mrr']:.2f}")


def guardar_json(resultados: list[ResultadoModelo], output_path: Path) -> None:
    data = [
        {
            "nombre":          r.nombre,
            "model_name":      r.model_name,
            "descripcion":     r.descripcion,
            "tiempo_s":        r.tiempo_s,
            "precision_media": round(r.precision_media, 4),
            "mrr_medio":       round(r.mrr_medio, 4),
            "hit_at_1_ratio":  round(r.hit_at_1_ratio, 4),
            "perfiles":        r.perfiles,
        }
        for r in resultados
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparativa de modelos de embedding para el sistema RAG."
    )
    parser.add_argument("--golden",  type=Path, default=GOLDEN_PATH)
    parser.add_argument("--db",      type=Path, default=DB_PATH)
    parser.add_argument("--output",  type=Path, default=OUTPUT_PATH)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING)
    resultados = evaluar(args.golden, args.db, args.verbose)
    imprimir_informe(resultados, verbose=args.verbose)
    guardar_json(resultados, args.output)


if __name__ == "__main__":
    main()
