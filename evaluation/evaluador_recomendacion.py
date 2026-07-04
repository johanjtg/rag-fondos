"""
Evaluador de calidad de recomendación (Evaluador 2).

Para cada perfil del golden dataset:
  1. Construye el UserProfile a partir de las respuestas de texto libre.
  2. Llama al motor de scoring (sin LLM).
  3. Mide si los fondos esperados aparecen en el top-N.

Métricas:
  - Precision@K  : fracción de fondos esperados en el top-K
  - Hit@1        : si el fondo_top1_esperado aparece en la posición 1
  - MRR          : 1 / rango del primer fondo esperado encontrado

Además compara distintas configuraciones de pesos (60/40, 70/30, 50/50, 80/20)
para justificar empíricamente la ponderación elegida.

Uso:
  python -m evaluation.evaluador_recomendacion
  python -m evaluation.evaluador_recomendacion --top 3
  python -m evaluation.evaluador_recomendacion --golden evaluation/golden_dataset_recomendacion.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_PATH = Path("evaluation/golden_dataset_recomendacion.json")
DB_PATH     = Path("database/funds.db")
CHROMA_PATH = Path("database/chroma")
OUTPUT_PATH = Path("evaluation/resultados_recomendacion.json")

CONFIGURACIONES_PESOS = [
    {"nombre": "60/40 (actual)", "coseno": 0.60, "semantico": 0.40},
    {"nombre": "70/30",          "coseno": 0.70, "semantico": 0.30},
    {"nombre": "50/50",          "coseno": 0.50, "semantico": 0.50},
    {"nombre": "80/20",          "coseno": 0.80, "semantico": 0.20},
    {"nombre": "100/0",          "coseno": 1.00, "semantico": 0.00},
    {"nombre": "0/100",          "coseno": 0.00, "semantico": 1.00},
]


# ── Estructuras de resultado ──────────────────────────────────────────────────

@dataclass
class ResultadoPerfil:
    id:                  str
    descripcion:         str
    fondos_esperados:    list[str]
    top1_esperado:       str
    fondos_obtenidos:    list[str]   # ISINs en orden de score
    precision_at_k:      float
    hit_at_1:            bool
    mrr:                 float
    config_pesos:        str


@dataclass
class ResultadoConfiguracion:
    nombre:      str
    coseno:      float
    semantico:   float
    perfiles:    list[ResultadoPerfil] = field(default_factory=list)

    @property
    def precision_media(self) -> float:
        if not self.perfiles:
            return 0.0
        return sum(p.precision_at_k for p in self.perfiles) / len(self.perfiles)

    @property
    def mrr_medio(self) -> float:
        if not self.perfiles:
            return 0.0
        return sum(p.mrr for p in self.perfiles) / len(self.perfiles)

    @property
    def hit_at_1_ratio(self) -> float:
        if not self.perfiles:
            return 0.0
        return sum(1 for p in self.perfiles if p.hit_at_1) / len(self.perfiles)


# ── Scoring con pesos configurables ──────────────────────────────────────────

def _score_con_pesos(
    perfil_usuario,
    fondos,
    w_coseno: float,
    w_semantico: float,
    top_n: int,
    chroma_path: Path,
) -> list[str]:
    """
    Reproduce el motor de scoring con pesos personalizados.
    Devuelve los ISINs ordenados por score descendente.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    from scoring.scorer import _pasa_filtros, _semantic_scores
    from scoring.fund_vectorizer import vectorize

    W_ESG = 0.10
    ESG_THRESHOLD = 0.6

    vector_usuario = perfil_usuario.to_vector().reshape(1, -1)

    candidatos = [f for f in fondos if _pasa_filtros(f, perfil_usuario)[0]]
    if not candidatos:
        return []

    vectores = np.array([vectorize(f) for f in candidatos])
    cosenos  = cosine_similarity(vector_usuario, vectores)[0]

    query = " ".join(filter(None, [
        perfil_usuario.preferencia_geografica,
        perfil_usuario.preferencia_sectorial,
    ]))
    isins = [f.isin for f in candidatos]
    semanticos = _semantic_scores(query, isins, chroma_path)

    scores = []
    for i, fondo in enumerate(candidatos):
        esg_boost = W_ESG if (perfil_usuario.sensibilidad_esg > ESG_THRESHOLD and fondo.esg) else 0.0
        score = w_coseno * float(cosenos[i]) + w_semantico * semanticos.get(fondo.isin, 0.0) + esg_boost
        scores.append((fondo.isin, min(score, 1.0)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [isin for isin, _ in scores[:top_n]]


# ── Métricas ──────────────────────────────────────────────────────────────────

def _precision_at_k(esperados: list[str], obtenidos: list[str]) -> float:
    if not esperados:
        return 0.0
    encontrados = sum(1 for e in esperados if e in obtenidos)
    return encontrados / len(esperados)


def _mrr(esperados: list[str], obtenidos: list[str]) -> float:
    for rank, isin in enumerate(obtenidos, start=1):
        if isin in esperados:
            return 1.0 / rank
    return 0.0


def _hit_at_1(top1_esperado: str, obtenidos: list[str]) -> bool:
    return bool(obtenidos) and obtenidos[0] == top1_esperado


# ── Evaluación ────────────────────────────────────────────────────────────────

def evaluar(
    golden_path: Path = GOLDEN_PATH,
    db_path: Path = DB_PATH,
    chroma_path: Path = CHROMA_PATH,
    top_n: int = 5,
) -> list[ResultadoConfiguracion]:

    from scoring.scorer import load_all_funds
    from scoring.user_profiler import UserProfile

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    fondos = load_all_funds(db_path)
    resultados_config: list[ResultadoConfiguracion] = []

    for config in CONFIGURACIONES_PESOS:
        rc = ResultadoConfiguracion(
            nombre=config["nombre"],
            coseno=config["coseno"],
            semantico=config["semantico"],
        )

        for entrada in golden:
            perfil = UserProfile()
            respuestas = entrada["respuestas"]
            perfil.update("capital",    respuestas["capital"])
            perfil.update("horizonte",  respuestas["horizonte"])
            perfil.update("riesgo",     respuestas["riesgo"])
            perfil.update("liquidez",   respuestas["liquidez"])
            perfil.update("tematica",   respuestas["tematica"])
            perfil.update("estrategia", respuestas["estrategia"])

            obtenidos = _score_con_pesos(
                perfil, fondos,
                w_coseno=config["coseno"],
                w_semantico=config["semantico"],
                top_n=top_n,
                chroma_path=chroma_path,
            )

            esperados = entrada["fondos_esperados"]
            top1      = entrada["fondo_top1_esperado"]

            rc.perfiles.append(ResultadoPerfil(
                id=entrada["id"],
                descripcion=entrada["descripcion"],
                fondos_esperados=esperados,
                top1_esperado=top1,
                fondos_obtenidos=obtenidos,
                precision_at_k=_precision_at_k(esperados, obtenidos),
                hit_at_1=_hit_at_1(top1, obtenidos),
                mrr=_mrr(esperados, obtenidos),
                config_pesos=config["nombre"],
            ))

        resultados_config.append(rc)

    return resultados_config


# ── Informe ───────────────────────────────────────────────────────────────────

def imprimir_informe(resultados: list[ResultadoConfiguracion], verbose: bool = False) -> None:
    print("\n" + "═" * 70)
    print("  EVALUADOR 2 — CALIDAD DE RECOMENDACIÓN")
    print("═" * 70)

    print(f"\n{'Configuración':<20} {'Precision@K':>12} {'Hit@1':>8} {'MRR':>8}")
    print("─" * 52)
    for rc in resultados:
        marca = " ◄ ACTUAL" if rc.nombre == "60/40 (actual)" else ""
        print(
            f"{rc.nombre:<20} {rc.precision_media:>11.0%} "
            f"{rc.hit_at_1_ratio:>7.0%} {rc.mrr_medio:>7.3f}{marca}"
        )

    if verbose:
        for rc in resultados:
            print(f"\n── {rc.nombre} ──")
            for p in rc.perfiles:
                estado = "✓" if p.precision_at_k > 0 else "✗"
                print(f"  {estado} {p.id:<30} P@K={p.precision_at_k:.0%}  MRR={p.mrr:.2f}  Hit@1={'✓' if p.hit_at_1 else '✗'}")
                if p.precision_at_k < 1.0:
                    print(f"    esperados : {p.fondos_esperados}")
                    print(f"    obtenidos : {p.fondos_obtenidos}")


def guardar_json(resultados: list[ResultadoConfiguracion], output_path: Path) -> None:
    data = []
    for rc in resultados:
        data.append({
            "configuracion": rc.nombre,
            "coseno": rc.coseno,
            "semantico": rc.semantico,
            "precision_media": round(rc.precision_media, 4),
            "mrr_medio": round(rc.mrr_medio, 4),
            "hit_at_1_ratio": round(rc.hit_at_1_ratio, 4),
            "perfiles": [
                {
                    "id": p.id,
                    "descripcion": p.descripcion,
                    "fondos_esperados": p.fondos_esperados,
                    "fondos_obtenidos": p.fondos_obtenidos,
                    "precision_at_k": round(p.precision_at_k, 4),
                    "hit_at_1": p.hit_at_1,
                    "mrr": round(p.mrr, 4),
                }
                for p in rc.perfiles
            ],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluador de calidad de recomendación del sistema RAG."
    )
    parser.add_argument("--golden",  type=Path, default=GOLDEN_PATH)
    parser.add_argument("--db",      type=Path, default=DB_PATH)
    parser.add_argument("--chroma",  type=Path, default=CHROMA_PATH)
    parser.add_argument("--output",  type=Path, default=OUTPUT_PATH)
    parser.add_argument("--top",     type=int,  default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    resultados = evaluar(args.golden, args.db, args.chroma, args.top)
    imprimir_informe(resultados, verbose=args.verbose)
    guardar_json(resultados, args.output)


if __name__ == "__main__":
    main()
