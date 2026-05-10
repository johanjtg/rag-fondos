"""
Convierte un FundModel en un vector numérico normalizado [0.0 - 1.0].

El vector resultante comparte el mismo espacio dimensional que el vector
de perfil del usuario (UserProfile) para que la similitud coseno sea comparable.

Dimensiones del vector (VECTOR_DIMS, en orden):
  0  tolerancia_riesgo         ← nivel_riesgo / 7
  1  horizonte_temporal        ← horizonte_recomendado_anios / 10 (cap 1.0)
  2  necesidad_liquidez        ← 0.0 si sin restricciones, 1.0 si con restricciones
  3  sensibilidad_esg          ← 1.0 si esg=True, 0.0 si False/None
  4  preferencia_gestion_activa← 1.0 si activa, 0.0 si pasiva, 0.5 si None
"""

from __future__ import annotations

import numpy as np

from extraction.fund_model import FundModel

# Nombres de las dimensiones vectoriales (mismo orden que UserProfile.to_vector)
VECTOR_DIMS = [
    "tolerancia_riesgo",
    "horizonte_temporal",
    "necesidad_liquidez",
    "sensibilidad_esg",
    "preferencia_gestion_activa",
]

HORIZON_MAX_YEARS = 10.0   # horizonte máximo de normalización


def vectorize(fondo: FundModel) -> np.ndarray:
    """
    Devuelve un vector numpy float32 de longitud len(VECTOR_DIMS)
    con todos los valores en [0.0, 1.0].

    Campos sin dato (None) se imputan al valor neutro 0.5 salvo indicación contraria.
    """
    # ── 0: tolerancia_riesgo ─────────────────────────────────────────────────
    if fondo.nivel_riesgo is not None:
        tolerancia = (fondo.nivel_riesgo - 1) / 6.0   # escala 1-7 → 0.0-1.0
    else:
        tolerancia = 0.5

    # ── 1: horizonte_temporal ────────────────────────────────────────────────
    if fondo.horizonte_recomendado_anios is not None:
        horizonte = min(fondo.horizonte_recomendado_anios / HORIZON_MAX_YEARS, 1.0)
    else:
        horizonte = 0.5

    # ── 2: necesidad_liquidez ────────────────────────────────────────────────
    # El fondo tiene restricciones → requiere baja liquidez del inversor (valor alto)
    if fondo.restricciones_liquidez is not None and fondo.restricciones_liquidez.strip():
        liquidez = 0.2   # fondo con bloqueo → solo apto para inversores sin prisa
    else:
        liquidez = 0.8   # fondo líquido → compatible con cualquier necesidad

    # ── 3: sensibilidad_esg ──────────────────────────────────────────────────
    if fondo.esg is True:
        esg = 1.0
    elif fondo.esg is False:
        esg = 0.0
    else:
        esg = 0.5

    # ── 4: preferencia_gestion_activa ────────────────────────────────────────
    if fondo.tipo_gestion is not None:
        gestion = 1.0 if "activa" in fondo.tipo_gestion.lower() else 0.0
    else:
        gestion = 0.5

    return np.array([tolerancia, horizonte, liquidez, esg, gestion], dtype=np.float32)
