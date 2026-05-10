"""
Construye el perfil del inversor a partir de las respuestas del diálogo.

Responsabilidades:
  1. Almacenar las respuestas crudas del usuario (texto libre).
  2. Mapear cada respuesta a las dimensiones de USER_PROFILE_VECTOR_SCHEMA.
  3. Exponer to_vector() con el mismo orden que fund_vectorizer.VECTOR_DIMS
     para que la similitud coseno sea directamente comparable.
  4. Exponer los campos de filtro duro (capital, nivel_riesgo_max, horizonte_anios).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from scoring.fund_vectorizer import HORIZON_MAX_YEARS, VECTOR_DIMS

# ── Helpers de parsing ────────────────────────────────────────────────────────

_MILES = re.compile(r"(\d[\d\s.,]*)(?:k\b|\.000)", re.IGNORECASE)
_NUMERO = re.compile(r"\d[\d\s.,]*")


def _parse_euros(texto: str) -> float | None:
    """
    Extrae un valor en euros de texto libre.
    Entiende '10k', '10.000', '10,000', '10 000', '€50', etc.
    """
    texto = texto.replace("€", "").replace("$", "").strip()

    # '10k' o '50.000' → miles
    m = _MILES.search(texto)
    if m:
        base = re.sub(r"[\s.]", "", m.group(1)).replace(",", ".")
        try:
            return float(base) * 1000
        except ValueError:
            pass

    # Número genérico (coma decimal española)
    m = _NUMERO.search(texto)
    if m:
        num_str = m.group().replace(" ", "").replace(".", "").replace(",", ".")
        try:
            return float(num_str)
        except ValueError:
            pass

    return None


def _parse_years(texto: str) -> float | None:
    """
    Extrae un horizonte temporal en años de texto libre.
    'más de 5 años' → 6, 'largo plazo' → 10, '3-5 años' → 4, '6 meses' → 0.5.
    """
    texto_lower = texto.lower()

    if any(p in texto_lower for p in ("largo plazo", "largo plazo", "indefinido", "siempre")):
        return 10.0
    if any(p in texto_lower for p in ("no sé", "no lo sé", "indiferente")):
        return 5.0

    # '6 meses' → 0.5
    m = re.search(r"(\d+)\s*mes", texto_lower)
    if m:
        return int(m.group(1)) / 12.0

    # 'más de X años'
    m = re.search(r"m[aá]s\s+de\s+(\d+)", texto_lower)
    if m:
        return float(m.group(1)) + 1

    # Rango 'X-Y años' → promedio
    m = re.search(r"(\d+)\s*[-–a]\s*(\d+)\s*a[ñn]", texto_lower)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2

    # Número simple seguido de 'año'
    m = re.search(r"(\d+)\s*a[ñn]", texto_lower)
    if m:
        return float(m.group(1))

    return None


def _parse_risk(texto: str) -> float:
    """
    Mapea la respuesta sobre tolerancia al riesgo a [0.0, 1.0].
    """
    texto_lower = texto.lower()
    señales_bajas = ("retiraría", "preocuparía", "conservador", "proteger", "seguro", "poco riesgo")
    señales_altas = ("oportunidad", "invertiría más", "agresivo", "maximizar", "decidido", "arriesgar")
    señales_medias = ("mantendría", "esperaría", "moderado", "crecimiento moderado")

    puntos_altos = sum(1 for s in señales_altas if s in texto_lower)
    puntos_bajos = sum(1 for s in señales_bajas if s in texto_lower)
    puntos_medios = sum(1 for s in señales_medias if s in texto_lower)

    total = puntos_altos + puntos_bajos + puntos_medios
    if total == 0:
        return 0.5   # neutral por defecto

    score = (puntos_altos * 0.9 + puntos_medios * 0.5 + puntos_bajos * 0.1) / total
    return round(min(max(score, 0.0), 1.0), 3)


def _parse_liquidez(texto: str) -> float:
    """
    Mapea la respuesta sobre liquidez a [0.0, 1.0].
    1.0 = necesita liquidez inmediata, 0.0 = no le importa bloquear.
    """
    texto_lower = texto.lower()
    señales_alta = ("cualquier momento", "necesito", "siempre", "inmediata", "retirarlo cuando")
    señales_baja = ("bloquearlo años", "no me importa", "sin liquidez", "largo plazo")
    señales_media = ("6 meses", "bloquear", "tiempo", "podría bloquearlo")

    if any(s in texto_lower for s in señales_alta):
        return 0.9
    if any(s in texto_lower for s in señales_baja):
        return 0.1
    if any(s in texto_lower for s in señales_media):
        return 0.4
    return 0.5


def _parse_esg(texto: str) -> float:
    """Mapea la sensibilidad ESG a [0.0, 1.0]."""
    texto_lower = texto.lower()
    señales_alta = ("sostenible", "esg", "verde", "responsable", "ético", "clima", "renovable")
    señales_baja = ("no me importa", "indiferente", "da igual", "no interesa")

    if any(s in texto_lower for s in señales_alta):
        return 0.9
    if any(s in texto_lower for s in señales_baja):
        return 0.0
    return 0.3   # por defecto leve indiferencia


def _parse_gestion(texto: str) -> float:
    """
    Mapea la preferencia de gestión activa/pasiva a [0.0, 1.0].
    1.0 = activa, 0.0 = pasiva.
    """
    texto_lower = texto.lower()
    señales_activa = ("activa", "superar", "gestor", "alguien gestione", "batir")
    señales_pasiva = ("pasiva", "índice", "etf", "menos comisiones", "automático")

    if any(s in texto_lower for s in señales_activa):
        return 0.9
    if any(s in texto_lower for s in señales_pasiva):
        return 0.1
    return 0.5


# ── Dataclass del perfil ──────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """
    Perfil del inversor construido progresivamente durante el diálogo.
    Todos los campos float están normalizados [0.0, 1.0] salvo capital_disponible.
    """

    # Dimensiones vectoriales (alineadas con VECTOR_DIMS)
    tolerancia_riesgo:          float = 0.5
    horizonte_temporal:         float = 0.5
    necesidad_liquidez:         float = 0.5
    sensibilidad_esg:           float = 0.3
    preferencia_gestion_activa: float = 0.5

    # Filtros duros (no se normalizan)
    capital_disponible:     float = 0.0     # en euros
    horizonte_anios:        float = 5.0     # años absolutos para filtro duro
    nivel_riesgo_max:       int   = 7       # nivel_riesgo máximo aceptado (1-7)

    # Texto libre para búsqueda semántica RAG
    preferencia_geografica: str = ""
    preferencia_sectorial:  str = ""

    # Estado interno: qué preguntas se han procesado
    _respondidas: set[str] = field(default_factory=set, repr=False)

    # ── Métodos de actualización ──────────────────────────────────────────────

    def update_capital(self, respuesta: str) -> None:
        """Actualiza capital_disponible desde texto libre."""
        euros = _parse_euros(respuesta)
        if euros is not None:
            self.capital_disponible = euros
            self._respondidas.add("capital")

    def update_horizonte(self, respuesta: str) -> None:
        """Actualiza horizonte_temporal y horizonte_anios desde texto libre."""
        anios = _parse_years(respuesta)
        if anios is not None:
            self.horizonte_anios = anios
            self.horizonte_temporal = min(anios / HORIZON_MAX_YEARS, 1.0)
            self._respondidas.add("horizonte")

    def update_riesgo(self, respuesta: str) -> None:
        """
        Actualiza tolerancia_riesgo y nivel_riesgo_max.
        nivel_riesgo_max = ceil(tolerancia * 6) + 1 → escala 1-7.
        """
        import math
        self.tolerancia_riesgo = _parse_risk(respuesta)
        self.nivel_riesgo_max = min(7, math.ceil(self.tolerancia_riesgo * 6) + 1)
        self._respondidas.add("riesgo")

    def update_liquidez(self, respuesta: str) -> None:
        """Actualiza necesidad_liquidez desde texto libre."""
        self.necesidad_liquidez = _parse_liquidez(respuesta)
        self._respondidas.add("liquidez")

    def update_tematica(self, respuesta: str) -> None:
        """
        Extrae preferencias geográficas y sectoriales como texto libre.
        Se usarán en la búsqueda semántica RAG sobre politica_inversion.
        """
        self.preferencia_geografica = respuesta
        self.preferencia_sectorial = respuesta
        self._respondidas.add("tematica")

    def update_estrategia(self, respuesta: str) -> None:
        """Actualiza sensibilidad_esg y preferencia_gestion_activa."""
        self.sensibilidad_esg = _parse_esg(respuesta)
        self.preferencia_gestion_activa = _parse_gestion(respuesta)
        self._respondidas.add("estrategia")

    # ── Despacho genérico por ID de pregunta ─────────────────────────────────

    def update(self, question_id: str, respuesta: str) -> None:
        """
        Actualiza el perfil a partir del ID de pregunta definido en
        INVESTOR_PROFILE_QUESTIONS y la respuesta en texto libre del usuario.
        """
        dispatch = {
            "capital":    self.update_capital,
            "horizonte":  self.update_horizonte,
            "riesgo":     self.update_riesgo,
            "liquidez":   self.update_liquidez,
            "tematica":   self.update_tematica,
            "estrategia": self.update_estrategia,
        }
        handler = dispatch.get(question_id)
        if handler:
            handler(respuesta)

    # ── Exportación ───────────────────────────────────────────────────────────

    def to_vector(self) -> np.ndarray:
        """
        Devuelve un vector numpy float32 con el mismo orden que VECTOR_DIMS:
          [tolerancia_riesgo, horizonte_temporal, necesidad_liquidez,
           sensibilidad_esg, preferencia_gestion_activa]
        """
        return np.array(
            [
                self.tolerancia_riesgo,
                self.horizonte_temporal,
                self.necesidad_liquidez,
                self.sensibilidad_esg,
                self.preferencia_gestion_activa,
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        """Exporta el perfil completo como diccionario serializable."""
        return {
            "tolerancia_riesgo":          self.tolerancia_riesgo,
            "horizonte_temporal":         self.horizonte_temporal,
            "necesidad_liquidez":         self.necesidad_liquidez,
            "sensibilidad_esg":           self.sensibilidad_esg,
            "preferencia_gestion_activa": self.preferencia_gestion_activa,
            "capital_disponible":         self.capital_disponible,
            "horizonte_anios":            self.horizonte_anios,
            "nivel_riesgo_max":           self.nivel_riesgo_max,
            "preferencia_geografica":     self.preferencia_geografica,
            "preferencia_sectorial":      self.preferencia_sectorial,
            "preguntas_respondidas":      sorted(self._respondidas),
        }

    @property
    def perfil_completo(self) -> bool:
        """True si se han respondido las 6 preguntas del flujo."""
        return len(self._respondidas) >= 6
