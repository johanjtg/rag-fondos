"""
Extracción del perfil inversor mediante LLM con salida estructurada (Pydantic).

Reemplaza los parseadores deterministas de user_profiler.py por llamadas
a Gemini con structured output, lo que permite interpretar respuestas
ambiguas, cortas o en lenguaje natural ("Sí", "No", "depende"...).

Cada pregunta tiene su propio modelo Pydantic con las dimensiones que actualiza
y descripciones detalladas de la escala para guiar al LLM.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from scoring.user_profiler import UserProfile

log = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


# ── Modelos Pydantic por pregunta ─────────────────────────────────────────────

class CapitalExtract(BaseModel):
    capital_euros: float = Field(
        description=(
            "Capital disponible para invertir en euros. "
            "Interpreta abreviaciones: '10k'=10000, '1M'=1000000. "
            "Si no se menciona una cantidad concreta, devuelve 0.0."
        )
    )

class HorizonteExtract(BaseModel):
    horizonte_anios: float = Field(
        description=(
            "Horizonte de inversión en años. "
            "'corto plazo' o 'menos de 1 año' → 1.0, "
            "'3 años' → 3.0, "
            "'5 años' → 5.0, "
            "'largo plazo' o 'más de 10 años' → 10.0, "
            "'no lo sé' o 'indiferente' → 5.0."
        )
    )

class RiesgoExtract(BaseModel):
    tolerancia_riesgo: float = Field(
        description=(
            "Tolerancia al riesgo normalizada entre 0.0 y 1.0. "
            "0.0 = muy conservador (retiraría el dinero ante cualquier pérdida). "
            "0.5 = moderado (mantendría y esperaría recuperación). "
            "1.0 = muy agresivo (ve las caídas como oportunidad, quiere maximizar). "
            "'Sí' a proteger ahorros → 0.1. "
            "'Sí' a crecimiento moderado → 0.5. "
            "'Sí' a maximizar rentabilidad → 0.9."
        )
    )
    nivel_riesgo_max: int = Field(
        description=(
            "Nivel de riesgo máximo aceptado en escala 1-7 (SRRI). "
            "Derivado de tolerancia_riesgo: "
            "0.0-0.2 → 1-2, 0.3-0.5 → 3-4, 0.6-0.8 → 5-6, 0.9-1.0 → 7."
        )
    )

class LiquidezExtract(BaseModel):
    necesidad_liquidez: float = Field(
        description=(
            "Necesidad de liquidez inmediata normalizada entre 0.0 y 1.0. "
            "1.0 = necesita poder retirar el dinero en cualquier momento. "
            "0.0 = dispuesto a bloquear el dinero años sin problema. "
            "0.5 = indiferente o sin preferencia clara. "
            "'Sí' a poder retirar cuando quiera → 0.9. "
            "'No' a bloquear / acepta bloqueo → 0.1. "
            "'Depende' o ambiguo → 0.5."
        )
    )

class TematicaExtract(BaseModel):
    preferencia_geografica: str = Field(
        description=(
            "Preferencia geográfica expresada en texto libre. "
            "Ejemplos: 'Europa', 'EEUU', 'Asia', 'mercados emergentes', 'global'. "
            "Si no tiene preferencia, devuelve cadena vacía ''."
        )
    )
    preferencia_sectorial: str = Field(
        description=(
            "Preferencia sectorial expresada en texto libre. "
            "Ejemplos: 'tecnología', 'salud', 'energía renovable', 'financiero'. "
            "Si no tiene preferencia, devuelve cadena vacía ''."
        )
    )

class EstrategiaExtract(BaseModel):
    preferencia_gestion_activa: float = Field(
        description=(
            "Preferencia por gestión activa normalizada entre 0.0 y 1.0. "
            "1.0 = prefiere gestión activa (gestor que bate al mercado). "
            "0.0 = prefiere gestión pasiva (ETF, índice, menos comisiones). "
            "0.5 = indiferente. "
            "'Sí' a gestión activa → 0.9. "
            "'Sí' a gestión pasiva / ETF → 0.1."
        )
    )
    sensibilidad_esg: float = Field(
        description=(
            "Sensibilidad a criterios ESG (sostenibilidad) entre 0.0 y 1.0. "
            "1.0 = muy importante, es criterio prioritario. "
            "0.0 = completamente indiferente. "
            "0.3 = no es prioritario pero tampoco rechaza. "
            "'Sí' a ESG / sostenible / ético / responsable / verde / criterios sostenibles → 0.9. "
            "'No' o 'no me importa' → 0.0. "
            "Sin mención explícita → 0.3. "
            "IMPORTANTE: si la respuesta es combinada (gestión + sostenibilidad), "
            "extrae la sensibilidad ESG de forma independiente sin que la preferencia "
            "de gestión influya en este valor."
        )
    )


class ESGExtract(BaseModel):
    sensibilidad_esg: float = Field(
        description=(
            "Sensibilidad a criterios de sostenibilidad ESG entre 0.0 y 1.0. "
            "Céntrate ÚNICAMENTE en si el usuario quiere que el fondo sea sostenible, "
            "ignorando cualquier otra preferencia mencionada. "
            "Cualquier mención a: sostenible, ESG, ético, responsable, verde, "
            "criterios sociales, medioambientales → 0.9. "
            "Sin mención o indiferente → 0.3. "
            "Rechazo explícito → 0.0."
        )
    )


# ── Extractor principal ───────────────────────────────────────────────────────

SYSTEM_EXTRACTOR = (
    "Eres un sistema de extracción de datos financieros. "
    "Tu tarea es interpretar la respuesta de un usuario a una pregunta sobre su perfil inversor "
    "y extraer los valores numéricos correspondientes con precisión. "
    "Interpreta el sentido de la respuesta aunque sea breve o ambigua. "
    "Devuelve siempre valores dentro del rango especificado en cada campo."
)


def _extraer(llm: ChatGoogleGenerativeAI, pregunta: str, respuesta: str, modelo_pydantic):
    """Llama a Gemini con structured output y devuelve el modelo extraído."""
    structured_llm = llm.with_structured_output(modelo_pydantic)
    prompt = (
        f"Pregunta formulada al usuario: «{pregunta}»\n"
        f"Respuesta del usuario: «{respuesta}»\n\n"
        f"Extrae los valores del perfil inversor según las instrucciones de cada campo."
    )
    try:
        return structured_llm.invoke([
            ("system", SYSTEM_EXTRACTOR),
            ("human", prompt),
        ])
    except Exception as e:
        log.warning("Error en extracción LLM para '%s': %s", modelo_pydantic.__name__, e)
        return None


# ── Actualización del perfil con LLM ─────────────────────────────────────────

def actualizar_perfil_llm(
    llm: ChatGoogleGenerativeAI,
    perfil: UserProfile,
    question_id: str,
    pregunta_texto: str,
    respuesta_usuario: str,
) -> None:
    """
    Actualiza el UserProfile usando extracción LLM en lugar de regex/keywords.
    Si la extracción falla, hace fallback al parser determinista original.
    """
    import math

    if question_id == "capital":
        resultado = _extraer(llm, pregunta_texto, respuesta_usuario, CapitalExtract)
        if resultado and resultado.capital_euros > 0:
            perfil.capital_disponible = resultado.capital_euros
            perfil._respondidas.add("capital")
            log.debug("LLM capital: %.0f€", resultado.capital_euros)
        else:
            perfil.update_capital(respuesta_usuario)

    elif question_id == "horizonte":
        resultado = _extraer(llm, pregunta_texto, respuesta_usuario, HorizonteExtract)
        if resultado and resultado.horizonte_anios > 0:
            anios = max(UserProfile.HORIZONTE_MIN_ANIOS, min(resultado.horizonte_anios, UserProfile.HORIZONTE_MAX_ANIOS))
            perfil.horizonte_anios    = anios
            perfil.horizonte_temporal = min(anios / 10.0, 1.0)
            perfil._respondidas.add("horizonte")
            log.debug("LLM horizonte: %.1f años", anios)
        else:
            perfil.update_horizonte(respuesta_usuario)

    elif question_id == "riesgo":
        resultado = _extraer(llm, pregunta_texto, respuesta_usuario, RiesgoExtract)
        if resultado:
            perfil.tolerancia_riesgo = max(0.0, min(1.0, resultado.tolerancia_riesgo))
            perfil.nivel_riesgo_max  = max(1, min(7, resultado.nivel_riesgo_max))
            perfil._respondidas.add("riesgo")
            log.debug("LLM riesgo: %.2f (max nivel %d)", resultado.tolerancia_riesgo, resultado.nivel_riesgo_max)
        else:
            perfil.update_riesgo(respuesta_usuario)

    elif question_id == "liquidez":
        resultado = _extraer(llm, pregunta_texto, respuesta_usuario, LiquidezExtract)
        if resultado:
            perfil.necesidad_liquidez = max(0.0, min(1.0, resultado.necesidad_liquidez))
            perfil._respondidas.add("liquidez")
            log.debug("LLM liquidez: %.2f", resultado.necesidad_liquidez)
        else:
            perfil.update_liquidez(respuesta_usuario)

    elif question_id == "tematica":
        resultado = _extraer(llm, pregunta_texto, respuesta_usuario, TematicaExtract)
        if resultado:
            perfil.preferencia_geografica = resultado.preferencia_geografica
            perfil.preferencia_sectorial  = resultado.preferencia_sectorial
            perfil._respondidas.add("tematica")
            log.debug("LLM tematica: geo='%s' sector='%s'",
                      resultado.preferencia_geografica, resultado.preferencia_sectorial)
        else:
            perfil.update_tematica(respuesta_usuario)

    elif question_id == "estrategia":
        resultado = _extraer(llm, pregunta_texto, respuesta_usuario, EstrategiaExtract)
        if resultado:
            perfil.preferencia_gestion_activa = max(0.0, min(1.0, resultado.preferencia_gestion_activa))
            esg_principal = max(0.0, min(1.0, resultado.sensibilidad_esg))
            # Segunda llamada LLM enfocada exclusivamente en ESG para evitar que
            # respuestas combinadas (gestión + sostenibilidad) diluyan la señal ESG.
            esg_resultado = _extraer(llm, pregunta_texto, respuesta_usuario, ESGExtract)
            esg_focalizado = max(0.0, min(1.0, esg_resultado.sensibilidad_esg)) if esg_resultado else esg_principal
            perfil.sensibilidad_esg = max(esg_principal, esg_focalizado)
            perfil._respondidas.add("estrategia")
            log.debug(
                "LLM estrategia: gestion=%.2f esg_principal=%.2f esg_focalizado=%.2f → esg=%.2f",
                resultado.preferencia_gestion_activa, esg_principal, esg_focalizado, perfil.sensibilidad_esg,
            )
        else:
            perfil.update_estrategia(respuesta_usuario)
