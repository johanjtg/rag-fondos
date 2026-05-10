"""
Preguntas del flujo conversacional para el perfilado del inversor.
Cada pregunta mapea directamente a uno o más campos del vector de perfil
y a los parámetros estructurados del FundModel.
"""

INVESTOR_PROFILE_QUESTIONS = [

    {
        "id": "capital",
        "dimension": "capital_disponible",
        "pregunta": (
            "Para empezar a buscar las mejores opciones para ti, "
            "¿cuál es el capital inicial aproximado que tienes pensado invertir?"
        ),
        "ejemplos_respuesta": ["5000 euros", "unos 10k", "50.000€", "menos de 1000"],
        "mapeo_campo_fondo": "importe_minimo_inversion",
        "tipo_filtro": "structured",  # filtro exacto en BD estructurada
        "nota": "Extraer valor numérico en euros. Descartar fondos con importe_minimo > capital."
    },

    {
        "id": "horizonte",
        "dimension": "horizonte_temporal",
        "pregunta": (
            "¿Durante cuánto tiempo planeas mantener este dinero invertido "
            "sin necesitar retirarlo?"
        ),
        "ejemplos_respuesta": ["1 año", "más de 5 años", "a largo plazo", "no lo sé"],
        "mapeo_campo_fondo": "horizonte_recomendado_anios",
        "tipo_filtro": "structured",
        "nota": "Normalizar a años. Penalizar fondos cuyo horizonte recomendado supere el del usuario."
    },

    {
        "id": "riesgo",
        "dimension": "tolerancia_riesgo",
        "pregunta": (
            "Si el mercado sufre una caída y tu inversión pierde un 10% de su valor "
            "en un mes, ¿cómo reaccionarías? "
            "¿Y cuál es tu objetivo principal: proteger tus ahorros, "
            "buscar crecimiento moderado o maximizar rentabilidad?"
        ),
        "ejemplos_respuesta": [
            "me preocuparía mucho y retiraría el dinero",
            "lo mantendría y esperaría",
            "lo vería como oportunidad para invertir más"
        ],
        "mapeo_campo_fondo": ["nivel_riesgo", "perfil_riesgo", "perfil_inversor"],
        "tipo_filtro": "structured",
        "nota": (
            "Mapear respuesta a escala 0.0-1.0: "
            "retiraría=0.1, esperaría=0.5, invertiría más=0.9. "
            "Convertir a nivel_riesgo 1-7 para filtro duro máximo."
        )
    },

    {
        "id": "liquidez",
        "dimension": "necesidad_liquidez",
        "pregunta": (
            "¿Es importante para ti poder retirar tu dinero en cualquier momento, "
            "o estarías dispuesto a mantenerlo bloqueado un tiempo "
            "a cambio de mejores condiciones?"
        ),
        "ejemplos_respuesta": [
            "necesito poder retirarlo cuando quiera",
            "podría bloquearlo 6 meses",
            "no me importa bloquearlo años si la rentabilidad es buena"
        ],
        "mapeo_campo_fondo": "restricciones_liquidez",
        "tipo_filtro": "structured",
        "nota": "Alta necesidad de liquidez (>0.7) → descartar fondos con restricciones de bloqueo."
    },

    {
        "id": "tematica",
        "dimension": ["preferencia_geografica", "preferencia_sectorial"],
        "pregunta": (
            "¿Tienes alguna preferencia sobre dónde invertir tu dinero? "
            "¿Te interesa alguna región específica (Europa, EEUU, Asia, global...) "
            "o algún sector en particular (tecnología, salud, energía renovable...)?"
        ),
        "ejemplos_respuesta": [
            "me interesa la tecnología americana",
            "prefiero Europa y algo sostenible",
            "no tengo preferencia, lo que mejor rinda"
        ],
        "mapeo_campo_fondo": ["distribucion_geografica", "distribucion_sectorial", "politica_inversion"],
        "tipo_filtro": "semantic",  # búsqueda vectorial RAG sobre politica_inversion
        "nota": (
            "Este filtro usa RAG semántico sobre el campo politica_inversion. "
            "Convertir respuesta del usuario en embedding y calcular similitud coseno."
        )
    },

    {
        "id": "estrategia",
        "dimension": ["preferencia_gestion_activa", "sensibilidad_esg"],
        "pregunta": (
            "¿Prefieres fondos que intenten superar al mercado (gestión activa) "
            "o fondos que sigan automáticamente a un índice con menores costes (gestión pasiva)? "
            "¿Y te importa que el fondo siga criterios de inversión sostenible (ESG)?"
        ),
        "ejemplos_respuesta": [
            "gestión pasiva, menos comisiones",
            "activa, quiero que alguien gestione mi dinero",
            "me da igual pero sí quiero que sea sostenible"
        ],
        "mapeo_campo_fondo": ["tipo_gestion", "esg", "comision_gestion"],
        "tipo_filtro": "structured",
        "nota": (
            "Gestión activa/pasiva → filtro exacto. "
            "ESG → boost de +0.1 en score final si usuario sensible y fondo ESG=True."
        )
    },
]


# ── Vector de perfil del usuario ────────────────────────────────────────────
# Dimensiones del vector normalizado [0.0 - 1.0]
# Se construye progresivamente durante el diálogo

USER_PROFILE_VECTOR_SCHEMA = {
    "tolerancia_riesgo":          "float [0.0=muy conservador, 1.0=muy agresivo]",
    "horizonte_temporal":         "float [0.0=corto plazo <1 año, 1.0=largo plazo >10 años]",
    "capital_disponible":         "float en euros (sin normalizar, se usa para filtro duro)",
    "necesidad_liquidez":         "float [0.0=no necesita liquidez, 1.0=necesita liquidez inmediata]",
    "sensibilidad_esg":           "float [0.0=indiferente, 1.0=muy importante]",
    "preferencia_gestion_activa": "float [0.0=prefiere pasiva, 1.0=prefiere activa]",
    "preferencia_geografica":     "str (texto libre para búsqueda semántica)",
    "preferencia_sectorial":      "str (texto libre para búsqueda semántica)",
}


# ── Fórmula de scoring final ─────────────────────────────────────────────────
SCORING_FORMULA = """
score_final(usuario, fondo) =
    0.60 * cosine_similarity(vector_usuario, vector_fondo)   ← compatibilidad estructural
  + 0.40 * semantic_similarity(tematica_usuario, politica_inversion_fondo)  ← RAG semántico
  + 0.10 * esg_boost  (si sensibilidad_esg > 0.6 y fondo.esg == True)

Filtros duros ANTES del scoring (descartan fondos directamente):
  - fondo.importe_minimo_inversion > usuario.capital_disponible  → EXCLUIR
  - fondo.nivel_riesgo > nivel_riesgo_maximo_usuario             → EXCLUIR
  - fondo.horizonte_recomendado_anios > usuario.horizonte_anios  → EXCLUIR
  - fondo.restricciones_liquidez != None y usuario.necesidad_liquidez > 0.7 → EXCLUIR
"""
