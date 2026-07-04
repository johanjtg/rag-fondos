"""
Chatbot conversacional para el perfilado del inversor y recomendación de fondos.

Flujo:
  1. Saludo e introducción.
  2. 6 preguntas secuenciales de INVESTOR_PROFILE_QUESTIONS.
     - El LLM extrae las dimensiones del perfil de cada respuesta libre.
     - UserProfile se actualiza tras cada respuesta.
  3. Resumen del perfil detectado; confirmación del usuario.
  4. Llamada a scorer.get_top_funds() → top 5 fondos.
  5. Presentación de resultados con justificación en lenguaje natural.
  6. Modo libre: el usuario puede hacer preguntas de seguimiento.

Uso:
  python chatbot/conversation.py
  python chatbot/conversation.py --top 3 --db database/funds.db
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from chatbot.questions import INVESTOR_PROFILE_QUESTIONS
from scoring.scorer import FundScore, get_top_funds
from scoring.user_profiler import UserProfile

load_dotenv()

log = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"

# ── Prompts del sistema ───────────────────────────────────────────────────────

SYSTEM_PERFILADO = dedent("""\
    Eres un asesor financiero profesional y empático llamado Fondos AI.
    Ayudas a inversores españoles a encontrar fondos de inversión adecuados a su perfil.

    Normas de comportamiento:
    - Habla siempre en español, con un tono cálido y profesional.
    - Estás haciendo una entrevista estructurada: haz UNA sola pregunta a la vez.
    - Cuando el usuario responda, acusa recibo brevemente (1-2 frases) y pasa
      a la siguiente pregunta sin repetir preguntas ya respondidas.
    - Si la respuesta es ambigua, pide una aclaración concreta antes de continuar.
    - No des recomendaciones de inversión hasta que se complete el perfilado.
    - No menciones nombres de fondos específicos durante el perfilado.
""")

SYSTEM_RECOMENDACION = dedent("""\
    Eres un asesor financiero profesional llamado Fondos AI.
    Acabas de completar el perfilado de un inversor y tienes los resultados
    del análisis cuantitativo. Tu tarea es presentar los fondos recomendados
    de forma clara, honesta y personalizada.

    Normas:
    - Habla siempre en español.
    - Para cada fondo, explica en 2-3 frases POR QUÉ encaja con el perfil del usuario,
      usando los datos concretos disponibles (riesgo, horizonte, ESG, gestión, política).
    - Menciona el score como "puntuación de compatibilidad" en porcentaje (score * 100).
    - Sé honesto si algún fondo tiene datos incompletos.
    - Al final, ofrece responder preguntas sobre cualquiera de los fondos.
""")


# ── Construcción del LLM y memoria ───────────────────────────────────────────

def _build_llm(temperature: float = 0.4) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY no está definida en el entorno.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )


def _build_chain(llm: ChatGoogleGenerativeAI, system_prompt: str):
    """
    Construye una cadena LangChain con historial de conversación en memoria.
    Devuelve (chain, history) para poder inspeccionar el historial.
    """
    history = InMemoryChatMessageHistory()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    return chain, history


# ── Helpers de presentación ───────────────────────────────────────────────────

def _separador(char: str = "─", ancho: int = 60) -> str:
    return char * ancho


def _formatear_fondos(fondos: list[FundScore], perfil: UserProfile) -> str:
    """
    Construye el prompt que se envía al LLM para que genere la presentación
    de resultados en lenguaje natural.
    """
    lineas = ["PERFIL DEL INVERSOR DETECTADO:"]
    lineas.append(f"  Capital: {perfil.capital_disponible:,.0f} €")
    lineas.append(f"  Horizonte: {perfil.horizonte_anios:.1f} años")
    lineas.append(f"  Tolerancia al riesgo: {perfil.tolerancia_riesgo:.0%}")
    lineas.append(f"  Nivel de riesgo máximo: {perfil.nivel_riesgo_max}/7")
    lineas.append(f"  Necesidad de liquidez: {perfil.necesidad_liquidez:.0%}")
    lineas.append(f"  Sensibilidad ESG: {perfil.sensibilidad_esg:.0%}")
    gestion = "activa" if perfil.preferencia_gestion_activa > 0.6 else (
        "pasiva" if perfil.preferencia_gestion_activa < 0.4 else "indiferente"
    )
    lineas.append(f"  Gestión preferida: {gestion}")
    if perfil.preferencia_geografica:
        lineas.append(f"  Preferencia temática: {perfil.preferencia_geografica}")
    lineas.append("")
    lineas.append(f"TOP {len(fondos)} FONDOS RECOMENDADOS:")

    for i, f in enumerate(fondos, 1):
        lineas.append(f"\n{i}. {f.nombre_fondo}")
        lineas.append(f"   ISIN: {f.isin}")
        lineas.append(f"   Gestora: {f.gestora}")
        lineas.append(f"   Puntuación de compatibilidad: {f.score_total * 100:.1f}%")
        lineas.append(f"   Desglose: coseno={f.score_coseno:.3f} | semántico={f.score_semantico:.3f} | ESG boost={f.esg_boost:.2f}")
        lineas.append(f"   Riesgo: {f.nivel_riesgo or '—'}/7 | Gestión: {f.tipo_gestion or '—'} | ESG: {f.esg} | Horizonte: {f.horizonte_anios or '—'} años")
        if f.politica_inversion:
            extracto = f.politica_inversion[:300].replace("\n", " ")
            lineas.append(f"   Política: {extracto}{'…' if len(f.politica_inversion) > 300 else ''}")

    lineas.append("\nPresenta estos resultados al usuario de forma clara y personalizada.")
    return "\n".join(lineas)


def _resumen_perfil(perfil: UserProfile) -> str:
    """Genera un resumen legible del perfil para confirmar con el usuario."""
    gestion = "activa" if perfil.preferencia_gestion_activa > 0.6 else (
        "pasiva" if perfil.preferencia_gestion_activa < 0.4 else "sin preferencia clara"
    )
    riesgo_texto = (
        "muy conservador" if perfil.tolerancia_riesgo < 0.25 else
        "conservador" if perfil.tolerancia_riesgo < 0.45 else
        "moderado" if perfil.tolerancia_riesgo < 0.65 else
        "dinámico" if perfil.tolerancia_riesgo < 0.85 else
        "muy agresivo"
    )
    esg_texto = (
        "muy importante" if perfil.sensibilidad_esg > 0.7 else
        "algo importante" if perfil.sensibilidad_esg > 0.4 else
        "no prioritaria"
    )

    lineas = [
        "Antes de buscar los fondos, permíteme confirmar tu perfil:",
        f"  • Capital disponible: {perfil.capital_disponible:,.0f} €",
        f"  • Horizonte de inversión: {perfil.horizonte_anios:.0f} años",
        f"  • Perfil de riesgo: {riesgo_texto} (nivel máximo {perfil.nivel_riesgo_max}/7)",
        f"  • Necesidad de liquidez: {'alta' if perfil.necesidad_liquidez > 0.6 else 'baja'}",
        f"  • Sostenibilidad ESG: {esg_texto}",
        f"  • Gestión preferida: {gestion}",
    ]
    if perfil.preferencia_geografica:
        lineas.append(f"  • Preferencias temáticas: {perfil.preferencia_geografica}")
    lineas.append("\n¿Es correcto? ¿Hay algo que quieras ajustar antes de ver los resultados?")
    return "\n".join(lineas)


# ── Bucle principal ───────────────────────────────────────────────────────────

class FondosAdvisor:
    """
    Orquestador del flujo conversacional de perfilado y recomendación.

    Estados internos:
      PERFILANDO   → preguntando las 6 preguntas
      CONFIRMANDO  → mostrando resumen del perfil y esperando confirmación
      RECOMENDANDO → presentando los fondos y respondiendo preguntas de seguimiento
    """

    ESTADO_PERFILANDO   = "perfilando"
    ESTADO_CONFIRMANDO  = "confirmando"
    ESTADO_RECOMENDANDO = "recomendando"

    def __init__(self, top_n: int = 5, db_path: Path = Path("database/funds.db")):
        self.top_n = top_n
        self.db_path = db_path
        self.perfil = UserProfile()
        self.estado = self.ESTADO_PERFILANDO
        self.pregunta_idx = 0
        self.historial: list[dict] = []   # {role, content} para memoria manual

        self.llm_perfilado = _build_llm(temperature=0.3)
        self.llm_recomendacion = _build_llm(temperature=0.5)

        self.chain_perfilado, self.history_perfilado = _build_chain(
            self.llm_perfilado, SYSTEM_PERFILADO
        )
        self.chain_recomendacion, self.history_recomendacion = _build_chain(
            self.llm_recomendacion, SYSTEM_RECOMENDACION
        )

    # ── Llamadas al LLM ───────────────────────────────────────────────────────

    def _invoke_perfilado(self, mensaje: str) -> str:
        """Invoca la cadena de perfilado añadiendo el mensaje al historial."""
        self.history_perfilado.add_user_message(mensaje)
        respuesta = self.chain_perfilado.invoke({
            "input": mensaje,
            "history": self.history_perfilado.messages[:-1],  # excluir el último (ya en {input})
        })
        texto = respuesta.content
        self.history_perfilado.add_ai_message(texto)
        return texto

    def _invoke_recomendacion(self, mensaje: str) -> str:
        """Invoca la cadena de recomendación con memoria propia."""
        self.history_recomendacion.add_user_message(mensaje)
        respuesta = self.chain_recomendacion.invoke({
            "input": mensaje,
            "history": self.history_recomendacion.messages[:-1],
        })
        texto = respuesta.content
        self.history_recomendacion.add_ai_message(texto)
        return texto

    # ── Máquina de estados ────────────────────────────────────────────────────

    def _siguiente_pregunta(self) -> str:
        """
        Pide al LLM que haga la siguiente pregunta del flujo de perfilado
        de forma natural, usando el contexto del historial previo.
        """
        q = INVESTOR_PROFILE_QUESTIONS[self.pregunta_idx]
        instruccion = (
            f"Haz ahora la siguiente pregunta del perfilado de forma natural. "
            f"La pregunta base es: «{q['pregunta']}». "
            f"Adapta el tono al contexto de la conversación, pero no cambies "
            f"el significado ni añadas preguntas adicionales."
        )
        return self._invoke_perfilado(instruccion)

    def _procesar_respuesta_perfilado(self, respuesta_usuario: str) -> str:
        """
        Actualiza el perfil con la respuesta recibida, avanza al siguiente
        estado y devuelve la respuesta del asistente.
        """
        q = INVESTOR_PROFILE_QUESTIONS[self.pregunta_idx]
        from chatbot.llm_profiler import actualizar_perfil_llm
        actualizar_perfil_llm(self.llm_perfilado, self.perfil, q["id"], q["pregunta"], respuesta_usuario)

        self.pregunta_idx += 1
        hay_mas_preguntas = self.pregunta_idx < len(INVESTOR_PROFILE_QUESTIONS)

        if hay_mas_preguntas:
            # Acusar recibo + siguiente pregunta en un solo turno
            acuse = (
                f"El usuario ha respondido: «{respuesta_usuario}». "
                f"Acusa recibo brevemente (máximo 2 frases) y formula "
                f"la siguiente pregunta: «{INVESTOR_PROFILE_QUESTIONS[self.pregunta_idx]['pregunta']}». "
                f"Hazlo de forma natural y fluida, sin numerarlas."
            )
            return self._invoke_perfilado(acuse)
        else:
            # Todas las preguntas respondidas → mostrar resumen
            self.estado = self.ESTADO_CONFIRMANDO
            resumen = _resumen_perfil(self.perfil)
            acuse = (
                f"El usuario ha respondido: «{respuesta_usuario}». "
                f"Acusa recibo brevemente y muestra este resumen del perfil tal cual:\n\n"
                f"{resumen}"
            )
            return self._invoke_perfilado(acuse)

    def _procesar_confirmacion(self, respuesta_usuario: str) -> str:
        """
        Interpreta si el usuario confirma el perfil o pide correcciones.
        Si confirma, ejecuta el scoring y pasa a modo recomendación.
        """
        resp_lower = respuesta_usuario.lower()
        confirma = any(p in resp_lower for p in (
            "sí", "si", "correcto", "adelante", "perfecto", "ok",
            "bien", "exacto", "así es", "de acuerdo", "continúa", "continua"
        ))
        corrige = any(p in resp_lower for p in (
            "no", "incorrecto", "cambiar", "modificar", "ajustar", "error",
            "equivocado", "mal", "diferente"
        ))

        if corrige:
            # Volver a tomar el control del perfilado manualmente
            self.estado = self.ESTADO_PERFILANDO
            self.pregunta_idx = 0   # reiniciar — sencillo y seguro
            return self._invoke_perfilado(
                f"El usuario quiere corregir su perfil: «{respuesta_usuario}». "
                f"Indícale que vamos a repasar las preguntas de nuevo y comienza "
                f"con la primera pregunta de forma natural."
            )

        # Si confirma (o respuesta ambigua → beneficio de la duda)
        self.estado = self.ESTADO_RECOMENDANDO
        return self._ejecutar_scoring()

    def _ejecutar_scoring(self) -> str:
        """Llama al motor de scoring y genera la presentación con el LLM."""
        print("\n[Buscando los mejores fondos para tu perfil...]\n")
        try:
            fondos = get_top_funds(
                perfil=self.perfil,
                top_n=self.top_n,
                db_path=self.db_path,
            )
        except FileNotFoundError:
            return (
                "Lo siento, no pude acceder a la base de datos de fondos. "
                "Asegúrate de haber ejecutado el pipeline de extracción primero "
                "(python extraction/pdf_extractor.py --input data/dfi_pdfs/)."
            )

        if not fondos:
            return self._invoke_recomendacion(
                "No se ha encontrado ningún fondo que supere los filtros de "
                "compatibilidad con el perfil del usuario. Explícale esto con "
                "empatía y sugiere que podría relajar alguna restricción, por "
                "ejemplo ampliando el horizonte o aumentando la tolerancia al riesgo."
            )

        prompt_resultados = _formatear_fondos(fondos, self.perfil)
        return self._invoke_recomendacion(prompt_resultados)

    def _procesar_seguimiento(self, pregunta_usuario: str) -> str:
        """Responde preguntas de seguimiento sobre los fondos recomendados."""
        return self._invoke_recomendacion(pregunta_usuario)

    # ── Punto de entrada público ──────────────────────────────────────────────

    def chat(self, mensaje_usuario: str) -> str:
        """
        Procesa un mensaje del usuario y devuelve la respuesta del asistente.
        Gestiona automáticamente las transiciones de estado.
        """
        if self.estado == self.ESTADO_PERFILANDO:
            return self._procesar_respuesta_perfilado(mensaje_usuario)
        elif self.estado == self.ESTADO_CONFIRMANDO:
            return self._procesar_confirmacion(mensaje_usuario)
        else:  # RECOMENDANDO
            return self._procesar_seguimiento(mensaje_usuario)

    def saludo_inicial(self) -> str:
        """Genera el mensaje de bienvenida e inicia con la primera pregunta."""
        intro = (
            "Saluda al usuario de forma cálida como Fondos AI. "
            "Explica en 2-3 frases que le vas a hacer 6 preguntas rápidas "
            "para conocer su perfil inversor y encontrar los fondos más adecuados. "
            "Acto seguido, formula la primera pregunta: "
            f"«{INVESTOR_PROFILE_QUESTIONS[0]['pregunta']}». "
            "Todo en un solo mensaje fluido."
        )
        return self._invoke_perfilado(intro)


# ── CLI interactivo ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chatbot asesor de fondos de inversión españoles."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Número de fondos a recomendar (por defecto: 5).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("database/funds.db"),
        metavar="PATH",
        help="Ruta de la base de datos SQLite.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activar logging DEBUG.",
    )
    return parser.parse_args()


def run_cli(top_n: int = 5, db_path: Path = Path("database/funds.db")) -> None:
    """Bucle REPL de la conversación en terminal."""
    advisor = FondosAdvisor(top_n=top_n, db_path=db_path)

    print(_separador("═"))
    print("  FONDOS AI — Asesor de Inversión")
    print(_separador("═"))
    print("  (escribe 'salir' para terminar)\n")

    # Saludo inicial + primera pregunta
    bienvenida = advisor.saludo_inicial()
    print(f"Fondos AI: {bienvenida}\n")

    while True:
        try:
            entrada = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nHasta pronto.")
            break

        if not entrada:
            continue
        if entrada.lower() in ("salir", "exit", "quit", "q"):
            print("\nFondos AI: Ha sido un placer ayudarte. ¡Hasta pronto!")
            break

        respuesta = advisor.chat(entrada)
        print(f"\nFondos AI: {respuesta}\n")
        print(_separador())


if __name__ == "__main__":
    args = _parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    run_cli(top_n=args.top, db_path=args.db)
