"""
Interfaz web Streamlit para el asesor conversacional Fondos AI.

Uso local:
  streamlit run app.py

Despliegue:
  docker build -t fondos-ai .
  docker run -p 8501:8501 -e GEMINI_API_KEY=tu_clave fondos-ai
"""

import streamlit as st
from pathlib import Path

# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Fondos AI — Asesor de Inversión",
    page_icon="📈",
    layout="centered",
)

st.title("📈 Fondos AI")
st.caption("Asesor de fondos de inversión basado en tus preferencias personales")

# ── Comprobación de base de datos ─────────────────────────────────────────────

DB_PATH = Path("database/funds.db")

if not DB_PATH.exists():
    st.error(
        "⚠️ No se encontró la base de datos de fondos (`database/funds.db`). "
        "Asegúrate de haber ejecutado primero el pipeline de extracción:\n\n"
        "```bash\npython main.py --mode extract\n```"
    )
    st.stop()

# ── Inicialización del asesor (una sola vez por sesión) ───────────────────────

if "advisor" not in st.session_state:
    from chatbot.conversation import FondosAdvisor
    with st.spinner("Iniciando el asesor..."):
        advisor = FondosAdvisor(top_n=5, db_path=DB_PATH)
        bienvenida = advisor.saludo_inicial()
    st.session_state.advisor = advisor
    st.session_state.messages = [
        {"role": "assistant", "content": bienvenida}
    ]

# ── Historial de mensajes ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# ── Input del usuario ─────────────────────────────────────────────────────────

if prompt := st.chat_input("Escribe tu respuesta aquí..."):
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Pensando..."):
            respuesta = st.session_state.advisor.chat(prompt)
        st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})

# ── Botón para reiniciar la conversación ──────────────────────────────────────

st.divider()
if st.button("🔄 Reiniciar conversación", use_container_width=False):
    del st.session_state.advisor
    del st.session_state.messages
    st.rerun()
