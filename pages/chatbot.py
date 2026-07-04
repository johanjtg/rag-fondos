"""
Página del chatbot conversacional Fondos AI.
"""

import streamlit as st
from pathlib import Path

DB_PATH = Path("database/funds.db")

st.title("📈 Fondos AI")
st.caption("Asesor de fondos de inversión basado en tu perfil personal")

if not DB_PATH.exists():
    st.error(
        "⚠️ No se encontró la base de datos de fondos (`database/funds.db`). "
        "Asegúrate de haber ejecutado primero el pipeline de extracción:\n\n"
        "```bash\npython main.py --mode extract\n```"
    )
    st.stop()

if "advisor" not in st.session_state:
    from chatbot.conversation import FondosAdvisor
    with st.spinner("Iniciando el asesor..."):
        advisor = FondosAdvisor(top_n=5, db_path=DB_PATH)
        bienvenida = advisor.saludo_inicial()
    st.session_state.advisor = advisor
    st.session_state.messages = [
        {"role": "assistant", "content": bienvenida}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu respuesta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Pensando..."):
            respuesta = st.session_state.advisor.chat(prompt)
        st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})

st.divider()
if st.button("🔄 Reiniciar conversación"):
    del st.session_state.advisor
    del st.session_state.messages
    st.rerun()
