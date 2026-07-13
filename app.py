"""
Punto de entrada principal de la aplicación Fondos AI.
Streamlit multipage — las páginas están en pages/
"""

import streamlit as st
from version import VERSION, VERSION_DATE, VERSION_NOTES

st.set_page_config(
    page_title="Fondos AI",
    page_icon="📈",
    layout="centered",
)

st.sidebar.markdown(
    f"**Fondos AI** `v{VERSION}`  \n"
    f"_{VERSION_DATE}_",
)

chatbot_page   = st.Page("pages/chatbot.py",    title="Asesor",     icon="💬", default=True)
evaluacion_page = st.Page("pages/evaluacion.py", title="Evaluación", icon="📊")

nav = st.navigation([chatbot_page, evaluacion_page])
nav.run()
