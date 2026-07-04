"""
Página de evaluación de calidad de extracción (Evaluador 1).
Muestra los resultados del golden dataset comparados contra la BD.
"""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

GOLDEN_PATH    = Path("evaluation/golden_dataset_extraccion.json")
RESULTADOS_PATH = Path("evaluation/resultados_extraccion.json")
DB_PATH        = Path("database/funds.db")

st.title("📊 Evaluación de Extracción")
st.caption("Calidad de los datos extraídos por Gemini comparados contra los valores reales del PDF")

# ── Comprobaciones previas ─────────────────────────────────────────────────────

if not GOLDEN_PATH.exists():
    st.error("No se encontró el golden dataset (`evaluation/golden_dataset_extraccion.json`).")
    st.stop()

if not DB_PATH.exists():
    st.error("No se encontró la base de datos (`database/funds.db`).")
    st.stop()

# ── Botón para ejecutar el evaluador ─────────────────────────────────────────

col1, col2 = st.columns([2, 1])
with col1:
    st.info("Ejecuta el evaluador para comparar los datos extraídos contra el golden dataset.")
with col2:
    if st.button("▶ Ejecutar evaluador", use_container_width=True):
        with st.spinner("Evaluando..."):
            result = subprocess.run(
                [sys.executable, "-m", "evaluation.evaluador_extraccion"],
                capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("Evaluación completada.")
        else:
            st.error(f"Error: {result.stderr}")

# ── Cargar resultados ──────────────────────────────────────────────────────────

if not RESULTADOS_PATH.exists():
    st.warning("Aún no hay resultados. Ejecuta el evaluador primero.")
    st.stop()

with open(RESULTADOS_PATH, encoding="utf-8") as f:
    resultados = json.load(f)

fondos_evaluados = [r for r in resultados if r["total"] > 0]

if not fondos_evaluados:
    st.warning(
        "El golden dataset tiene todos los campos en `null`. "
        "Rellena los valores reales en `evaluation/golden_dataset_extraccion.json` y vuelve a ejecutar."
    )
    st.stop()

# ── Métricas globales ─────────────────────────────────────────────────────────

total_campos  = sum(r["total"] for r in fondos_evaluados)
campos_ok     = sum(r["correctos"] for r in fondos_evaluados)
precision_global = campos_ok / total_campos if total_campos > 0 else 0.0

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Fondos evaluados",  len(fondos_evaluados))
c2.metric("Campos evaluados",  total_campos)
c3.metric("Precisión global",  f"{precision_global:.0%}")

# ── Precisión por fondo ───────────────────────────────────────────────────────

st.subheader("Precisión por fondo")

import pandas as pd

df_fondos = pd.DataFrame([
    {
        "Fondo": r["nombre_fondo"],
        "ISIN": r["isin"],
        "Correctos": r["correctos"],
        "Total": r["total"],
        "Precisión": f"{r['precision']:.0%}",
    }
    for r in fondos_evaluados
]).sort_values("Precisión", ascending=True)

st.dataframe(df_fondos, use_container_width=True, hide_index=True)

# ── Precisión por campo ───────────────────────────────────────────────────────

st.subheader("Precisión por campo")

conteo: dict[str, dict] = {}
for r in fondos_evaluados:
    for c in r["campos"]:
        campo = c["campo"]
        if campo not in conteo:
            conteo[campo] = {"ok": 0, "total": 0}
        conteo[campo]["total"] += 1
        if c["correcto"]:
            conteo[campo]["ok"] += 1

df_campos = pd.DataFrame([
    {
        "Campo": campo,
        "Correctos": v["ok"],
        "Total": v["total"],
        "Precisión": round(v["ok"] / v["total"], 2),
    }
    for campo, v in conteo.items()
]).sort_values("Precisión", ascending=True)

st.bar_chart(df_campos.set_index("Campo")["Precisión"])
st.dataframe(df_campos, use_container_width=True, hide_index=True)

# ── Detalle por fondo ─────────────────────────────────────────────────────────

st.subheader("Detalle por fondo")

for r in resultados:
    campos_con_valor = [c for c in r["campos"] if c["esperado"] is not None]
    if not campos_con_valor:
        continue

    precision = r["precision"]
    color = "🟢" if precision >= 0.8 else "🟡" if precision >= 0.5 else "🔴"

    with st.expander(f"{color} {r['nombre_fondo']} — {precision:.0%}"):
        for c in campos_con_valor:
            icono = "✅" if c["correcto"] else "❌"
            st.markdown(
                f"{icono} **{c['campo']}** — "
                f"esperado: `{c['esperado']}` | "
                f"obtenido: `{c['obtenido']}` | "
                f"_{c['motivo']}_"
            )
