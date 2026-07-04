"""
Página de evaluación — Evaluador 1 (extracción), Evaluador 2 (recomendación)
y Evaluador 3 (modelos de embedding).
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH    = Path("database/funds.db")
GOLDEN_EXT = Path("evaluation/golden_dataset_extraccion.json")
GOLDEN_REC = Path("evaluation/golden_dataset_recomendacion.json")
RES_EMB    = Path("evaluation/resultados_embeddings.json")
RES_EXT    = Path("evaluation/resultados_extraccion.json")
RES_REC    = Path("evaluation/resultados_recomendacion.json")

st.title("📊 Evaluación del Sistema")

tab1, tab2, tab3 = st.tabs([
    "🔍 Evaluador 1 — Extracción",
    "🎯 Evaluador 2 — Recomendación",
    "🧠 Evaluador 3 — Modelos de Embedding",
])


# ══════════════════════════════════════════════════════════════════════════════
# EVALUADOR 1 — Calidad de extracción
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Calidad de extracción de datos")
    st.caption("Compara los campos extraídos por Gemini contra los valores reales del PDF")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("Rellena `evaluation/golden_dataset_extraccion.json` con los valores reales y ejecuta el evaluador.")
    with col2:
        if st.button("▶ Ejecutar", key="run_ext", use_container_width=True):
            with st.spinner("Evaluando extracción..."):
                result = subprocess.run(
                    [sys.executable, "-m", "evaluation.evaluador_extraccion"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Completado.")
                st.rerun()
            else:
                st.error(result.stderr[-500:])

    if not RES_EXT.exists():
        st.warning("Sin resultados aún. Ejecuta el evaluador primero.")

    else:
        with open(RES_EXT, encoding="utf-8") as f:
            res_ext = json.load(f)

        fondos_eval = [r for r in res_ext if r["total"] > 0]

        if not fondos_eval:
            st.warning("El golden dataset tiene todos los campos en `null`. Rellénalo y vuelve a ejecutar.")
        else:
            total = sum(r["total"] for r in fondos_eval)
            ok    = sum(r["correctos"] for r in fondos_eval)
            prec  = ok / total if total > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Fondos evaluados", len(fondos_eval))
            c2.metric("Campos evaluados", total)
            c3.metric("Precisión global", f"{prec:.0%}")

            st.subheader("Precisión por fondo")
            df_f = pd.DataFrame([
                {"Fondo": r["nombre_fondo"], "ISIN": r["isin"],
                 "Correctos": r["correctos"], "Total": r["total"],
                 "Precisión": r["precision"]}
                for r in fondos_eval
            ]).sort_values("Precisión")
            st.dataframe(df_f, use_container_width=True, hide_index=True)

            st.subheader("Precisión por campo")
            conteo: dict[str, dict] = {}
            for r in fondos_eval:
                for c in r["campos"]:
                    if c["esperado"] is None:
                        continue
                    campo = c["campo"]
                    if campo not in conteo:
                        conteo[campo] = {"ok": 0, "total": 0}
                    conteo[campo]["total"] += 1
                    if c["correcto"]:
                        conteo[campo]["ok"] += 1

            df_c = pd.DataFrame([
                {"Campo": k, "Precisión": round(v["ok"] / v["total"], 2),
                 "Correctos": v["ok"], "Total": v["total"]}
                for k, v in conteo.items()
            ]).sort_values("Precisión")
            st.bar_chart(df_c.set_index("Campo")["Precisión"])

            st.subheader("Detalle por fondo")
            for r in fondos_eval:
                campos = [c for c in r["campos"] if c["esperado"] is not None]
                prec_f = r["precision"]
                icono  = "🟢" if prec_f >= 0.8 else "🟡" if prec_f >= 0.5 else "🔴"
                with st.expander(f"{icono} {r['nombre_fondo']} — {prec_f:.0%}"):
                    for c in campos:
                        mark = "✅" if c["correcto"] else "❌"
                        st.markdown(
                            f"{mark} **{c['campo']}** — "
                            f"esperado: `{c['esperado']}` | "
                            f"obtenido: `{c['obtenido']}` | _{c['motivo']}_"
                        )


# ══════════════════════════════════════════════════════════════════════════════
# EVALUADOR 2 — Calidad de recomendación
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Calidad de recomendación")
    st.caption("Mide si el sistema recomienda los fondos correctos y compara configuraciones de pesos coseno/semántico")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("Compara 6 configuraciones de pesos para justificar empíricamente la ponderación 60/40.")
    with col2:
        if st.button("▶ Ejecutar", key="run_rec", use_container_width=True):
            with st.spinner("Evaluando recomendaciones..."):
                result = subprocess.run(
                    [sys.executable, "-m", "evaluation.evaluador_recomendacion"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Completado.")
                st.rerun()
            else:
                st.error(result.stderr[-500:])

    if not RES_REC.exists():
        st.warning("Sin resultados aún. Ejecuta el evaluador primero.")

    else:
        with open(RES_REC, encoding="utf-8") as f:
            res_rec = json.load(f)

        st.subheader("Comparativa de configuraciones de pesos")

        df_config = pd.DataFrame([
            {
                "Configuración": r["configuracion"],
                "Coseno": f"{r['coseno']:.0%}",
                "Semántico": f"{r['semantico']:.0%}",
                "Precision@K": r["precision_media"],
                "Hit@1": r["hit_at_1_ratio"],
                "MRR": r["mrr_medio"],
            }
            for r in res_rec
        ])

        def highlight_actual(row):
            if row["Configuración"] == "60/40 (actual)":
                return ["background-color: #d4edda"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_config.style.apply(highlight_actual, axis=1),
            use_container_width=True, hide_index=True
        )
        st.bar_chart(df_config.set_index("Configuración")[["Precision@K", "Hit@1", "MRR"]])

        st.subheader("Detalle por perfil — configuración 60/40")
        config_actual = next((r for r in res_rec if r["configuracion"] == "60/40 (actual)"), None)
        if config_actual:
            for p in config_actual["perfiles"]:
                prec  = p["precision_at_k"]
                icono = "🟢" if prec >= 0.8 else "🟡" if prec >= 0.5 else "🔴"
                with st.expander(
                    f"{icono} {p['descripcion']} — "
                    f"P@K={prec:.0%}  MRR={p['mrr']:.2f}  Hit@1={'✓' if p['hit_at_1'] else '✗'}"
                ):
                    c1, c2 = st.columns(2)
                    c1.markdown("**Esperados:**")
                    for isin in p["fondos_esperados"]:
                        c1.markdown(f"- `{isin}`")
                    c2.markdown("**Obtenidos (top-5):**")
                    for i, isin in enumerate(p["fondos_obtenidos"], 1):
                        marca = "✅" if isin in p["fondos_esperados"] else "　"
                        c2.markdown(f"{i}. {marca} `{isin}`")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUADOR 3 — Comparativa de modelos de embedding
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Comparativa de modelos de embedding")
    st.caption(
        "Compara 3 modelos de embedding sobre el mismo golden dataset "
        "para justificar la elección de `paraphrase-multilingual-MiniLM-L12-v2`"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("Re-indexa ChromaDB con cada modelo y mide Precision@K, Hit@1 y MRR. Tarda ~1-2 minutos.")
    with col2:
        if st.button("▶ Ejecutar", key="run_emb", use_container_width=True):
            with st.spinner("Evaluando modelos de embedding... (puede tardar ~2 min)"):
                result = subprocess.run(
                    [sys.executable, "-m", "evaluation.evaluador_embeddings"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Completado.")
                st.rerun()
            else:
                st.error(result.stderr[-500:])

    if not RES_EMB.exists():
        st.warning("Sin resultados aún. Ejecuta el evaluador primero.")
    else:
        with open(RES_EMB, encoding="utf-8") as f:
            res_emb = json.load(f)

        st.subheader("Resultados por modelo")

        df_emb = pd.DataFrame([
            {
                "Modelo": r["nombre"],
                "Descripción": r["descripcion"],
                "Precision@K": r["precision_media"],
                "Hit@1": r["hit_at_1_ratio"],
                "MRR": r["mrr_medio"],
                "Tiempo indexación (s)": r["tiempo_s"],
            }
            for r in res_emb
        ])

        def highlight_actual_emb(row):
            if "actual" in row["Modelo"]:
                return ["background-color: #d4edda"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_emb.style.apply(highlight_actual_emb, axis=1),
            use_container_width=True, hide_index=True
        )

        st.bar_chart(df_emb.set_index("Modelo")[["Precision@K", "Hit@1", "MRR"]])

        st.subheader("Detalle por perfil — modelo actual")
        modelo_actual = next((r for r in res_emb if "actual" in r["nombre"]), None)
        if modelo_actual:
            for p in modelo_actual["perfiles"]:
                prec  = p["precision_at_k"]
                icono = "🟢" if prec >= 0.8 else "🟡" if prec >= 0.5 else "🔴"
                with st.expander(
                    f"{icono} {p['descripcion']} — "
                    f"P@K={prec:.0%}  MRR={p['mrr']:.2f}  Hit@1={'✓' if p['hit_at_1'] else '✗'}"
                ):
                    c1, c2 = st.columns(2)
                    c1.markdown("**Esperados:**")
                    for isin in p["esperados"]:
                        c1.markdown(f"- `{isin}`")
                    c2.markdown("**Obtenidos (top-5):**")
                    for i, isin in enumerate(p["obtenidos"], 1):
                        marca = "✅" if isin in p["esperados"] else "　"
                        c2.markdown(f"{i}. {marca} `{isin}`")
