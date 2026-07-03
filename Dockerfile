# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Variables de entorno ───────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Dependencias del sistema ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo ──────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias Python ────────────────────────────────────────────────────────
# Copiamos solo requirements primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalamos sin docling (pesado y no necesario en producción)
RUN pip install --no-cache-dir \
    streamlit>=1.35.0 \
    && grep -v "^docling" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# ── Código fuente ──────────────────────────────────────────────────────────────
COPY . .

# ── Puerto expuesto ────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Comando de arranque ────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py"]
