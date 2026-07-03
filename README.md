# Asesor de Fondos de Inversión con RAG

Sistema de recomendación de fondos de inversión españoles basado en RAG (Retrieval-Augmented Generation). Descarga DFIs del portal CNMV, extrae datos estructurados con Gemini, y recomienda fondos mediante similitud vectorial a través de un chatbot conversacional.

---

## Arquitectura

```
Portal CNMV
    │
    ▼
scraper/cnmv_scraper.py     →  data/dfi_pdfs/*.pdf
    │
    ▼
extraction/pdf_extractor.py →  database/funds.db     (SQLite, campos estructurados)
                            →  database/chroma/       (ChromaDB, embeddings de política de inversión)
    │
    ▼
scoring/
  ├── user_profiler.py      →  vector de usuario float[9] normalizado [0–1]
  ├── fund_vectorizer.py    →  vector de fondo float[9] en el mismo espacio
  └── scorer.py             →  score = 0.6·coseno + 0.4·semántico + 0.1·esg_boost
    │
    ▼
chatbot/conversation.py     →  6 preguntas de perfilado → top-5 fondos recomendados
    │
    ▼
evaluation/ragas_eval.py    →  métricas RAGAS (faithfulness, answer relevancy, context precision)
```

---

## Requisitos

- Python 3.11+
- Cuenta en [Google AI Studio](https://aistudio.google.com/) para obtener una `GEMINI_API_KEY`

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/johanjtg/rag-fondos.git
cd rag-fondos

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env y añadir tu GEMINI_API_KEY
```

---

## Uso

### Pipeline completo

```bash
python main.py
```

### Paso a paso

#### 1. Descargar PDFs del portal CNMV

```bash
# Descargar 50 fondos
python -m scraper.cnmv_scraper --max 50

# Descargar todos los fondos disponibles
python -m scraper.cnmv_scraper

# Reanudar una descarga interrumpida
python -m scraper.cnmv_scraper --resume
```

#### 2. Extraer datos de los PDFs

```bash
# Extracción estándar (pypdf)
python -m extraction.pdf_extractor --input data/dfi_pdfs/

# Con Docling (mejor extracción de tablas y nivel de riesgo)
python -m extraction.pdf_extractor --input data/dfi_pdfs/ --docling

# Un solo fondo
python -m extraction.pdf_extractor --input data/dfi_pdfs/ES0123456789.pdf

# Procesamiento paralelo (más rápido, consume más cuota de API)
python -m extraction.pdf_extractor --input data/dfi_pdfs/ --workers 4
```

#### 3. Iniciar el chatbot

```bash
python -m chatbot.conversation

# Opciones adicionales
python -m chatbot.conversation --top 3       # recomendar top 3 en lugar de 5
python -m chatbot.conversation --verbose     # logging detallado
```

#### 4. Inspeccionar los datos extraídos

```bash
# Resumen general de la base de datos
python scripts/inspect_data.py

# Ver todos los campos de un fondo por ISIN
python scripts/inspect_data.py --isin ES0156873004

# Buscar fondos por nombre o gestora
python scripts/inspect_data.py --search "abante"

# Fondos con campos críticos vacíos (diagnóstico de extracción)
python scripts/inspect_data.py --nulls

# Búsqueda semántica en ChromaDB
python scripts/inspect_data.py --chroma "tecnología americana"
```

#### 5. Ejecutar evaluación RAGAS

```bash
python -m evaluation.ragas_eval
```

---

## Fórmula de scoring

```
score_final =
    0.60 × similitud_coseno(vector_usuario, vector_fondo)
  + 0.40 × similitud_semántica(preferencias → ChromaDB)
  + 0.10 × esg_boost  (si sensibilidad_esg > 0.6 y fondo.esg == True)
```

**Filtros duros** aplicados antes del scoring (excluyen el fondo directamente):
- `importe_minimo_inversion > capital_disponible`
- `nivel_riesgo > nivel_riesgo_maximo_usuario`
- `horizonte_recomendado_anios > horizonte_usuario`
- Fondo con restricciones de liquidez y usuario con alta necesidad de liquidez

---

## Base de datos

Los archivos `database/funds.db` (SQLite) y `database/chroma/` (ChromaDB) se incluyen directamente en el repositorio para que la imagen Docker pueda arrancar sin necesidad de ejecutar el pipeline de extracción. Esto significa que para actualizar el catálogo de fondos hay que regenerar la base de datos localmente y volver a hacer push.

> **Escalado futuro:** en un entorno de producción, la base de datos debería almacenarse en **Azure Blob Storage** y montarse en el contenedor en tiempo de arranque. Esto permitiría actualizar el catálogo sin reconstruir la imagen Docker y soportaría múltiples instancias del servicio compartiendo los mismos datos.

---

## Reiniciar la base de datos

```bash
rm database/funds.db
rm -rf database/chroma/
```

Luego volver a ejecutar el paso de extracción.

---

## Variables de entorno

| Variable | Descripción |
|----------|-------------|
| `GEMINI_API_KEY` | API key de Google Gemini (obligatoria) |
