# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based investment fund advisor for Spanish DFI/KID documents. Scrapes the CNMV portal, extracts structured data from PDFs using Gemini, stores results in SQLite + ChromaDB, and recommends funds via vector similarity scoring through a conversational chatbot.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY
```

## Running

```bash
# Full pipeline (scrape → extract → chat)
python main.py

# Individual components
python -m scraper.cnmv_scraper        # Download PDFs from CNMV
python -m extraction.pdf_extractor    # Extract structured data from PDFs
python -m chatbot.conversation        # Start the conversational advisor
```

## Architecture

```
CNMV Portal
    │
    ▼
scraper/cnmv_scraper.py     → data/dfi_pdfs/*.pdf
    │
    ▼
extraction/pdf_extractor.py → database/funds.db (SQLite, structured fields)
                            → ChromaDB (investment_policy embeddings)
    │
    ▼
scoring/
  ├── user_profiler.py      → float[9] user vector (normalized 0–1)
  ├── fund_vectorizer.py    → float[9] fund vector (same space)
  └── scorer.py             → final score = 0.6·cosine + 0.4·semantic
    │
    ▼
chatbot/conversation.py     → 6-question profiling → top-5 fund recommendations
    │
    ▼
evaluation/                 → RAGAS metrics
```

## Key Design Decisions

- **LLM**: Gemini 2.5 via `langchain-google-genai`, used with structured output (Pydantic models) in the extraction step.
- **Vector space**: 9 normalized dimensions shared between user profile and fund vectors — cosine similarity computed with scikit-learn.
- **Scoring formula**: `0.6 * cosine_similarity + 0.4 * semantic_similarity` (ChromaDB on `investment_policy`).
- **Language**: All chatbot dialogue and docstrings are in Spanish.
- **Rate limiting**: CNMV scraper enforces 1 request/second.
- **SQLite** stores all structured fund fields; **ChromaDB** stores only the `investment_policy` text for semantic search.

## Fund Data Model (Pydantic)

Fields extracted per fund: `fund_name`, `isin`, `manager`, `category`, `risk_level` (1–7), `investment_policy`, `min_investment`, `recommended_horizon_years`, `entry_fee`, `exit_fee`, `management_fee`, `performance_fee`, `deposit_fee`, `sector_distribution` (dict), `geographic_distribution` (dict), `management_type` (active/passive), `benchmark`, `esg` (bool), `currency_hedged` (bool), `accumulation` (bool), `asset_universe` (list), `liquidity_restrictions`, `volatility`.

## User Profile Vector Dimensions

`risk_tolerance`, `time_horizon`, `capital`, `thematic_preferences`, `liquidity_need`, `esg_sensitivity`, `active_mgmt_preference`, `geographic_preference`, `sector_preference` — all normalized to [0.0, 1.0].
