"""
api/main.py — FastAPI Application
====================================
Run with:
    uvicorn api.main:app --reload --port 8000

Interactive docs available at:
    http://localhost:8000/docs
"""
from __future__ import annotations

# Charger .env en tout premier (développement local — NFR-05)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=False)
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="YouTube Quality Analyzer API",
    description=(
        "Multi-agent LLM pipeline that predicts YouTube video quality "
        "from pre-collected comments. Exposes POST /analyze and GET /report/{video_id}."
    ),
    version="2.0.0",
)

# CORS — allow the Chrome extension (and local dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to extension origin in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}
