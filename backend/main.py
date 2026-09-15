"""
backend/main.py — MedScan AI REST API.

A separate, deployable inference service on top of the same predictor used
by the Streamlit app. This is what you'd point a mobile app, a third-party
integration, or a future non-Streamlit frontend at.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000

Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from backend.schemas import (
    ErrorResponse, HealthResponse, HistoryItem, ModelInfoResponse,
    PredictionResponse, StatsResponse,
)
from inference.predictor import (
    InferenceError, ModelNotReadyError, get_predictor,
)
from inference.validation import ImageValidationError, validate_image_bytes
from storage import db

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="MedScan AI API",
    description=(
        "AI-assisted skin lesion screening — research/educational tool, "
        "not a medical device. See /health and /model/info before use."
    ),
    version="2.0.0",
)

# CORS: restrictive by default, configurable via env for real deployments.
_allowed_origins = os.environ.get("MEDSCAN_CORS_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    # Touch the predictor once at startup so the first real request isn't
    # the one paying for checkpoint loading.
    get_predictor()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info() -> ModelInfoResponse:
    status = get_predictor().status()
    return ModelInfoResponse(
        ready=status.ready,
        device=status.device,
        checkpoint_stage=status.stage,
        checkpoint_epoch=status.epoch,
        validation_f1=status.val_f1,
        error=status.error,
    )


@app.post(
    "/model/reload", response_model=ModelInfoResponse, tags=["System"],
    summary="Re-check disk for a newly trained checkpoint without restarting the service",
)
def model_reload() -> ModelInfoResponse:
    predictor = get_predictor()
    predictor.reload()
    status = predictor.status()
    return ModelInfoResponse(
        ready=status.ready, device=status.device, checkpoint_stage=status.stage,
        checkpoint_epoch=status.epoch, validation_f1=status.val_f1, error=status.error,
    )


@app.get("/stats", response_model=StatsResponse, tags=["History"])
def stats() -> StatsResponse:
    return StatsResponse(**db.get_stats())


@app.get("/history", response_model=list[HistoryItem], tags=["History"])
def history(limit: int = 20) -> list[HistoryItem]:
    limit = max(1, min(limit, 100))
    records = db.get_recent(limit=limit)
    return [
        HistoryItem(
            id=r.id, created_at=r.created_at, filename=r.filename,
            predicted_label=r.predicted_label, is_suspicious=bool(r.is_suspicious),
            confidence=r.confidence, inference_ms=r.inference_ms,
        )
        for r in records
    ]


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Inference"],
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    raw_bytes = await file.read()

    try:
        validated = validate_image_bytes(
            raw_bytes, filename=file.filename or "upload", content_type=file.content_type
        )
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    predictor = get_predictor()
    try:
        result = predictor.predict(validated)
    except ModelNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except InferenceError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    buf = BytesIO()
    result.gradcam_overlay.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    db.record_analysis(
        filename=file.filename or "upload",
        predicted_label=result.label,
        is_suspicious=result.is_suspicious,
        confidence=result.confidence,
        prob_benign=result.probabilities[config.CLASS_NAMES[0]],
        prob_suspicious=result.probabilities[config.CLASS_NAMES[1]],
        image_width=validated.width,
        image_height=validated.height,
        inference_ms=result.inference_ms,
        model_stage=result.model_stage,
        model_epoch=result.model_epoch,
        source="api",
    )

    return PredictionResponse(
        label=result.label,
        is_suspicious=result.is_suspicious,
        confidence=result.confidence,
        probabilities=result.probabilities,
        inference_ms=result.inference_ms,
        gradcam_overlay_base64=overlay_b64,
    )
