"""backend/schemas.py — Request/response models for the MedScan AI API."""

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class ModelInfoResponse(BaseModel):
    ready: bool
    architecture: str = "ResNet18"
    framework: str = "PyTorch"
    dataset: str = "HAM10000"
    explainability: str = "Grad-CAM"
    device: str
    checkpoint_stage: Optional[str] = None
    checkpoint_epoch: Optional[int] = None
    validation_f1: Optional[float] = None
    error: Optional[str] = None


class PredictionResponse(BaseModel):
    label: str
    is_suspicious: bool
    confidence: float
    probabilities: dict[str, float]
    inference_ms: float
    gradcam_overlay_base64: str = Field(
        ..., description="PNG-encoded Grad-CAM overlay, base64-encoded"
    )
    disclaimer: str = (
        "This is a research/educational screening signal, not a medical diagnosis. "
        "It does not replace evaluation by a qualified clinician."
    )


class ErrorResponse(BaseModel):
    detail: str


class HistoryItem(BaseModel):
    id: int
    created_at: str
    filename: str
    predicted_label: str
    is_suspicious: bool
    confidence: float
    inference_ms: Optional[float] = None


class StatsResponse(BaseModel):
    total_analyses: int
    suspicious_count: int
    benign_count: int
    avg_confidence: Optional[float] = None
    avg_inference_ms: Optional[float] = None
    last_analysis_at: Optional[str] = None
