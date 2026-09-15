"""
inference/predictor.py — Single shared inference service.

Both the Streamlit app and the FastAPI backend call into this module rather
than duplicating model-loading/prediction logic. That way there's exactly
one place that knows how to load the checkpoint, run preprocessing, and
produce a prediction + Grad-CAM overlay — no risk of the two front-ends
drifting out of sync.

The model is loaded lazily and cached as a process-wide singleton
(`get_predictor()`), so repeated calls don't reload weights from disk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

import torch
from PIL import Image

import config
from explainability.gradcam import GradCAM, overlay_heatmap_on_image
from inference.validation import ValidatedImage
from models.resnet_model import build_model, load_checkpoint
from utils.common import get_eval_transforms

log = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when a prediction is requested but no trained checkpoint exists yet."""


class InferenceError(RuntimeError):
    """Raised when a prediction fails for reasons other than a missing checkpoint
    (corrupt checkpoint, unexpected model output, CUDA error, etc). The message
    is safe to show to a user; details are logged separately."""


@dataclass
class PredictionResult:
    predicted_class: int
    label: str
    confidence: float
    probabilities: dict          # {"benign-pattern": 0.87, "suspicious-pattern": 0.13}
    is_suspicious: bool
    original_image: Image.Image
    gradcam_overlay: Image.Image
    inference_ms: float
    model_stage: Optional[str] = None
    model_epoch: Optional[int] = None


@dataclass
class ModelStatus:
    ready: bool
    checkpoint_path: str
    device: str
    val_f1: Optional[float] = None
    stage: Optional[str] = None
    epoch: Optional[int] = None
    error: Optional[str] = None


class Predictor:
    """Wraps model + device + checkpoint metadata. Thread-safe lazy singleton
    via get_predictor(). Reload with reload() if a new checkpoint is trained
    while the process is running (used by the app's "refresh model" action).
    """

    def __init__(self):
        self._lock = Lock()
        self.device = config.get_device()
        self.model: Optional[torch.nn.Module] = None
        self.checkpoint: Optional[dict] = None
        self.load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        with self._lock:
            self.model = None
            self.checkpoint = None
            self.load_error = None

            if not config.BEST_MODEL_PATH.exists():
                log.info("No checkpoint at %s — inference disabled until a model is trained.",
                          config.BEST_MODEL_PATH)
                return

            try:
                model = build_model(pretrained=False)
                checkpoint = load_checkpoint(model, config.BEST_MODEL_PATH, self.device)
            except Exception as e:  # noqa: BLE001 — deliberately broad: any load failure must degrade gracefully
                log.exception("Failed to load checkpoint at %s", config.BEST_MODEL_PATH)
                self.load_error = str(e)
                return

            self.model = model
            self.checkpoint = checkpoint
            log.info(
                "Loaded checkpoint (stage=%s, epoch=%s, val_f1=%s) on device=%s",
                checkpoint.get("stage"), checkpoint.get("epoch"),
                checkpoint.get("val_f1"), self.device,
            )

    def reload(self) -> None:
        """Re-check disk for a checkpoint (e.g. after training finishes)."""
        self._load()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def status(self) -> ModelStatus:
        return ModelStatus(
            ready=self.is_ready,
            checkpoint_path=str(config.BEST_MODEL_PATH),
            device=str(self.device),
            val_f1=(self.checkpoint or {}).get("val_f1"),
            stage=(self.checkpoint or {}).get("stage"),
            epoch=(self.checkpoint or {}).get("epoch"),
            error=self.load_error,
        )

    def predict(self, validated: ValidatedImage) -> PredictionResult:
        if not self.is_ready:
            raise ModelNotReadyError(
                "No trained model checkpoint is available yet. Train the model "
                "first (see README) — the app runs in design-review mode until then."
            )

        image = validated.image
        start = time.perf_counter()
        try:
            transform = get_eval_transforms()
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            gradcam = GradCAM(self.model, target_layer_name=config.GRADCAM_TARGET_LAYER)
            heatmap, predicted_class, confidence = gradcam.generate(input_tensor)

            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

            overlay = overlay_heatmap_on_image(image, heatmap)
        except Exception as e:  # noqa: BLE001
            log.exception("Inference failed")
            raise InferenceError(
                "Something went wrong while analyzing this image. This has been logged."
            ) from e

        elapsed_ms = (time.perf_counter() - start) * 1000
        label = config.CLASS_NAMES[predicted_class]
        is_suspicious = predicted_class == config.LABEL_SUSPICIOUS

        return PredictionResult(
            predicted_class=predicted_class,
            label=label,
            confidence=confidence,
            probabilities={
                config.CLASS_NAMES[0]: float(probs[0]),
                config.CLASS_NAMES[1]: float(probs[1]),
            },
            is_suspicious=is_suspicious,
            original_image=image,
            gradcam_overlay=overlay,
            inference_ms=elapsed_ms,
            model_stage=(self.checkpoint or {}).get("stage"),
            model_epoch=(self.checkpoint or {}).get("epoch"),
        )


_predictor_singleton: Optional[Predictor] = None
_singleton_lock = Lock()


def get_predictor() -> Predictor:
    """Process-wide singleton so the checkpoint is loaded once, not per-request."""
    global _predictor_singleton
    if _predictor_singleton is None:
        with _singleton_lock:
            if _predictor_singleton is None:
                _predictor_singleton = Predictor()
    return _predictor_singleton
