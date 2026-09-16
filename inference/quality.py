"""
inference/quality.py — Pre-inference image quality assessment.

Runs cheap, deterministic checks on an uploaded image BEFORE it reaches the
model: resolution, blur (via Laplacian variance, computed with plain numpy —
no OpenCV dependency needed), brightness, and contrast. This exists so the
app can warn the user about a genuinely unreliable photo instead of quietly
running the model on it and returning a confident-looking number.

These thresholds are heuristic, not clinically validated — they catch
obviously bad photos (very blurry, near-black, washed out, tiny) and are
deliberately conservative about calling something "Poor" so a normal decent
phone photo isn't blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

MIN_DIMENSION = 200          # below this, a lesion photo is too small to trust
BLUR_VARIANCE_THRESHOLD = 80.0    # Laplacian variance below this ≈ visibly blurry
BRIGHTNESS_MIN = 35           # 0-255 mean luminance
BRIGHTNESS_MAX = 220
CONTRAST_MIN = 12.0           # 0-255 std deviation of luminance


@dataclass
class QualityReport:
    overall: str                 # "Good" | "Acceptable" | "Poor"
    width: int
    height: int
    resolution_ok: bool
    blur_score: float
    blur_ok: bool
    brightness: float
    brightness_ok: bool
    contrast: float
    contrast_ok: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def should_warn(self) -> bool:
        return self.overall != "Good"

    @property
    def should_block(self) -> bool:
        """Poor quality doesn't hard-block analysis — the user can still
        proceed — but callers can use this to decide whether to require an
        explicit 'analyze anyway' confirmation."""
        return self.overall == "Poor"


def _laplacian_variance(gray: np.ndarray) -> float:
    """Approximates OpenCV's cv2.Laplacian(...).var() blur metric using a
    vectorized 4-neighbor Laplacian kernel — avoids adding an OpenCV
    dependency for one metric. Lower variance = less edge detail = blurrier."""
    gray = gray.astype(np.float32)
    laplacian = (
        -4 * gray
        + np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
    )
    # Edge pixels wrap around with np.roll and produce artificial edges;
    # trim a 1px border before computing variance to avoid skewing the score.
    trimmed = laplacian[1:-1, 1:-1]
    return float(trimmed.var())


def assess_quality(image: Image.Image) -> QualityReport:
    """Runs all quality checks on a PIL image and returns a structured report."""
    width, height = image.size
    resolution_ok = width >= MIN_DIMENSION and height >= MIN_DIMENSION

    gray = np.array(image.convert("L"))
    blur_score = _laplacian_variance(gray)
    blur_ok = blur_score >= BLUR_VARIANCE_THRESHOLD

    brightness = float(gray.mean())
    brightness_ok = BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX

    contrast = float(gray.std())
    contrast_ok = contrast >= CONTRAST_MIN

    warnings: list[str] = []
    if not resolution_ok:
        warnings.append(
            f"Resolution is low ({width}×{height}px, minimum recommended is "
            f"{MIN_DIMENSION}×{MIN_DIMENSION}px). Results may be less reliable."
        )
    if not blur_ok:
        warnings.append("The image appears blurry. A sharper, in-focus photo gives a more reliable result.")
    if brightness < BRIGHTNESS_MIN:
        warnings.append("The image appears too dark. Try retaking it in brighter, even lighting.")
    elif brightness > BRIGHTNESS_MAX:
        warnings.append("The image appears overexposed or washed out. Try reducing glare or direct flash.")
    if not contrast_ok:
        warnings.append("The image has low contrast, which can make the lesion harder to distinguish from surrounding skin.")

    failed_checks = sum([not resolution_ok, not blur_ok, not brightness_ok, not contrast_ok])
    if failed_checks == 0:
        overall = "Good"
    elif failed_checks == 1:
        overall = "Acceptable"
    else:
        overall = "Poor"

    return QualityReport(
        overall=overall,
        width=width, height=height, resolution_ok=resolution_ok,
        blur_score=round(blur_score, 1), blur_ok=blur_ok,
        brightness=round(brightness, 1), brightness_ok=brightness_ok,
        contrast=round(contrast, 1), contrast_ok=contrast_ok,
        warnings=warnings,
    )
