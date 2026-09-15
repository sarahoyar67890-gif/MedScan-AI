"""
inference/validation.py — Input validation for uploaded images.

Every upload (Streamlit or the API) passes through here before it ever
reaches the model. This is deliberately strict and deliberately boring:
reject early, reject with a clear reason, never let a bad file reach
PyTorch or the filesystem unchecked.
"""

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Limits (kept here, not scattered across the app, so they're easy to audit)
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION_PX = 32                    # below this, Grad-CAM/ResNet features are meaningless
MAX_DIMENSION_PX = 8000                  # guards against decompression-bomb-style images
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageValidationError(ValueError):
    """Raised when an uploaded file fails validation. Message is safe to show the user."""


@dataclass
class ValidatedImage:
    image: Image.Image
    width: int
    height: int
    size_bytes: int


def _check_extension(filename: str) -> None:
    if not filename or "." not in filename:
        raise ImageValidationError("File has no extension. Please upload a .jpg or .png file.")
    ext = "." + filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext}'. Only JPG and PNG images are accepted."
        )


def validate_image_bytes(
    raw_bytes: bytes,
    filename: str = "upload",
    content_type: Optional[str] = None,
) -> ValidatedImage:
    """Validates raw file bytes and returns a decoded, RGB-converted PIL image.

    Checks (in order): size limit, extension allowlist, declared content-type
    (when provided), that the bytes actually decode as an image, and that the
    decoded dimensions are sane. Raises ImageValidationError with a
    user-safe message on any failure — never raises a raw PIL/OS exception
    up to the caller.
    """
    if not raw_bytes:
        raise ImageValidationError("The uploaded file is empty.")

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise ImageValidationError(f"File is too large. Maximum allowed size is {mb:.0f} MB.")

    _check_extension(filename)

    if content_type is not None and content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            f"Unsupported content type '{content_type}'. Only JPEG and PNG images are accepted."
        )

    # Verify first (cheap integrity check), then re-open to actually decode —
    # PIL's Image.verify() leaves the file object unusable for further reads.
    try:
        probe = Image.open(BytesIO(raw_bytes))
        probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ImageValidationError("The file couldn't be read as a valid image.") from e

    try:
        image = Image.open(BytesIO(raw_bytes))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ImageValidationError("The image file is corrupted and could not be decoded.") from e

    width, height = image.size
    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        raise ImageValidationError(
            f"Image is too small ({width}×{height}px). Minimum is "
            f"{MIN_DIMENSION_PX}×{MIN_DIMENSION_PX}px."
        )
    if width > MAX_DIMENSION_PX or height > MAX_DIMENSION_PX:
        raise ImageValidationError(
            f"Image is too large ({width}×{height}px). Maximum is "
            f"{MAX_DIMENSION_PX}×{MAX_DIMENSION_PX}px."
        )

    return ValidatedImage(image=image, width=width, height=height, size_bytes=len(raw_bytes))
