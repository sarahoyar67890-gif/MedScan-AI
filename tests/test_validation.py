import pytest

from inference.validation import (
    MAX_FILE_SIZE_BYTES,
    ImageValidationError,
    validate_image_bytes,
)


def test_valid_jpeg_passes(valid_jpeg_bytes):
    result = validate_image_bytes(valid_jpeg_bytes, "lesion.jpg", "image/jpeg")
    assert result.width == 224
    assert result.height == 224
    assert result.image.mode == "RGB"


def test_valid_png_passes(valid_png_bytes):
    result = validate_image_bytes(valid_png_bytes, "lesion.png", "image/png")
    assert result.width == 300


def test_empty_file_rejected():
    with pytest.raises(ImageValidationError):
        validate_image_bytes(b"", "empty.jpg", "image/jpeg")


def test_corrupted_file_rejected(corrupted_bytes):
    with pytest.raises(ImageValidationError):
        validate_image_bytes(corrupted_bytes, "fake.jpg", "image/jpeg")


def test_wrong_extension_rejected(valid_jpeg_bytes):
    with pytest.raises(ImageValidationError):
        validate_image_bytes(valid_jpeg_bytes, "lesion.exe", "image/jpeg")


def test_missing_extension_rejected(valid_jpeg_bytes):
    with pytest.raises(ImageValidationError):
        validate_image_bytes(valid_jpeg_bytes, "lesion", "image/jpeg")


def test_bad_content_type_rejected(valid_jpeg_bytes):
    with pytest.raises(ImageValidationError):
        validate_image_bytes(valid_jpeg_bytes, "lesion.jpg", "application/pdf")


def test_too_small_image_rejected(tiny_image_bytes):
    with pytest.raises(ImageValidationError):
        validate_image_bytes(tiny_image_bytes, "tiny.jpg", "image/jpeg")


def test_oversized_file_rejected():
    oversized = b"\x00" * (MAX_FILE_SIZE_BYTES + 1)
    with pytest.raises(ImageValidationError):
        validate_image_bytes(oversized, "big.jpg", "image/jpeg")
