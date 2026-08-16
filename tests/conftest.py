import io
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_image_bytes(size=(224, 224), color=(120, 80, 60), fmt="JPEG") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    return _make_image_bytes(size=(224, 224), fmt="JPEG")


@pytest.fixture
def valid_png_bytes() -> bytes:
    return _make_image_bytes(size=(300, 300), fmt="PNG")


@pytest.fixture
def tiny_image_bytes() -> bytes:
    return _make_image_bytes(size=(8, 8), fmt="JPEG")


@pytest.fixture
def corrupted_bytes() -> bytes:
    return b"this is not an image file, just plain text pretending to be one"
