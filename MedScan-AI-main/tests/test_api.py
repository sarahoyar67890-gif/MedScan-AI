import io

import pytest
from fastapi.testclient import TestClient

import config


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Isolate this test run from any real checkpoint/history on disk.
    monkeypatch.setattr(config, "BEST_MODEL_PATH", tmp_path / "no_checkpoint.pt")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)

    import inference.predictor as predictor_module
    predictor_module._predictor_singleton = None

    from backend.main import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info_not_ready(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is False
    assert body["architecture"] == "ResNet18"


def test_predict_returns_503_without_checkpoint(client):
    img_bytes = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (224, 224), color=(100, 100, 100)).save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    resp = client.post(
        "/predict",
        files={"file": ("lesion.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 503


def test_predict_rejects_invalid_file(client):
    resp = client.post(
        "/predict",
        files={"file": ("notanimage.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
    )
    assert resp.status_code == 400


def test_stats_empty(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_analyses"] == 0


def test_history_empty(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.json() == []
