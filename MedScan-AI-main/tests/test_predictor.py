import config
from inference.predictor import ModelNotReadyError, Predictor
from inference.validation import validate_image_bytes


def test_predictor_reports_not_ready_without_checkpoint(monkeypatch, tmp_path):
    # Point at a checkpoint path that doesn't exist, regardless of the real project state.
    fake_path = tmp_path / "does_not_exist.pt"
    monkeypatch.setattr(config, "BEST_MODEL_PATH", fake_path)

    predictor = Predictor()
    assert predictor.is_ready is False
    status = predictor.status()
    assert status.ready is False
    assert status.error is None  # no checkpoint present is not a load error


def test_predict_raises_when_not_ready(monkeypatch, tmp_path, valid_jpeg_bytes):
    fake_path = tmp_path / "does_not_exist.pt"
    monkeypatch.setattr(config, "BEST_MODEL_PATH", fake_path)

    predictor = Predictor()
    validated = validate_image_bytes(valid_jpeg_bytes, "lesion.jpg", "image/jpeg")

    try:
        predictor.predict(validated)
        assert False, "expected ModelNotReadyError"
    except ModelNotReadyError:
        pass


def test_reload_picks_up_state_changes(monkeypatch, tmp_path):
    fake_path = tmp_path / "does_not_exist.pt"
    monkeypatch.setattr(config, "BEST_MODEL_PATH", fake_path)

    predictor = Predictor()
    assert predictor.is_ready is False
    predictor.reload()  # still no checkpoint — should not raise, should stay not-ready
    assert predictor.is_ready is False
