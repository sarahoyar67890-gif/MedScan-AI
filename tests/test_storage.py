import pytest

import storage.db as db


def _use_temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_history.db")
    db.init_db()


def test_record_and_read_analysis(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)

    row_id = db.record_analysis(
        filename="lesion.jpg",
        predicted_label="benign-pattern",
        is_suspicious=False,
        confidence=0.91,
        prob_benign=0.91,
        prob_suspicious=0.09,
        image_width=224,
        image_height=224,
        inference_ms=42.0,
    )
    assert row_id is not None

    recent = db.get_recent(limit=10)
    assert len(recent) == 1
    assert recent[0].filename == "lesion.jpg"
    assert recent[0].is_suspicious is False


def test_stats_aggregate_correctly(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)

    db.record_analysis("a.jpg", "benign-pattern", False, 0.8, 0.8, 0.2, inference_ms=10.0)
    db.record_analysis("b.jpg", "suspicious-pattern", True, 0.7, 0.3, 0.7, inference_ms=20.0)

    stats = db.get_stats()
    assert stats["total_analyses"] == 2
    assert stats["suspicious_count"] == 1
    assert stats["benign_count"] == 1
    assert stats["avg_confidence"] == pytest.approx((0.8 + 0.7) / 2)


def test_clear_history(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    db.record_analysis("a.jpg", "benign-pattern", False, 0.8, 0.8, 0.2)
    assert db.get_stats()["total_analyses"] == 1

    assert db.clear_history() is True
    assert db.get_stats()["total_analyses"] == 0
