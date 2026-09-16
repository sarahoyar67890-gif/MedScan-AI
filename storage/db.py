"""
storage/db.py — Lightweight SQLite store for analysis history.

Deliberately not a "real" database — this is a portfolio app meant to run
locally or in a single small container, and SQLite is the honest choice at
that scale (zero setup, no extra service, no extra failure mode).

v2 change: now also saves the uploaded image itself to disk
(outputs/scan_images/) and stores its path, so History/Progress can show
real thumbnails and a genuine side-by-side "compare two scans" view instead
of metadata-only rows. Existing databases are migrated in place (a new
nullable image_path column is added if missing) — no data is lost, and rows
recorded before this change simply have image_path = None.

Every write is wrapped so a storage failure never breaks the analysis flow
itself — history is a nice-to-have, not a dependency of the core feature.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import config

log = logging.getLogger(__name__)

DB_PATH: Path = config.OUTPUTS_DIR / "medscan_history.db"
IMAGES_DIR: Path = config.OUTPUTS_DIR / "scan_images"

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    image_width INTEGER,
    image_height INTEGER,
    predicted_label TEXT NOT NULL,
    is_suspicious INTEGER NOT NULL,
    confidence REAL NOT NULL,
    prob_benign REAL NOT NULL,
    prob_suspicious REAL NOT NULL,
    inference_ms REAL,
    model_stage TEXT,
    model_epoch INTEGER,
    source TEXT NOT NULL DEFAULT 'streamlit'
);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);
"""


@dataclass
class AnalysisRecord:
    id: int
    created_at: str
    filename: str
    image_width: Optional[int]
    image_height: Optional[int]
    predicted_label: str
    is_suspicious: bool
    confidence: float
    prob_benign: float
    prob_suspicious: float
    inference_ms: Optional[float]
    model_stage: Optional[str]
    model_epoch: Optional[int]
    source: str
    image_path: Optional[str] = None


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate_add_image_path(conn: sqlite3.Connection) -> None:
    """Adds the image_path column to a pre-v2 database in place. Safe to
    call on every startup — it's a no-op once the column already exists."""
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()}
    if "image_path" not in existing_cols:
        conn.execute("ALTER TABLE analyses ADD COLUMN image_path TEXT")
        log.info("Migrated analyses table: added image_path column")


def init_db() -> None:
    try:
        with _connect() as conn:
            conn.executescript(SCHEMA)
            _migrate_add_image_path(conn)
    except sqlite3.Error:
        log.exception("Failed to initialize history database at %s", DB_PATH)


def _save_image(image_bytes: bytes, original_filename: str) -> Optional[str]:
    """Saves the uploaded image under outputs/scan_images/ with a random
    filename (so uploads with the same name never collide) and returns its
    path, or None if saving failed — a failure here never blocks recording
    the analysis itself."""
    try:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(original_filename).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        stored_path = IMAGES_DIR / f"{uuid.uuid4().hex}{ext}"
        stored_path.write_bytes(image_bytes)
        return str(stored_path)
    except OSError:
        log.exception("Failed to save scan image for %s", original_filename)
        return None


def record_analysis(
    filename: str,
    predicted_label: str,
    is_suspicious: bool,
    confidence: float,
    prob_benign: float,
    prob_suspicious: float,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    inference_ms: Optional[float] = None,
    model_stage: Optional[str] = None,
    model_epoch: Optional[int] = None,
    source: str = "streamlit",
    image_bytes: Optional[bytes] = None,
) -> Optional[int]:
    """Best-effort write. Returns the new row id, or None if the write failed
    (failure is logged, never raised — history must never break analysis).
    If image_bytes is provided, the image is saved to disk and its path is
    stored alongside the record, enabling real thumbnails and scan comparison."""
    image_path = _save_image(image_bytes, filename) if image_bytes else None
    try:
        with _connect() as conn:
            cur = conn.execute(
                """INSERT INTO analyses
                   (created_at, filename, image_width, image_height, predicted_label,
                    is_suspicious, confidence, prob_benign, prob_suspicious,
                    inference_ms, model_stage, model_epoch, source, image_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    filename, image_width, image_height, predicted_label,
                    int(is_suspicious), confidence, prob_benign, prob_suspicious,
                    inference_ms, model_stage, model_epoch, source, image_path,
                ),
            )
            return cur.lastrowid
    except sqlite3.Error:
        log.exception("Failed to record analysis for %s", filename)
        return None


def get_recent(limit: int = 20) -> list[AnalysisRecord]:
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [AnalysisRecord(**dict(r)) for r in rows]
    except sqlite3.Error:
        log.exception("Failed to read recent analyses")
        return []


def get_by_id(record_id: int) -> Optional[AnalysisRecord]:
    """Used by the Progress page's scan-comparison view to fetch two
    specific records by id."""
    try:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE id = ?", (record_id,)).fetchone()
            return AnalysisRecord(**dict(row)) if row else None
    except sqlite3.Error:
        log.exception("Failed to fetch analysis id=%s", record_id)
        return None


def get_stats() -> dict:
    """Aggregate stats for the dashboard. Returns zeros on any failure rather
    than raising, since the dashboard should still render."""
    empty = {
        "total_analyses": 0, "suspicious_count": 0, "benign_count": 0,
        "avg_confidence": None, "avg_inference_ms": None, "last_analysis_at": None,
    }
    try:
        with _connect() as conn:
            row = conn.execute(
                """SELECT
                     COUNT(*) AS total,
                     SUM(is_suspicious) AS suspicious,
                     AVG(confidence) AS avg_conf,
                     AVG(inference_ms) AS avg_ms,
                     MAX(created_at) AS last_at
                   FROM analyses"""
            ).fetchone()
            total = row["total"] or 0
            suspicious = row["suspicious"] or 0
            return {
                "total_analyses": total,
                "suspicious_count": suspicious,
                "benign_count": total - suspicious,
                "avg_confidence": row["avg_conf"],
                "avg_inference_ms": row["avg_ms"],
                "last_analysis_at": row["last_at"],
            }
    except sqlite3.Error:
        log.exception("Failed to compute analysis stats")
        return empty


def clear_history() -> bool:
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM analyses")
        return True
    except sqlite3.Error:
        log.exception("Failed to clear history")
        return False
