"""SQLite persistence for scan jobs, per-model results, pretrained models, and cached predictions."""
import sqlite3
import uuid
import json
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: str):
    """Create tables if they don't exist. Enable WAL mode for concurrent access."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scan_jobs (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            data_source TEXT NOT NULL,
            trading_style TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            total_models INTEGER NOT NULL DEFAULT 0,
            completed_models INTEGER NOT NULL DEFAULT 0,
            current_model TEXT,
            best_model TEXT,
            best_model_reason TEXT,
            prediction_json TEXT,
            error_msg TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scan_model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_job_id TEXT NOT NULL REFERENCES scan_jobs(id),
            model_class TEXT NOT NULL,
            status TEXT NOT NULL,
            error_msg TEXT,
            result_json TEXT,
            recorder_id TEXT,
            duration_s REAL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pretrained_models (
            id TEXT PRIMARY KEY,
            model_class TEXT NOT NULL,
            feature_set TEXT NOT NULL,
            label TEXT NOT NULL,
            training_pool TEXT NOT NULL,
            trading_style TEXT NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            experiment_name TEXT NOT NULL,
            performance_json TEXT,
            status TEXT NOT NULL DEFAULT 'training',
            error_msg TEXT,
            duration_s REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cached_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            pretrained_model_id TEXT NOT NULL REFERENCES pretrained_models(id),
            model_class TEXT NOT NULL,
            trading_style TEXT NOT NULL,
            prediction_json TEXT NOT NULL,
            alpha_score_json TEXT,
            cached_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_pretrained_lookup
            ON pretrained_models(model_class, feature_set, trading_style, status);

        CREATE INDEX IF NOT EXISTS idx_cached_ticker
            ON cached_predictions(ticker, trading_style);
    """)
    conn.close()


def create_scan_job(db_path: str, ticker: str, data_source: str, trading_style: str) -> str:
    """Create a new scan job. Returns the job ID (UUID)."""
    job_id = str(uuid.uuid4())
    now = _now()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO scan_jobs (id, ticker, data_source, trading_style, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
        (job_id, ticker, data_source, trading_style, now, now),
    )
    conn.commit()
    conn.close()
    return job_id


def get_scan_job(db_path: str, job_id: str) -> dict | None:
    """Get a scan job by ID. Returns None if not found."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM scan_jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def update_scan_job(db_path: str, job_id: str, **fields):
    """Update arbitrary fields on a scan job."""
    if not fields:
        return
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    conn = sqlite3.connect(db_path)
    conn.execute(f"UPDATE scan_jobs SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()


def list_scan_jobs(db_path: str) -> list[dict]:
    """List all scan jobs, ordered by created_at DESC."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM scan_jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_model_result(
    db_path: str,
    scan_job_id: str,
    model_class: str,
    status: str,
    result_json: str | None = None,
    error_msg: str | None = None,
    recorder_id: str | None = None,
    duration_s: float | None = None,
):
    """Save a single model's result within a scan job."""
    now = _now()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO scan_model_results
           (scan_job_id, model_class, status, error_msg, result_json, recorder_id, duration_s, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (scan_job_id, model_class, status, error_msg, result_json, recorder_id, duration_s, now),
    )
    conn.commit()
    conn.close()


def get_model_results(db_path: str, scan_job_id: str) -> list[dict]:
    """Get all model results for a scan job, ordered by creation time."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM scan_model_results WHERE scan_job_id = ? ORDER BY id",
        (scan_job_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_completed_scan(db_path: str, ticker: str) -> dict | None:
    """Get the most recent completed scan for a ticker. Returns None if none found."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT * FROM scan_jobs
           WHERE ticker = ? AND status = 'completed'
           ORDER BY created_at DESC LIMIT 1""",
        (ticker.upper(),),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


# =============================================
# Pretrained model registry
# =============================================

def save_pretrained_model(
    db_path: str,
    model_id: str,
    model_class: str,
    feature_set: str,
    label: str,
    training_pool: str,
    trading_style: str,
    train_start: str,
    train_end: str,
    experiment_name: str,
    performance_json: str | None = None,
    status: str = "completed",
    error_msg: str | None = None,
    duration_s: float | None = None,
) -> str:
    """Save a pretrained model to the registry. Returns model_id."""
    now = _now()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO pretrained_models
           (id, model_class, feature_set, label, training_pool, trading_style,
            train_start, train_end, experiment_name, performance_json,
            status, error_msg, duration_s, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (model_id, model_class, feature_set, label, training_pool, trading_style,
         train_start, train_end, experiment_name, performance_json,
         status, error_msg, duration_s, now, now),
    )
    conn.commit()
    conn.close()
    return model_id


def get_pretrained_model(db_path: str, model_id: str) -> dict | None:
    """Get a pretrained model by ID."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM pretrained_models WHERE id = ?", (model_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_pretrained_model(db_path: str, model_id: str, **fields):
    """Update fields on a pretrained model."""
    if not fields:
        return
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [model_id]
    conn = sqlite3.connect(db_path)
    conn.execute(f"UPDATE pretrained_models SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()


def find_pretrained_model(
    db_path: str,
    model_class: str,
    feature_set: str,
    trading_style: str,
) -> dict | None:
    """Find a completed pretrained model matching the given config.
    Returns the most recently created match, or None."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT * FROM pretrained_models
           WHERE model_class = ? AND feature_set = ? AND trading_style = ?
                 AND status = 'completed'
           ORDER BY created_at DESC LIMIT 1""",
        (model_class, feature_set, trading_style),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_pretrained_models(db_path: str, status: str | None = None) -> list[dict]:
    """List pretrained models, optionally filtered by status."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if status:
        rows = conn.execute(
            "SELECT * FROM pretrained_models WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM pretrained_models ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_pretrained_models_by_style(db_path: str, trading_style: str):
    """Delete all pretrained models for a trading_style (used before re-training)."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "DELETE FROM pretrained_models WHERE trading_style = ?", (trading_style,)
    )
    conn.commit()
    conn.close()


# =============================================
# Cached predictions
# =============================================

def save_cached_prediction(
    db_path: str,
    ticker: str,
    pretrained_model_id: str,
    model_class: str,
    trading_style: str,
    prediction_json: str,
    alpha_score_json: str | None = None,
) -> int:
    """Save a cached prediction. Returns the row id."""
    now = _now()
    conn = sqlite3.connect(db_path)
    # Remove old prediction for same ticker + trading_style
    conn.execute(
        "DELETE FROM cached_predictions WHERE ticker = ? AND trading_style = ?",
        (ticker.upper(), trading_style),
    )
    cur = conn.execute(
        """INSERT INTO cached_predictions
           (ticker, pretrained_model_id, model_class, trading_style,
            prediction_json, alpha_score_json, cached_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ticker.upper(), pretrained_model_id, model_class, trading_style,
         prediction_json, alpha_score_json, now),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_cached_prediction(db_path: str, ticker: str, trading_style: str) -> dict | None:
    """Get cached prediction for a ticker + trading_style. Returns None if not found."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT * FROM cached_predictions
           WHERE ticker = ? AND trading_style = ?
           ORDER BY cached_at DESC LIMIT 1""",
        (ticker.upper(), trading_style),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_cached_predictions(db_path: str, trading_style: str | None = None) -> list[dict]:
    """List cached predictions, optionally filtered by trading_style."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if trading_style:
        rows = conn.execute(
            "SELECT * FROM cached_predictions WHERE trading_style = ? ORDER BY cached_at DESC",
            (trading_style,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM cached_predictions ORDER BY cached_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_cached_predictions(db_path: str, trading_style: str | None = None):
    """Clear cached predictions, optionally filtered by trading_style."""
    conn = sqlite3.connect(db_path)
    if trading_style:
        conn.execute(
            "DELETE FROM cached_predictions WHERE trading_style = ?", (trading_style,)
        )
    else:
        conn.execute("DELETE FROM cached_predictions")
    conn.commit()
    conn.close()
