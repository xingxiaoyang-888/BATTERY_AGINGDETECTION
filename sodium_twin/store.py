"""Small durable local workspace; no shared default account or external service."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4
import pandas as pd

from .data import demo_frame, summarize


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "workspace.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, chemistry TEXT NOT NULL,
                    provenance TEXT NOT NULL, created_at TEXT NOT NULL, rows_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL,
                    request_json TEXT NOT NULL, result_json TEXT NOT NULL
                );
            """)
            if not db.execute("SELECT 1 FROM datasets WHERE id='demo'").fetchone():
                db.execute(
                    "INSERT INTO datasets VALUES (?,?,?,?,?,?)",
                    (
                        "demo",
                        "Sodium reference fleet",
                        "sodium-ion",
                        "synthetic",
                        self.now(),
                        demo_frame().to_json(orient="records"),
                    ),
                )

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def datasets(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [self._dataset(row) for row in rows]

    @staticmethod
    def _dataset(row):
        data = dict(row)
        frame = pd.DataFrame(json.loads(data.pop("rows_json")))
        return {**data, **summarize(frame)}

    def frame(self, dataset_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if row is None:
            raise KeyError("Dataset not found.")
        return pd.DataFrame(json.loads(row["rows_json"])), self._dataset(row)

    def import_frame(self, name, chemistry, frame):
        dataset_id = uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO datasets VALUES (?,?,?,?,?,?)",
                (
                    dataset_id,
                    name,
                    chemistry,
                    "user-upload",
                    self.now(),
                    frame.to_json(orient="records"),
                ),
            )
        return self.frame(dataset_id)[1]

    def save_run(self, kind, request, result):
        run_id, created = uuid4().hex, self.now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?)",
                (
                    run_id,
                    kind,
                    created,
                    json.dumps(request, allow_nan=False),
                    json.dumps(result, allow_nan=False),
                ),
            )
        return {"id": run_id, "created_at": created, **result}

    def runs(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,kind,created_at,request_json FROM runs "
                "ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "created_at": row["created_at"],
                "request": json.loads(row["request_json"]),
            }
            for row in rows
        ]

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("Run not found.")
        return {
            "id": row["id"],
            "kind": row["kind"],
            "created_at": row["created_at"],
            "request": json.loads(row["request_json"]),
            "result": json.loads(row["result_json"]),
        }
