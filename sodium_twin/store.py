"""Small durable local workspace; no shared default account or external service."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
import pandas as pd

from .data import demo_frame, summarize
from .database import connect_database, StorageLimitError


class Store:
    def __init__(
        self, directory: Path, *, max_datasets=50, max_runs=1000, max_bytes=256 * 1024 * 1024
    ):
        self.max_datasets, self.max_runs, self.max_bytes = max_datasets, max_runs, max_bytes
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

    def connect(self):
        return connect_database(self.path, self.max_bytes)

    def datasets(self):
        with self.connect() as db:
            return [
                self._dataset(row)
                for row in db.execute("SELECT * FROM datasets ORDER BY created_at DESC")
            ]

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
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM datasets").fetchone()[0] >= self.max_datasets:
                raise StorageLimitError(
                    "Dataset limit reached. Archive workspace data before importing more."
                )
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
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM runs").fetchone()[0] >= self.max_runs:
                raise StorageLimitError(
                    "Run limit reached. Archive workspace data before creating more runs."
                )
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
