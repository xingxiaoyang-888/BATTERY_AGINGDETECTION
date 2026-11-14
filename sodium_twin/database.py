"""Private SQLite files and bounded connections shared by both application generations."""

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3


class StorageLimitError(ValueError):
    pass


def prepare_database(path: Path):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Database symlinks are not supported.")
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)
    if os.name == "posix":
        path.chmod(0o600)


@contextmanager
def connect_database(path: Path, max_bytes=256 * 1024 * 1024):
    prepare_database(path)
    db = sqlite3.connect(path, timeout=5)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        # Only a validated integer is interpolated into a SQLite pragma.
        pages = max(16, int(max_bytes) // page_size)
        db.execute(f"PRAGMA max_page_count={pages}")
        with db:
            yield db
    except sqlite3.OperationalError as exc:
        if getattr(exc, "sqlite_errorcode", None) == getattr(
            sqlite3, "SQLITE_FULL", 13
        ) or "database or disk is full" in str(exc):
            raise StorageLimitError(
                "Workspace storage limit reached. Archive data before adding more."
            ) from exc
        raise
    finally:
        db.close()
