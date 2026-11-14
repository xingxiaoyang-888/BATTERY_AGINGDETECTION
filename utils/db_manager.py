"""Legacy local accounts and history with scoped connections and explicit provisioning."""

from contextlib import contextmanager
from datetime import datetime
import json
import os
import sqlite3
import bcrypt

DB_PATH = os.getenv(
    "SODIUM_LEGACY_DB", os.path.join(os.path.dirname(os.path.dirname(__file__)), "battery_app.db")
)


@contextmanager
def _connection():
    connection = sqlite3.connect(DB_PATH, timeout=15)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _validate_credentials(username, password):
    if not 3 <= len(username.strip()) <= 80:
        raise ValueError("Username must contain 3–80 characters.")
    if not 12 <= len(password.encode("utf-8")) <= 72:
        raise ValueError("Password must contain 12–72 UTF-8 bytes.")


def init_db():
    """Create tables; provision an administrator only through explicit environment values."""
    password = os.getenv("SODIUM_ADMIN_PASSWORD")
    username = os.getenv("SODIUM_ADMIN_USER", "admin")
    if password:
        _validate_credentials(username, password)
    with _connection() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                timestamp DATETIME,
                config TEXT,
                kpis TEXT,
                FOREIGN KEY(username) REFERENCES users(username)
            );
        """)
        if password:
            encoded = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
            connection.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", (username, encoded))


def verify_user(username, password):
    """Verify existing hashes without changing or silently truncating a password."""
    if len(password.encode("utf-8")) > 72:
        return False
    with _connection() as connection:
        result = connection.execute(
            "SELECT password_hash FROM users WHERE username=?", (username,)
        ).fetchone()
    if not result:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), result[0])
    except (ValueError, TypeError):
        return False


def register_user(username, password):
    """Register a validated local account and preserve the legacy return contract."""
    try:
        _validate_credentials(username, password)
    except ValueError as exc:
        return False, str(exc)
    encoded = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    try:
        with _connection() as connection:
            connection.execute("INSERT INTO users VALUES (?, ?)", (username.strip(), encoded))
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def save_simulation(username, config_dict, kpi_dict):
    """Persist a simulation summary using bound SQL parameters."""
    clean = {
        key: value.item() if hasattr(value, "item") else value for key, value in kpi_dict.items()
    }
    with _connection() as connection:
        connection.execute(
            "INSERT INTO history (username,timestamp,config,kpis) VALUES (?,?,?,?)",
            (
                username,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                json.dumps(config_dict, allow_nan=False),
                json.dumps(clean, allow_nan=False),
            ),
        )


def get_history(username):
    """Return the ten most recent runs belonging to the selected account."""
    with _connection() as connection:
        rows = connection.execute(
            "SELECT timestamp,config,kpis FROM history WHERE username=? ORDER BY id DESC LIMIT 10",
            (username,),
        ).fetchall()
    return [
        {"time": row[0], "config": json.loads(row[1]), "kpis": json.loads(row[2])} for row in rows
    ]
