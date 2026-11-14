"""Legacy local accounts and history with scoped connections and explicit provisioning."""

from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import os
import secrets
from pathlib import Path
import sqlite3
import time
import bcrypt
from sodium_twin.database import connect_database
from utils.revoked_credentials import REVOKED_PASSWORD_FINGERPRINTS

DB_PATH = os.getenv(
    "SODIUM_LEGACY_DB", os.path.join(os.path.dirname(os.path.dirname(__file__)), "battery_app.db")
)


@contextmanager
def _connection():
    with connect_database(Path(DB_PATH)) as connection:
        yield connection


def _validate_credentials(username, password):
    if not isinstance(username, str) or not 3 <= len(username.strip()) <= 80:
        raise ValueError("Username must contain 3–80 characters.")
    if (
        not isinstance(password, str)
        or len(password) > 72
        or not 12 <= len(password.encode("utf-8")) <= 72
    ):
        raise ValueError("Password must contain 12–72 UTF-8 bytes.")


def init_db():
    """Create tables; provision an administrator only through explicit environment values."""
    password = os.getenv("SODIUM_ADMIN_PASSWORD")
    username = os.getenv("SODIUM_ADMIN_USER", "admin").strip()
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
            CREATE TABLE IF NOT EXISTS login_attempts (
                username TEXT PRIMARY KEY, failures INTEGER NOT NULL, first_attempt REAL NOT NULL
            );
        """)
        if password:
            encoded = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
            connection.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", (username, encoded))


@lru_cache(maxsize=1)
def _dummy_hash():
    return bcrypt.hashpw(os.urandom(32), bcrypt.gensalt())


def verify_user(username, password):
    """Throttle failed logins and reject weak or historically disclosed credentials."""
    try:
        _validate_credentials(username, password)
    except ValueError:
        return False
    username = username.strip()
    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        now = time.time()
        connection.execute("DELETE FROM login_attempts WHERE first_attempt < ?", (now - 300,))
        attempt = connection.execute(
            "SELECT failures FROM login_attempts WHERE username=?", (username,)
        ).fetchone()
        if attempt and attempt[0] >= 5:
            return False
        result = connection.execute(
            "SELECT password_hash FROM users WHERE username=?", (username,)
        ).fetchone()
        encoded = result[0] if result else _dummy_hash()
        if isinstance(encoded, str):
            encoded = encoded.encode()
        try:
            revoked = hashlib.sha256(encoded).hexdigest() in REVOKED_PASSWORD_FINGERPRINTS
            matched = bcrypt.checkpw(password.encode("utf-8"), encoded)
            valid = bool(result) and not revoked and matched
        except (ValueError, TypeError):
            valid = False
        if valid:
            connection.execute("DELETE FROM login_attempts WHERE username=?", (username,))
        else:
            # Bound attempt metadata independently of arbitrary submitted usernames.
            if connection.execute("SELECT count(*) FROM login_attempts").fetchone()[0] >= 1024:
                connection.execute(
                    "DELETE FROM login_attempts WHERE username IN (SELECT username FROM login_attempts ORDER BY first_attempt LIMIT 16)"
                )
            connection.execute(
                "INSERT INTO login_attempts VALUES (?,1,?) ON CONFLICT(username) DO UPDATE SET failures=failures+1",
                (username, now),
            )
        return valid


def register_user(username, password):
    """Register a validated local account and preserve the legacy return contract."""
    if os.getenv("SODIUM_ALLOW_REGISTRATION") != "1":
        return (
            False,
            "Self-registration is disabled. Ask the local operator to provision an account.",
        )
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


def reset_password(username, password):
    """Rotate one existing credential from the local administrative CLI."""
    _validate_credentials(username, password)
    encoded = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    with _connection() as connection:
        changed = connection.execute(
            "UPDATE users SET password_hash=? WHERE username=?", (encoded, username.strip())
        ).rowcount
        if not changed:
            raise ValueError("Account not found.")
        connection.execute("DELETE FROM login_attempts WHERE username=?", (username.strip(),))


def credential_revision(username):
    """Return an opaque session revision; never expose a stored password hash."""
    with _connection() as connection:
        result = connection.execute(
            "SELECT password_hash FROM users WHERE username=?", (username,)
        ).fetchone()
    if not result:
        return None
    value = result[0].encode() if isinstance(result[0], str) else result[0]
    revision = hashlib.sha256(value).hexdigest()
    return None if revision in REVOKED_PASSWORD_FINGERPRINTS else revision


def session_valid(username, revision):
    current = credential_revision(username)
    return bool(current and isinstance(revision, str) and secrets.compare_digest(current, revision))


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
