"""Verify removed default credentials and isolation of retained local accounts."""

import pytest

pytest.importorskip("bcrypt")
from utils import db_manager


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_PATH", str(tmp_path / "accounts.db"))
    monkeypatch.delenv("SODIUM_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("SODIUM_ALLOW_REGISTRATION", raising=False)
    return db_manager


def test_no_default_account_and_password_validation(accounts, monkeypatch):
    accounts.init_db()
    assert not accounts.register_user("researcher", "valid-test-password")[0]
    monkeypatch.setenv("SODIUM_ALLOW_REGISTRATION", "1")
    assert not accounts.verify_user("admin", "1234")
    assert not accounts.register_user("researcher", "1234")[0]
    assert not accounts.register_user("researcher", "a" * 73)[0]
    assert accounts.register_user("researcher", "valid-test-password")[0]
    assert accounts.verify_user("researcher", "valid-test-password")
    assert not accounts.verify_user("researcher", "incorrect")
    assert not accounts.register_user("researcher", "valid-test-password")[0]


def test_explicit_provisioning_does_not_overwrite_an_account(accounts, monkeypatch):
    monkeypatch.setenv("SODIUM_ADMIN_PASSWORD", "valid-admin-password")
    accounts.init_db()
    monkeypatch.setenv("SODIUM_ADMIN_PASSWORD", "different-admin-password")
    accounts.init_db()
    assert accounts.verify_user("admin", "valid-admin-password")
    assert not accounts.verify_user("admin", "different-admin-password")


def test_account_history_is_scoped_and_sql_parameters_are_bound(accounts, monkeypatch):
    accounts.init_db()
    monkeypatch.setenv("SODIUM_ALLOW_REGISTRATION", "1")
    assert accounts.register_user("alice", "valid-test-password")[0]
    assert accounts.register_user("bob", "valid-test-password")[0]
    accounts.save_simulation("alice", {"current": 1}, {"soh": 99})
    accounts.save_simulation("bob", {"current": 2}, {"soh": 98})
    assert accounts.get_history("alice")[0]["kpis"]["soh"] == 99
    assert accounts.get_history("' OR 1=1 --") == []


def test_foreign_keys_reject_orphan_history(accounts):
    import sqlite3

    accounts.init_db()
    with pytest.raises(sqlite3.IntegrityError):
        accounts.save_simulation("missing-user", {}, {})


def test_failed_logins_are_throttled_and_password_rotation_revokes_sessions(accounts, monkeypatch):
    monkeypatch.setenv("SODIUM_ADMIN_PASSWORD", "old-valid-password")
    accounts.init_db()
    revision = accounts.credential_revision("admin")
    assert accounts.session_valid("admin", revision)
    accounts.save_simulation("admin", {}, {"soh": 99})
    for _ in range(5):
        assert not accounts.verify_user("admin", "wrong-valid-length-password")
    assert not accounts.verify_user("admin", "old-valid-password")
    accounts.reset_password("admin", "new-valid-password")
    assert not accounts.session_valid("admin", revision)
    assert accounts.verify_user("admin", "new-valid-password")
    assert accounts.get_history("admin")[0]["kpis"]["soh"] == 99


def test_published_hash_is_rejected_even_with_correct_password(accounts, monkeypatch):
    monkeypatch.setenv("SODIUM_ADMIN_PASSWORD", "known-valid-password")
    accounts.init_db()
    revision = accounts.credential_revision("admin")
    monkeypatch.setattr(accounts, "REVOKED_PASSWORD_FINGERPRINTS", {revision})
    assert not accounts.verify_user("admin", "known-valid-password")
    assert not accounts.session_valid("admin", revision)
