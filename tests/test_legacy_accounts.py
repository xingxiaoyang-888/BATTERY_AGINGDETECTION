"""Verify removed default credentials and isolation of retained local accounts."""

import pytest

pytest.importorskip("bcrypt")
from utils import db_manager


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_PATH", str(tmp_path / "accounts.db"))
    monkeypatch.delenv("SODIUM_ADMIN_PASSWORD", raising=False)
    return db_manager


def test_no_default_account_and_password_validation(accounts):
    accounts.init_db()
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


def test_account_history_is_scoped_and_sql_parameters_are_bound(accounts):
    accounts.init_db()
    accounts.save_simulation("alice", {"current": 1}, {"soh": 99})
    accounts.save_simulation("bob", {"current": 2}, {"soh": 98})
    assert accounts.get_history("alice")[0]["kpis"]["soh"] == 99
    assert accounts.get_history("' OR 1=1 --") == []
