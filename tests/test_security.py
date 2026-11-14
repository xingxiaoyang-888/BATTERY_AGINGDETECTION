"""Reproduce trust-boundary failures without touching real user data or credentials."""

import asyncio
import base64
from dataclasses import replace
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sodium_twin.api import create_app
from sodium_twin.config import Settings
from sodium_twin.data import parse_csv
from sodium_twin.database import StorageLimitError
from sodium_twin.security import AccessPolicy, WorkspaceBoundary, require_safe_bind
from sodium_twin.store import Store

KEY = "a-test-only-key-with-at-least-thirty-two-characters"


def client_for(tmp_path, policy=None, *, peer="127.0.0.1", host="localhost"):
    settings = Settings(
        tmp_path / "data", tmp_path / "models", tmp_path / "web", policy or AccessPolicy()
    )
    return TestClient(create_app(settings), base_url=f"http://{host}", client=(peer, 50000))


@pytest.mark.parametrize(
    "host", ["attacker.example", "localhost.attacker.example", "127.0.0.1.attacker.example"]
)
def test_dns_rebinding_host_cannot_read_or_write(tmp_path, host):
    with client_for(tmp_path, host=host) as client:
        assert client.get("/api/v2/datasets").status_code == 400
        assert (
            client.post(
                "/api/v2/simulations", json={"duration_s": 10}, headers={"Origin": f"http://{host}"}
            ).status_code
            == 400
        )


def test_non_loopback_peer_cannot_use_keyless_workspace(tmp_path):
    with client_for(tmp_path, peer="198.51.100.23") as client:
        assert client.get("/api/v2/datasets").status_code == 403
        assert (
            client.get("/api/v2/datasets", headers={"X-Forwarded-For": "127.0.0.1"}).status_code
            == 403
        )


@pytest.mark.parametrize(
    "route", ["/", "/docs", "/openapi.json", "/api/v2/datasets", "/api/v2/runs", "/api/v2/health"]
)
def test_key_protects_pages_metadata_and_all_data_routes(tmp_path, route):
    with client_for(tmp_path, AccessPolicy(api_key=KEY), peer="198.51.100.23") as client:
        response = client.get(route)
        assert response.status_code == 401
        assert "Basic" in response.headers["www-authenticate"]
        assert KEY not in response.text


@pytest.mark.parametrize("scheme", ["Bearer", "Basic"])
def test_authenticated_operator_can_access_remote_workspace(tmp_path, scheme):
    value = KEY if scheme == "Bearer" else base64.b64encode(f"workspace:{KEY}".encode()).decode()
    with client_for(tmp_path, AccessPolicy(api_key=KEY), peer="198.51.100.23") as client:
        response = client.get("/api/v2/datasets", headers={"Authorization": f"{scheme} {value}"})
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-frame-options"] == "DENY"


@pytest.mark.parametrize(
    "headers",
    [{"Origin": "null"}, {"Origin": "http://attacker.example"}, {"Sec-Fetch-Site": "cross-site"}],
)
def test_cross_site_reads_and_writes_are_rejected(client, headers):
    assert client.get("/api/v2/datasets", headers=headers).status_code == 403
    assert (
        client.post("/api/v2/simulations", json={"duration_s": 10}, headers=headers).status_code
        == 403
    )


def test_dev_origin_requires_explicit_opt_in(tmp_path):
    with client_for(tmp_path) as client:
        assert (
            client.post(
                "/api/v2/simulations",
                json={"duration_s": 10},
                headers={"Origin": "http://localhost:5173"},
            ).status_code
            == 403
        )
    policy = AccessPolicy(dev_origins=("http://localhost:5173",))
    with client_for(tmp_path, policy) as client:
        assert (
            client.post(
                "/api/v2/simulations",
                json={"duration_s": 10},
                headers={"Origin": "http://localhost:5173"},
            ).status_code
            == 200
        )


def test_chunked_body_cannot_bypass_limit_or_persist_a_run(client):
    response = client.post(
        "/api/v2/simulations",
        content=iter([b" " * 2_200_000, b'{"duration_s":10}']),
        headers={"Content-Type": "application/json"},
    )
    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert client.get("/api/v2/runs").json() == []


@pytest.mark.parametrize("value", ["-1", "x", "3"])
def test_invalid_or_inconsistent_content_length_is_rejected(client, value):
    response = client.post(
        "/api/v2/simulations",
        content=b"{}",
        headers={"Content-Length": value, "Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_write_rate_and_concurrency_are_bounded(tmp_path):
    with client_for(tmp_path, AccessPolicy(writes_per_minute=1)) as client:
        assert client.post("/api/v2/simulations", json={"duration_s": 10}).status_code == 200
        assert client.post("/api/v2/simulations", json={"duration_s": 10}).status_code == 429
        assert client.get("/api/v2/runs").status_code == 200

    async def scenario():
        app = FastAPI()
        entered, release = asyncio.Event(), asyncio.Event()

        @app.post("/work")
        async def work():
            entered.set()
            await release.wait()
            return {"ok": True}

        app.add_middleware(WorkspaceBoundary, policy=AccessPolicy(concurrent_jobs=1))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            first = asyncio.create_task(client.post("/work"))
            await asyncio.wait_for(entered.wait(), 2)
            assert (await client.post("/work")).status_code == 429
            release.set()
            assert (await first).status_code == 200

    asyncio.run(scenario())


def test_failed_authentication_is_throttled_without_locking_valid_key(tmp_path):
    with client_for(tmp_path, AccessPolicy(api_key=KEY)) as client:
        for _ in range(30):
            assert client.get("/api/v2/datasets").status_code == 401
        assert client.get("/api/v2/datasets").status_code == 429
        assert (
            client.get("/api/v2/datasets", headers={"Authorization": f"Bearer {KEY}"}).status_code
            == 200
        )


def test_unsafe_network_binding_and_weak_keys_fail_closed():
    with pytest.raises(ValueError):
        require_safe_bind("0.0.0.0", AccessPolicy())
    with pytest.raises(ValueError):
        AccessPolicy(api_key="short")
    with pytest.raises(ValueError):
        AccessPolicy(allowed_hosts=("*",))
    require_safe_bind("0.0.0.0", AccessPolicy(api_key=KEY))
    require_safe_bind("127.0.0.1", AccessPolicy())


def test_dataset_run_and_database_capacity_limits_are_enforced(tmp_path, cycle_csv):
    store = Store(tmp_path / "quota", max_datasets=1, max_runs=1)
    with pytest.raises(StorageLimitError):
        store.import_frame("extra", "sodium-ion", parse_csv(cycle_csv))
    saved = store.save_run("test", {}, {"value": 1})
    with pytest.raises(StorageLimitError):
        store.save_run("test", {}, {"value": 2})
    assert store.run(saved["id"])["result"]["value"] == 1
    bounded = Store(tmp_path / "size")
    bounded.max_bytes = bounded.path.stat().st_size + 4096
    with pytest.raises(StorageLimitError):
        bounded.save_run("test", {}, {"payload": "x" * 1_000_000})
    assert bounded.runs() == []


def test_sql_injection_identifiers_do_not_return_private_rows(client, cycle_csv):
    payload = {"name": "Robert'); DROP TABLE datasets; --", "csv": cycle_csv}
    assert client.post("/api/v2/datasets", json=payload).status_code == 201
    assert len(client.get("/api/v2/datasets").json()) == 2
    assert client.get("/api/v2/runs/' OR 1=1 --").status_code == 404
    for path in (
        "/data/workspace/workspace.sqlite3",
        "/battery_app.db",
        "/.env",
        "/assets/%2e%2e/%2e%2e/battery_app.db",
    ):
        assert client.get(path).status_code == 404


@pytest.mark.skipif(os.name != "posix", reason="Windows database privacy uses filesystem ACLs.")
def test_database_is_owner_only_on_posix(tmp_path):
    store = Store(tmp_path / "private")
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_legacy_deserialization_requires_explicit_trust(monkeypatch, tmp_path):
    from sodium_twin.legacy_artifacts import load_pickle_model, load_tensor_checkpoint

    monkeypatch.delenv("SODIUM_TRUST_LEGACY_ARTIFACTS", raising=False)
    for load in (load_pickle_model, load_tensor_checkpoint):
        with pytest.raises(ValueError, match="disabled"):
            load(tmp_path / "untrusted.pkl")


def test_new_weights_round_trip_without_using_pickle(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from sodium_twin.ml.networks import BatteryForecaster, ModelConfig

    def forbidden(*args, **kwargs):
        raise AssertionError("Pickle-based checkpoint loading is forbidden.")

    monkeypatch.setattr(torch, "load", forbidden)
    cfg = ModelConfig(width=16, layers=1, lookback=16, horizon=20)
    source, target = BatteryForecaster(cfg).eval(), BatteryForecaster(cfg).eval()
    path = tmp_path / "random-test-state.safetensors"
    safetensors.save_file(source.state_dict(), str(path))
    target.load_state_dict(safetensors.load(path.read_bytes()))
    x, anchor = torch.randn(1, 16, 4), torch.tensor([95.0])
    with torch.inference_mode():
        torch.testing.assert_close(source(x, anchor), target(x, anchor))


def test_legacy_http_boundary_rejects_remote_access_and_nonfinite_input():
    import server

    with TestClient(
        server.app, base_url="http://localhost", client=("198.51.100.10", 60000)
    ) as client:
        assert client.get("/api/v1/health").status_code == 403
    with TestClient(server.app, base_url="http://localhost", client=("127.0.0.1", 60000)) as client:
        response = client.post(
            "/api/v1/simulate/predict",
            content='{"pack_current":NaN}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "NaN" not in response.text


def test_quota_errors_are_actionable_without_exposing_sql(client, cycle_csv):
    client.app.state.store.max_datasets = 1
    response = client.post("/api/v2/datasets", json={"name": "extra", "csv": cycle_csv})
    assert response.status_code == 507
    assert "limit" in response.json()["detail"]
    assert "INSERT" not in response.text


def test_slow_request_body_times_out_without_running_the_application():
    async def scenario():
        called, messages = [], []

        async def app(scope, receive, send):
            called.append(True)

        async def receive():
            await asyncio.sleep(1)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v2/simulations",
            "headers": [(b"host", b"localhost")],
            "client": ("127.0.0.1", 50000),
        }
        await WorkspaceBoundary(app, AccessPolicy(body_timeout=0.01))(scope, receive, send)
        assert messages[0]["status"] == 408
        assert not called

    asyncio.run(scenario())
