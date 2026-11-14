"""Isolated test workspaces; no production data or model artifacts are modified."""

import pytest
from fastapi.testclient import TestClient
from sodium_twin.api import create_app
from sodium_twin.config import Settings


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(tmp_path / "data", tmp_path / "models", tmp_path / "web"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def cycle_csv():
    header = "cell_id,cycle_index,soh_pct,temperature_c,c_rate,capacity_ah\n"
    return header + "\n".join(f"CELL-01,{i},{100 - 0.02 * i},25,1,50" for i in range(100))
