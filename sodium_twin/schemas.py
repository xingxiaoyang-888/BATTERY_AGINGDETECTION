"""Explicit physical units and bounded requests for the v2 API."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ImportRequest(Contract):
    name: str = Field(min_length=1, max_length=80)
    csv: str = Field(min_length=20, max_length=2_000_000)
    chemistry: Literal["sodium-ion", "lithium-ion"] = "sodium-ion"


class ForecastRequest(Contract):
    dataset_id: str = Field(min_length=1, max_length=64)
    cell_id: str = Field(min_length=1, max_length=80)
    horizon: int = Field(default=600, ge=20, le=3000)
    eol_pct: float = Field(default=80, ge=50, le=95)
    temperature_c: float = Field(default=25, ge=-20, le=60)
    c_rate: float = Field(default=1, ge=0.1, le=5)
    model: Literal["robust-trend", "patchtst", "itransformer", "selective-ssm", "tcn"] = (
        "robust-trend"
    )


class SimulationRequest(Contract):
    series: int = Field(default=16, ge=1, le=480)
    parallel: int = Field(default=4, ge=1, le=8)
    capacity_ah: float = Field(default=50, ge=0.1, le=500)
    current_a: float = Field(default=40, ge=-500, le=500)
    duration_s: int = Field(default=1800, ge=10, le=7200)
    ambient_c: float = Field(default=25, ge=-20, le=60)
    initial_soc_pct: float = Field(default=80, ge=5, le=95)
    initial_soh_pct: float = Field(default=95, ge=50, le=100)
    cooling_w_k: float = Field(default=2, ge=0.1, le=20)
