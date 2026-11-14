# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Optional, Dict, Any
import time
import os
import threading
import uvicorn
from starlette.concurrency import run_in_threadpool


from models.simulation_engine import BatteryDigitalTwin
from models.data_structures import SimulationConfig
from utils.fmu_interface import (
    discover_available_fmus,
    discover_available_models,
    get_ns_np_options,
)

app = FastAPI(
    title="Sodium-Ion Battery Digital Twin API",
    description="基于 FMI 与 AI 双轨驱动的钠离子电池孪生平台后端",
    version="2.0.0",
)

# ==========================================


# ==========================================
class StrictRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class SimulationRequest(StrictRequest):
    """Validated input for the legacy FMU simulation contract."""

    duration_s: float = Field(default=600.0, gt=0, le=86400)
    pack_current: float = Field(default=50.0, ge=-5000, le=5000)
    current_profile: Optional[List["CurrentProfilePoint"]] = Field(default=None, max_length=10000)
    env_temp: float = Field(default=25.0, ge=-40, le=80)
    initial_cell_temp: Optional[float] = Field(default=None, ge=-40, le=100)
    coolant_inlet_temp: Optional[float] = Field(default=None, ge=-40, le=100)
    coolant_flow_kg_s: float = Field(default=0.035, gt=0, le=2.0)
    cooling_ua_w_per_k: float = Field(default=2.0, gt=0, le=100.0)
    cell_capacity_ah: float = Field(default=50.0, gt=0, le=2000.0)
    init_soc: float = Field(default=80.0, ge=0, le=100)
    init_soh: float = Field(default=100.0, ge=5, le=100)

    series_num: int = Field(default=8, ge=1, le=480)
    parallel_num: int = Field(default=2, ge=1, le=8)
    fault_mode: int = Field(default=1, ge=1, le=6)
    fault_s_index: int = Field(default=3, ge=1)
    fault_p_index: int = Field(default=1, ge=1)
    fault_severity: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_profile_and_fault(self):
        if self.fault_s_index > self.series_num or self.fault_p_index > self.parallel_num:
            raise ValueError("故障位置超出电池包拓扑")
        if self.current_profile:
            times = [point.time_s for point in self.current_profile]
            if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("电流工况至少两个点且时间必须严格递增")
            if times[0] < 0 or times[-1] > self.duration_s + 1e-9:
                raise ValueError("电流工况时间必须位于仿真时长内")
        return self


class CurrentProfilePoint(StrictRequest):
    time_s: float = Field(ge=0)
    current_a: float = Field(ge=-5000, le=5000)


SimulationRequest.model_rebuild()


class LifetimeHistoryPoint(StrictRequest):
    """One observed cycle; SOH is expressed in percent."""

    cycle_index: int = Field(ge=0)
    soh_pct: float = Field(gt=0, le=105)


class LifetimeScenarioRequest(StrictRequest):
    """Future operating conditions for cohort matching and domain checks."""

    temperature_c: float = Field(default=25.0, ge=-30, le=80)
    temperature_spread_c: float = Field(default=3.0, ge=0, le=50)
    c_rate_charge: float = Field(default=1.0, gt=0, le=15)
    c_rate_discharge: float = Field(default=1.0, gt=0, le=15)
    rest_time_h: float = Field(default=0.0, ge=0, le=720)
    soc_min_pct: float = Field(default=0.0, ge=0, le=100)
    soc_max_pct: float = Field(default=100.0, ge=0, le=100)
    nominal_capacity_ah: float = Field(default=1.2, gt=0, le=2000)

    @model_validator(mode="after")
    def validate_soc_window(self):
        if self.soc_min_pct >= self.soc_max_pct:
            raise ValueError("SOC 下限必须小于 SOC 上限")
        return self


class LifetimePredictionRequest(StrictRequest):
    """Lifetime request; an empty history selects the cohort baseline."""

    current_soh_pct: float = Field(gt=0, le=105)
    current_cycle: int = Field(default=0, ge=0)
    eol_soh_pct: float = Field(default=80.0, gt=0, le=100)
    cycles_per_day: float = Field(default=1.0, gt=0, le=100)
    max_horizon_cycles: int = Field(default=5000, gt=0, le=50000)
    scenario: LifetimeScenarioRequest = Field(default_factory=LifetimeScenarioRequest)
    history: List[LifetimeHistoryPoint] = Field(default_factory=list, max_length=20000)

    @model_validator(mode="after")
    def validate_history(self):
        cycles = [point.cycle_index for point in self.history]
        if any(current <= previous for previous, current in zip(cycles, cycles[1:])):
            raise ValueError("历史循环编号必须严格递增")
        if cycles and cycles[-1] > self.current_cycle:
            raise ValueError("历史循环不能晚于当前循环")
        if self.history and self.history[-1].cycle_index == self.current_cycle:
            if abs(self.history[-1].soh_pct - self.current_soh_pct) > 3.0:
                raise ValueError("当前 SOH 与最后一个历史点偏差超过 3 个百分点")
        return self


# ==========================================

# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FMU_DIR = os.path.join(BASE_DIR, "fmu_models")
AI_PATH = os.getenv("SODIUM_THERMAL_MODEL")
LIFETIME_FEATURE_PATH = os.getenv(
    "SODIUM_LIFETIME_FEATURE_PATH",
    os.path.join(BASE_DIR, "models", "data", "processed", "sodium_ion", "feature_table.parquet"),
)


_engine_cache: Dict[tuple, BatteryDigitalTwin] = {}
_lifetime_service = None
_lifetime_service_lock = threading.Lock()
_engine_lock = threading.RLock()


def _get_lifetime_service():
    """Lazily initialize the lifetime cohort and optional learned models."""
    global _lifetime_service
    if _lifetime_service is None:
        with _lifetime_service_lock:
            if _lifetime_service is None:
                from models.soh_ai.lifetime import SodiumLifetimePredictor

                _lifetime_service = SodiumLifetimePredictor(LIFETIME_FEATURE_PATH)
    return _lifetime_service


def _get_fmu_path(ns: int, np: int) -> str:
    """Resolve a compiled FMU for the requested series/parallel topology."""
    fmu_name = f"SystemForFMI_{ns}x{np}.fmu"
    fmu_path = os.path.join(FMU_DIR, fmu_name)
    if os.path.exists(fmu_path):
        return fmu_path
    return None


def _get_or_create_engine(ns: int, np: int) -> BatteryDigitalTwin:
    """Retrieve or initialize a cached engine for one fixed FMU topology."""
    key = (ns, np)
    if key not in _engine_cache:
        fmu_path = _get_fmu_path(ns, np)
        if fmu_path is None:
            available = [f["filename"] for f in discover_available_fmus(FMU_DIR)]
            raise HTTPException(
                status_code=400,
                detail=f"没有找到 {ns}s{np}p 对应的 FMU 文件。"
                f"请先在 OMEdit 中加载 BatterySystemEngineering.mo 库，"
                f"再打开 mo_system_models/SystemForFMI.mo 模板，"
                f"设置 Ns={ns} Np={np} 后导出 FMU，"
                f"并将生成的 SystemForFMI.fmu 重命名为 SystemForFMI_{ns}x{np}.fmu 放入 fmu_models/。"
                f"当前可用: {available}",
            )
        _engine_cache[key] = BatteryDigitalTwin(fmu_path=fmu_path, ai_model_path=AI_PATH)
    return _engine_cache[key]


# ==========================================

# ==========================================


def _get_engine_locked(ns, np):
    """Serialize native engine initialization and environment changes."""
    with _engine_lock:
        return _get_or_create_engine(ns, np)


def _run_profile_locked(engine, config):
    """Run native FMUs off the event loop and serialize shared runtime access."""
    with _engine_lock:
        return engine.run_profile(config)


@app.get("/api/v1/health", tags=["System"])
async def check_health():
    """Check health."""
    fmus = discover_available_fmus(FMU_DIR)
    return {
        "status": "Running",
        "fmu_count": len(fmus),
        "available_configs": [f"{f['ns']}s{f['np']}p" for f in fmus],
        "backend": "FastAPI + FMU Runtime (multi-FMU)",
        "server_time": time.time(),
    }


@app.get("/api/v1/fmu/configurations", tags=["FMU"])
async def list_fmu_configurations():
    """List exported FMUs and source-only Modelica topologies."""
    fmu_dir = FMU_DIR

    fmu_configs = discover_available_fmus(fmu_dir)
    fmu_set = {(c["ns"], c["np"]) for c in fmu_configs}

    all_models = discover_available_models(os.path.join(BASE_DIR, "mo_system_models"))
    ns_np_options = get_ns_np_options(all_models)

    configurations = []
    for ns in sorted(ns_np_options.keys()):
        np_list = ns_np_options[ns]
        np_details = []
        for np in np_list:
            has_fmu = (ns, np) in fmu_set
            np_details.append(
                {
                    "np": np,
                    "has_fmu": has_fmu,
                    "total_cells": ns * np,
                    "nominal_voltage": round(ns * 3.1, 1),
                }
            )
        configurations.append(
            {
                "ns": ns,
                "np_options": np_details,
            }
        )

    return {
        "configurations": configurations,
        "total_fmus_ready": len(fmu_configs),
        "default_ns": 8,
        "default_np": 2,
    }


@app.post("/api/v1/simulate/predict", tags=["Simulation"])
async def run_twin_prediction(req: SimulationRequest):
    """Execute the native FMU and return pack metrics and cell matrices."""
    try:
        engine = await run_in_threadpool(_get_engine_locked, req.series_num, req.parallel_num)

        config = SimulationConfig(
            sim_duration_s=req.duration_s,
            pack_current_a=req.pack_current,
            current_profile=(
                [(point.time_s, point.current_a) for point in req.current_profile]
                if req.current_profile
                else None
            ),
            env_temp_c=req.env_temp,
            initial_cell_temp_c=req.initial_cell_temp,
            coolant_inlet_temp_c=req.coolant_inlet_temp,
            coolant_flow_kg_s=req.coolant_flow_kg_s,
            cooling_ua_w_per_k=req.cooling_ua_w_per_k,
            cell_capacity_ah=req.cell_capacity_ah,
            init_soc=req.init_soc,
            init_soh=req.init_soh,
            series_num=req.series_num,
            parallel_num=req.parallel_num,
            fault_mode=req.fault_mode,
            fault_s_index=req.fault_s_index,
            fault_p_index=req.fault_p_index,
            fault_severity=req.fault_severity,
        )

        _, kpis, ts_data = await run_in_threadpool(_run_profile_locked, engine, config)

        return {
            "code": 200,
            "message": f"数字孪生解算成功 [{req.series_num}s{req.parallel_num}p]",
            "payload": {
                "summary": kpis.to_dict(),
                "time_series": {
                    "time": ts_data.timestamps,
                    "voltage": ts_data.voltages,
                    "current": ts_data.currents,
                    "soc": ts_data.soc_array,
                    "t_max": ts_data.temperatures_max,
                    "t_min": ts_data.temperatures_min,
                },
                "spatial_thermal_matrix": ts_data.temp_matrix_frames,
                "spatial_soc_matrix": ts_data.soc_matrix_frames,
                "spatial_soh_matrix": ts_data.soh_matrix_frames,
                "diagnostics": kpis.diagnostic_warnings,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"孪生引擎解算崩溃: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/v1/lifetime/evidence", tags=["Lifetime"])
async def get_lifetime_evidence():
    """Report cohort EOL events, right censoring, and model availability."""
    try:
        service = await run_in_threadpool(_get_lifetime_service)
        return {
            "code": 200,
            "message": "钠电寿命证据摘要",
            "payload": await run_in_threadpool(service.evidence_summary),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"寿命证据加载失败: {exc}")


@app.post("/api/v1/lifetime/predict", tags=["Lifetime"])
async def predict_sodium_lifetime(req: LifetimePredictionRequest):
    """Fuse the censored cohort, observed history, and optional XGBoost trend."""
    scenario = req.scenario.model_dump()
    scenario["soc_min"] = scenario.pop("soc_min_pct") / 100.0
    scenario["soc_max"] = scenario.pop("soc_max_pct") / 100.0
    history = [
        {"cycle_index": point.cycle_index, "soh": point.soh_pct / 100.0} for point in req.history
    ]
    if not history or history[-1]["cycle_index"] < req.current_cycle:
        history.append({"cycle_index": req.current_cycle, "soh": req.current_soh_pct / 100.0})
    else:
        history[-1]["soh"] = req.current_soh_pct / 100.0

    try:
        service = await run_in_threadpool(_get_lifetime_service)
        result = await run_in_threadpool(
            service.predict,
            current_soh=req.current_soh_pct / 100.0,
            current_cycle=req.current_cycle,
            eol_threshold=req.eol_soh_pct / 100.0,
            scenario=scenario,
            history=history,
            cycles_per_day=req.cycles_per_day,
            max_horizon_cycles=req.max_horizon_cycles,
        )
        return {
            "code": 200,
            "message": "钠离子电池寿命预测完成",
            "payload": result,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"寿命预测失败: {exc}")


@app.get("/api/v1/aging/fast_scan", tags=["Aging"], deprecated=True)
async def fast_aging_scan(soh: float, temp: float, rate: float):
    """Compatibility endpoint backed by the sodium lifetime service."""
    if not 0 < soh <= 105 or not -30 <= temp <= 80 or not 0 < rate <= 15:
        raise HTTPException(status_code=422, detail="SOH、温度或C倍率超出允许范围")
    service = await run_in_threadpool(_get_lifetime_service)
    result = await run_in_threadpool(
        service.predict,
        current_soh=soh / 100.0,
        current_cycle=0,
        eol_threshold=0.80,
        scenario={"temperature_c": temp, "c_rate_charge": rate, "c_rate_discharge": rate},
        cycles_per_day=1.0,
    )
    return {
        "input_soh": soh,
        "single_cycle_loss_ppm": round(result["prediction"]["degradation_rate_per_cycle"] * 1e6, 2),
        "estimated_remaining_days": result["prediction"]["rul_days"],
        "confidence": result["uncertainty"],
        "warnings": result["warnings"],
        "deprecated": "请迁移到 POST /api/v1/lifetime/predict",
    }


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("SODIUM_HOST", "127.0.0.1"), port=8000)
