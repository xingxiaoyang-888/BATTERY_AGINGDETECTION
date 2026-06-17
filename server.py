# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import os
import uvicorn

# 导入我们重构后的核心组件
from models.simulation_engine import BatteryDigitalTwin
from models.data_structures import SimulationConfig
from utils.fmu_interface import discover_available_fmus, discover_available_models, get_ns_np_options

app = FastAPI(
    title="Sodium-Ion Battery Digital Twin API",
    description="基于 FMI 与 AI 双轨驱动的钠离子电池孪生平台后端",
    version="2.0.0"
)

# ==========================================
# 1. 声明 API 请求协议 (Pydantic Models)
# ==========================================
class SimulationRequest(BaseModel):
    """
    对齐 SimulationConfig 的前端请求结构
    """
    duration_s: float = 600.0
    pack_current: float = 50.0
    env_temp: float = 25.0
    init_soc: float = 80.0
    init_soh: float = 100.0

    # [FMU 关键参数] — 必须匹配已有 FMU 的 Ns/Np
    series_num: int = 8
    parallel_num: int = 2
    fault_mode: int = 1
    fault_s_index: int = 3
    fault_p_index: int = 1
    fault_severity: float = 0.0

# ==========================================
# 2. 路径与引擎管理
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FMU_DIR = os.path.join(BASE_DIR, "fmu_models")
AI_PATH = os.path.join(BASE_DIR, "models", "weights", "heat_ai_model.pkl")

# 引擎缓存: {(Ns, Np): BatteryDigitalTwin}
_engine_cache: Dict[tuple, BatteryDigitalTwin] = {}


def _get_fmu_path(ns: int, np: int) -> str:
    """根据 Ns/Np 查找对应的 FMU 文件路径"""
    fmu_name = f"SystemForFMI_{ns}x{np}.fmu"
    fmu_path = os.path.join(FMU_DIR, fmu_name)
    if os.path.exists(fmu_path):
        return fmu_path
    return None


def _get_or_create_engine(ns: int, np: int) -> BatteryDigitalTwin:
    """获取或创建指定规格的引擎实例（按需加载+缓存）"""
    key = (ns, np)
    if key not in _engine_cache:
        fmu_path = _get_fmu_path(ns, np)
        if fmu_path is None:
            available = [f['filename'] for f in discover_available_fmus(FMU_DIR)]
            raise HTTPException(
                status_code=400,
                detail=f"没有找到 {ns}s{np}p 对应的 FMU 文件。"
                       f"请先将 SystemForFMI_{ns}x{np}.mo 导出为 FMU 放入 fmu_models/。"
                       f"当前可用: {available}"
            )
        _engine_cache[key] = BatteryDigitalTwin(fmu_path=fmu_path, ai_model_path=AI_PATH)
    return _engine_cache[key]


# ==========================================
# 3. API 路由端点
# ==========================================

@app.get("/api/v1/health", tags=["System"])
async def check_health():
    """系统健康度检查"""
    fmus = discover_available_fmus(FMU_DIR)
    return {
        "status": "Running",
        "fmu_count": len(fmus),
        "available_configs": [f"{f['ns']}s{f['np']}p" for f in fmus],
        "backend": "FastAPI + FMU Runtime (multi-FMU)",
        "server_time": time.time()
    }


@app.get("/api/v1/fmu/configurations", tags=["FMU"])
async def list_fmu_configurations():
    """
    返回所有可用的电池包规格（含已导出 FMU 和仅有 .mo 的）

    前端用此接口构建 Ns → Np 的级联下拉菜单
    """
    fmu_dir = FMU_DIR
    # 已导出的 FMU
    fmu_configs = discover_available_fmus(fmu_dir)
    fmu_set = {(c['ns'], c['np']) for c in fmu_configs}

    # 所有 .mo 模型（含未导出 FMU 的）
    all_models = discover_available_models(os.path.join(BASE_DIR, "mo_system_models"))
    ns_np_options = get_ns_np_options(all_models)

    # 构建响应：标记哪些已有 FMU、哪些还未导出
    configurations = []
    for ns in sorted(ns_np_options.keys()):
        np_list = ns_np_options[ns]
        np_details = []
        for np in np_list:
            has_fmu = (ns, np) in fmu_set
            np_details.append({
                'np': np,
                'has_fmu': has_fmu,
                'total_cells': ns * np,
                'nominal_voltage': round(ns * 3.1, 1),
            })
        configurations.append({
            'ns': ns,
            'np_options': np_details,
        })

    return {
        "configurations": configurations,
        "total_fmus_ready": len(fmu_configs),
        "default_ns": 8,   # 默认选中（必须有 FMU）
        "default_np": 2,
    }


@app.post("/api/v1/simulate/predict", tags=["Simulation"])
async def run_twin_prediction(req: SimulationRequest):
    """
    [核心端点] 执行钠离子电池数字孪生预测
    根据 Ns/Np 自动匹配对应 FMU → 驱动 C++ 求解器 → 返回 3D 热场矩阵
    """
    try:
        # 1. 根据前端选择的规格加载对应引擎
        engine = _get_or_create_engine(req.series_num, req.parallel_num)

        # 2. 将请求参数转换为内部仿真配置
        config = SimulationConfig(
            sim_duration_s=req.duration_s,
            pack_current_a=req.pack_current,
            env_temp_c=req.env_temp,
            init_soc=req.init_soc,
            init_soh=req.init_soh,
            series_num=req.series_num,
            parallel_num=req.parallel_num,
            fault_mode=req.fault_mode,
            fault_s_index=req.fault_s_index,
            fault_p_index=req.fault_p_index,
            fault_severity=req.fault_severity
        )

        # 3. 执行解算
        _, kpis, ts_data = engine.run_profile(config)

        # 4. 封装返回结果
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
                    "t_min": ts_data.temperatures_min
                },
                "spatial_thermal_matrix": ts_data.temp_matrix_frames,
                "spatial_soc_matrix": ts_data.soc_matrix_frames,
                "spatial_soh_matrix": ts_data.soh_matrix_frames,
                "diagnostics": kpis.diagnostic_warnings
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"孪生引擎解算崩溃: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/v1/aging/fast_scan", tags=["Aging"])
async def fast_aging_scan(soh: float, temp: float, rate: float):
    """
    利用老化算法进行快速寿命扫描 (不跑物理仿真，仅做逻辑推算)
    """
    from models.aging_algorithm import AgingModel

    step_loss = AgingModel.calculate_step_loss(
        temp_c=temp,
        current_a=rate * 50.0,
        soc_pct=50.0,
        dt_seconds=3600.0,
        cell_capacity_ah=50.0
    )

    return {
        "input_soh": soh,
        "single_hour_loss_ppm": round(step_loss * 1e6, 2),
        "estimated_remaining_days": int((soh - 80) / (step_loss * 24 * 100)) if step_loss > 0 else "Infinity"
    }


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)