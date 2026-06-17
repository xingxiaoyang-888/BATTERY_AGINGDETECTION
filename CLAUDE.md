# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
**Language Rules:**
- 请务必永远使用 **简体中文 (Simplified Chinese)** 与我进行对话。
- 无论是解释架构、分析代码还是生成注释，都必须输出中文。
## Architecture Overview

This is a **Sodium-Ion Battery Digital Twin platform** — a two-process web application with a Streamlit frontend and a FastAPI backend. The core simulation engine drives an FMU (Functional Mock-up Unit) compiled from Modelica, augmented with AI thermal compensation and aging algorithms.

```
Browser → Streamlit (main.py, :8501) ──HTTP──→ FastAPI (server.py, :8000) → FMUClient (fmpy) → SystemForFMI.fmu
                                                    ↓
                                            BatteryDigitalTwin
                                            ├── AgingModel (NREL semi-empirical)
                                            ├── BatteryPhysicsLibrary
                                            └── AI compensator (RandomForest + PyTorch DNN)
```

### Key directories

| Directory | Role |
|-----------|------|
| `views/` | Streamlit UI pages — dashboard, sidebar, spatial (3D), components, login |
| `models/` | Simulation engine, data structures, aging algorithm, physics library |
| `utils/` | FMU interface (fmpy wrapper), SQLite db, PDF report, CSS loader, CSV parser |
| `config/` | Theme colors, CSS injection |
| `test/` | Unit tests (unittest) |
| `fmu_models/` | Compiled FMU binary (`SystemForFMI.fmu`) |
| `assets/` | CSS stylesheet, CSV template for cycle profiles |

## Running the Application

The app requires **two processes** running simultaneously:

```bash
# Terminal 1: Start the FastAPI backend (port 8000)
python server.py

# Terminal 2: Start the Streamlit frontend (port 8501)
streamlit run main.py
```

**FMU prerequisite**: OpenModelica must be installed at `D:\openmodelica\bin` (Windows). The FMU client (`utils/fmu_interface.py`) adds this to `PATH` and calls `os.add_dll_directory()` at runtime. Without it, FMU simulation will fail — the Streamlit dashboard will show a connection error from the backend.

**Default login**: `admin` / `1234` (stored in SQLite via bcrypt).

## Testing

```bash
# Run all unit tests
python -m pytest test/ -v

# Or with unittest
python -m unittest discover test/ -v

# Run a single test file
python -m unittest test.test_engine -v
```

Tests cover: aging model (calendar loss, cycle stress), physics library (OCV bounds, entropic heat direction). The test suite does **not** require the FMU or backend to be running.

## Key Architectural Patterns

### FMU simulation flow
1. `server.py` receives `SimulationRequest` (duration, current, temps, pack topology, fault injection params).
2. `BatteryDigitalTwin.run_profile()` translates to FMU inputs and calls `FMUClient.run_simulation()`.
3. `FMUClient` uses the `fmpy` library to execute the compiled Modelica FMU, extracting pack-level signals and per-cell temperatures (`pack.TCell[1,1]` through `[Ns,Np]`).
4. Results are reassembled into a 3D thermal matrix `[time_frame][Ns][Np]` and returned as JSON to the frontend.

### AI dual-model architecture
The AI component (`train_ai_model.py`) trains two models for thermal prediction:
- **RandomForest** — robust baseline regressor
- **PyTorch DNN** (`BatteryThermalDNN`) — 3-layer network for nonlinear transient thermal features

Trained artifacts: `models/weights/heat_ai_model.pkl` (joblib: RF + scalers), `models/weights/dnn_weights.pth` (PyTorch state_dict). The AI compensator is loaded at server startup as a fallback for when FMU outputs NaN.

### Data flow: Frontend ↔ Backend
The Streamlit dashboard does **not** import the simulation engine directly. It POSTs to `http://localhost:8000/api/v1/simulate/predict` and reconstructs DataFrames + KPI dicts from the JSON response. This keeps the heavy C++ FMU runtime isolated in the backend process.

### Aging model (NREL-based)
`AgingModel.calculate_step_loss()` computes SOH degradation as the sum of:
- **Calendar aging**: Arrhenius temperature dependence × SOC stress (exponential)
- **Cycle aging**: Ah-throughput × C-rate factor × low-temperature plating penalty

All parameters are calibratable class constants (e.g., `CAL_A`, `CYC_A`).

### Database
SQLite (`battery_app.db`, auto-created) with two tables:
- `users` — bcrypt-hashed passwords
- `history` — JSON-serialized simulation config + KPIs per user

## Modelica/FMU Modification Workflow

### Architecture: Library + System Models

项目采用**库 + 系统模型**分离架构：

| 文件 | 角色 |
|------|------|
| `SodiumIonBattery_backup.mo` | **模型库**（SodiumIonBattery 包），包含电芯、冷却、电池包、监控等可复用组件 |
| `SystemForFMI_8x2.mo` | **微型模组** — 8串2并 (16电芯)，~3.4kWh，适用两轮车/便携储能 |
| `SystemForFMI_16x4.mo` | **储能单簇** — 16串4并 (64电芯)，~13.6kWh，适用户用/工商业储能 |
| `SystemForFMI_96x1.mo` | **乘用车标准包** — 96串1并 (96电芯)，~14.9kWh，适用乘用车/400V平台 |

系统模型文件是**顶层模型**（`within ;`），各自固定了 Ns（串联数）和 Np（并联数）。
每个文件在 OMEdit 中打开时需先加载 `SodiumIonBattery_backup.mo` 库。

### FMU 导出步骤

1. 在 OpenModelica OMEdit 中先加载 `SodiumIonBattery_backup.mo`
2. 再打开目标系统模型文件（如 `SystemForFMI_96x1.mo`）
3. 检查语法无误后，导出 FMU（FMI 2.0 Co-Simulation）
4. 将生成的 `.fmu` 放入 `fmu_models/` 目录
5. 更新 `server.py` 中的 `FMU_PATH` 指向新 FMU
6. **同步**：确保 Python 端 `fmu_interface.py` 读取的 Ns/Np 与 FMU 一致，或从 `modelDescription.xml` 动态读取

### FMU 变量约定

所有系统模型统一使用以下组件命名，确保 Python 端兼容：
- `pack` — 电池包实例（`pack.V_pack`, `pack.TCell[s,j]` 等）
- `cooling` — 冷却板实例
- `controlled_load` — 外部可控电流源
- `I_load_external` — **唯一运行时输入**（A，正值放电）
- 变量名中 `pack.TCell[s,j]` 的索引格式因编译器而异（`[s,j]` 或 `[s, j]`），`_extract_spatial_matrix` 已兼容两种格式

## Important Constraints

- **Windows-only FMU runtime**: The `_setup_windows_env()` method in `FMUClient` hardcodes `D:\openmodelica\bin`. For Linux deployment, this block is a no-op but the FMU must be recompiled for the target OS.
- **Ns/Np 是编译时固定的结构参数**：FMU 导出后数组维度不可变，切换规格需重新导出不同的系统模型 FMU。
- **PDF reports are ASCII-only**: `report_generator.py` uses `fpdf` (latin-1), so Chinese characters are replaced with `?` via `safe_text()`.
- **Streamlit is stateful**: `st.session_state` holds `logged_in`, `username`, and `sim_result` (DataFrame, KPIs, config, thermal_matrix tuple). Stale session state after backend restart requires a browser refresh.
- The FastAPI `BatteryDigitalTwin` engine is initialized as a **module-level singleton** — it loads the FMU once at import time. Restart the backend after replacing the FMU file.
