# 🔋 Battery Aging Detection — 钠离子电池数字孪生平台

基于 **Modelica FMU** + **AI 热补偿** 的钠离子电池数字孪生系统，支持多规格电池包仿真、老化预测与 3D 可视化。

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.x-green.svg)](https://fastapi.tiangolo.com)
[![Modelica](https://img.shields.io/badge/Modelica-FMI%202.0-orange.svg)](https://fmi-standard.org)

---

## 🏗️ 架构概览

```
浏览器 → Streamlit (main.py, :8501) ──HTTP──→ FastAPI (server.py, :8000) → FMUClient (fmpy) → SystemForFMI.fmu
                                                    ↓
                                            BatteryDigitalTwin
                                            ├── AgingModel (NREL 半经验模型)
                                            ├── BatteryPhysicsLibrary
                                            └── AI 补偿器 (RandomForest + PyTorch DNN)
```

| 目录 | 说明 |
|------|------|
| `views/` | Streamlit 前端页面 — 仪表盘、侧边栏、3D 空间视图、组件 |
| `models/` | 仿真引擎、数据结构、老化算法、物理库 |
| `utils/` | FMU 接口 (fmpy 封装)、SQLite 数据库、PDF 报告、CSS 加载器 |
| `config/` | 主题颜色、CSS 注入 |
| `test/` | 单元测试 (unittest/pytest) |
| `fmu_models/` | 编译后的 FMU 二进制文件 (FMI 2.0 Co-Simulation) |
| `assets/` | CSS 样式表、循环工况 CSV 模板 |

---

## 🚀 快速开始

### 环境要求

- **Windows** (FMU 运行时依赖 `D:\openmodelica\bin`)
- Python 3.10+
- OpenModelica（已安装于 `D:\openmodelica\`）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动应用

需要同时运行两个进程：

```bash
# 终端 1: 启动 FastAPI 后端 (端口 8000)
python server.py

# 终端 2: 启动 Streamlit 前端 (端口 8501)
streamlit run main.py
```

### 默认登录

- 用户名: `admin`
- 密码: `1234`

---

## 🧪 运行测试

```bash
# 运行所有单元测试
python -m pytest test/ -v

# 或使用 unittest
python -m unittest discover test/ -v

# 运行单个测试文件
python -m unittest test.test_engine -v
```

---

## 🔬 核心特性

### 1. FMU 联合仿真
基于 Modelica 编译的物理模型，通过 FMI 2.0 Co-Simulation 接口与 Python 联合仿真，支持多种电池包规格：

| 模型文件 | 规格 | 能量 | 适用场景 |
|----------|------|------|----------|
| `SystemForFMI_8x2.fmu` | 8串2并 (16电芯) | ~3.4kWh | 两轮车/便携储能 |
| `SystemForFMI_16x4.fmu` | 16串4并 (64电芯) | ~13.6kWh | 户用/工商业储能 |
| `SystemForFMI_96x1.fmu` | 96串1并 (96电芯) | ~14.9kWh | 乘用车/400V平台 |

### 2. AI 双模型热补偿

- **RandomForest** — 鲁棒基线回归器
- **PyTorch DNN** — 3层网络捕获非线性瞬态热特征

训练产物：`models/weights/heat_ai_model.pkl` (RF + 缩放器), `models/weights/dnn_weights.pth` (PyTorch state_dict)

### 3. NREL 半经验老化模型

SOH 退化计算包含：
- **日历老化**: Arrhenius 温度依赖 × SOC 应力（指数）
- **循环老化**: Ah 吞吐量 × C-rate 因子 × 低温析锂惩罚

### 4. 3D 热场可视化

实时渲染电池包空间温度分布矩阵 `[时间帧][Ns][Np]`，支持热失控故障注入仿真。

---

## 📁 项目结构

```
Battery_Desktop/
├── main.py                    # Streamlit 入口
├── server.py                  # FastAPI 后端入口
├── requirements.txt           # Python 依赖
├── CLAUDE.md                  # Claude Code 项目指南
├── SodiumIonBattery_backup.mo # Modelica 模型库
├── views/                     # 前端视图
│   ├── components.py          # 通用组件
│   ├── dashboard_view.py      # 仪表盘
│   ├── sidebar_view.py        # 侧边栏
│   └── spatial_view.py        # 3D 空间视图
├── models/                    # 仿真引擎
│   ├── simulation_engine.py   # 数字孪生核心
│   ├── data_structures.py     # 数据结构定义
│   └── weights/               # AI 模型权重
├── utils/                     # 工具模块
│   ├── fmu_interface.py       # FMU 接口封装
│   ├── database.py            # SQLite 数据库
│   └── report_generator.py    # PDF 报告生成
├── config/                    # 配置
├── test/                      # 测试用例
├── fmu_models/                # FMU 二进制文件
└── assets/                    # 静态资源
```

---

## ⚠️ 重要约束

- **仅限 Windows FMU 运行时**: `FMUClient._setup_windows_env()` 硬编码了 `D:\openmodelica\bin` 路径
- **Ns/Np 编译时固定**: FMU 导出后数组维度不可变，切换规格需重新导出
- **PDF 报告仅限 ASCII**: `report_generator.py` 使用 `fpdf` (latin-1)，中文字符会替换为 `?`
- **Streamlit 有状态**: `st.session_state` 持有登录态和仿真结果，后端重启后需刷新浏览器

---

## 📄 License

MIT License
