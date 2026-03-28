from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class BatteryBOM:
    """
    动力电池物料清单 (Bill of Materials) 数据结构
    定义了电池包的物理拓扑与电芯基础化学参数
    """
    chemistry: str = "NCM811"         # 电池化学体系
    cell_capacity_ah: float = 100.0   # 单体容量 (Ah)
    series_num: int = 96              # 串联数量
    parallel_num: int = 1             # 并联数量
    base_resistance_ohm: float = 0.003 # 标称交流内阻
    weight_kg: float = 2.1            # 单体重量估算
    
    @property
    def pack_voltage(self) -> float:
        """计算 Pack 级额定电压"""
        return self.series_num * 3.7
        
    @property
    def pack_energy_kwh(self) -> float:
        """计算 Pack 级总能量 (kWh)"""
        return (self.pack_voltage * self.cell_capacity_ah * self.parallel_num) / 1000.0

@dataclass
class SimulationConfig:
    """
    仿真引擎全局配置项
    用于规范前端传递到后端的各类边界条件
    """
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    profile_mode: str = "CC"          # 工况模式: CC (恒流) 或 CSV
    sim_duration_s: float = 600.0     # 仿真总时长
    pack_current_a: float = 80.0      # 总线电流
    env_temp_c: float = 25.0          # 环境温度
    cooling_type: str = "Liquid Cooling" # 热管理策略
    init_soc: float = 90.0            # 初始荷电状态
    init_soh: float = 100.0           # 初始健康状态
    
    # 高级求解器参数 (可作为预留接口增加代码量)
    solver_method: str = "dassl"
    tolerance: float = 1e-6
    step_size: float = 0.1

@dataclass
class KPIResult:
    """
    核心关键绩效指标 (Key Performance Indicators) 结果集
    """
    final_soc: float = 0.0
    final_soh: float = 0.0
    soh_loss_ppm: float = 0.0
    max_temp_c: float = 0.0
    avg_delta_t: float = 0.0
    max_discharge_power_kw: float = 0.0
    max_charge_power_kw: float = 0.0
    diagnostic_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典以便存入数据库"""
        return {
            "soc": round(self.final_soc, 2),
            "soh": round(self.final_soh, 4),
            "loss_ppm": round(self.soh_loss_ppm, 2),
            "max_temp": round(self.max_temp_c, 1),
            "delta_t": round(self.avg_delta_t, 1),
            "sop_dch": round(self.max_discharge_power_kw, 1),
            "warnings": "|".join(self.diagnostic_warnings)
        }

@dataclass
class TimeSeriesData:
    """
    时序数据容器，用于图表渲染和 CSV 导出
    """
    timestamps: List[float] = field(default_factory=list)
    voltages: List[float] = field(default_factory=list)
    currents: List[float] = field(default_factory=list)
    temperatures_max: List[float] = field(default_factory=list)
    temperatures_min: List[float] = field(default_factory=list)
    soc_array: List[float] = field(default_factory=list)
    
    def add_step(self, t, v, i, t_max, t_min, soc):
        self.timestamps.append(t)
        self.voltages.append(v)
        self.currents.append(i)
        self.temperatures_max.append(t_max)
        self.temperatures_min.append(t_min)
        self.soc_array.append(soc)
