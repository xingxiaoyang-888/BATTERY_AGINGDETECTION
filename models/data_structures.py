from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np

@dataclass
class BatteryBOM:
    """Legacy lithium-ion bill of materials and fixed pack topology."""
    chemistry: str = "NCM811"
    cell_capacity_ah: float = 100.0
    series_num: int = 96
    parallel_num: int = 1
    base_resistance_ohm: float = 0.003
    weight_kg: float = 2.1

    @property
    def pack_voltage(self) -> float:
        """Pack voltage."""
        return self.series_num * 3.7

    @property
    def pack_energy_kwh(self) -> float:
        """Pack energy kwh."""
        return (self.pack_voltage * self.cell_capacity_ah * self.parallel_num) / 1000.0

@dataclass
class SimulationConfig:
    """Boundary conditions and fault settings passed to the native solver."""
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    profile_mode: str = "CC"
    sim_duration_s: float = 600.0
    pack_current_a: float = 80.0
    current_profile: Optional[List[Tuple[float, float]]] = None  # [(time_s, current_a), ...]
    env_temp_c: float = 25.0
    initial_cell_temp_c: Optional[float] = None
    cooling_type: str = "liquid"
    coolant_inlet_temp_c: Optional[float] = None
    coolant_flow_kg_s: float = 0.035
    cooling_ua_w_per_k: float = 2.0
    init_soc: float = 90.0
    init_soh: float = 100.0
    cell_capacity_ah: float = 50.0


    series_num: int = 8
    parallel_num: int = 2
    fault_mode: int = 1
    fault_s_index: int = 3
    fault_p_index: int = 1
    fault_severity: float = 0.8


    solver_method: str = "dassl"
    tolerance: float = 1e-6
    step_size: float = 0.1

@dataclass
class KPIResult:
    """Pack-level simulation metrics with explicit physical units."""
    final_soc: float = 0.0
    final_soh: float = 0.0
    soh_loss_ppm: float = 0.0
    max_temp_c: float = 0.0
    avg_delta_t: float = 0.0
    max_delta_t: float = 0.0
    ah_throughput: float = 0.0
    equivalent_full_cycles: float = 0.0
    energy_throughput_kwh: float = 0.0
    max_discharge_power_kw: float = 0.0
    max_charge_power_kw: float = 0.0
    diagnostic_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "soc": round(self.final_soc, 2),
            "soh": round(self.final_soh, 4),
            "loss_ppm": round(self.soh_loss_ppm, 2),
            "max_temp": round(self.max_temp_c, 1),
            "delta_t": round(self.max_delta_t, 2),
            "avg_delta_t": round(self.avg_delta_t, 2),
            "ah_throughput": round(self.ah_throughput, 4),
            "efc": round(self.equivalent_full_cycles, 6),
            "energy_throughput_kwh": round(self.energy_throughput_kwh, 6),
            "sop_dch": round(self.max_discharge_power_kw, 1),
            "warnings": "|".join(self.diagnostic_warnings)
        }

@dataclass
class TimeSeriesData:
    """Time-series and cell-matrix containers for visualization and export."""
    timestamps: List[float] = field(default_factory=list)
    voltages: List[float] = field(default_factory=list)
    currents: List[float] = field(default_factory=list)
    temperatures_max: List[float] = field(default_factory=list)
    temperatures_min: List[float] = field(default_factory=list)
    soc_array: List[float] = field(default_factory=list)


    temp_matrix_frames: List[List[List[float]]] = field(default_factory=list)
    soc_matrix_frames:  List[List[List[float]]] = field(default_factory=list)
    soh_matrix_frames:  List[List[List[float]]] = field(default_factory=list)

    def add_step(self, t: float, v: float, i: float, t_max: float, t_min: float, soc: float,
                 temp_matrix: Optional[List[List[float]]] = None,
                 soc_matrix: Optional[List[List[float]]] = None,
                 soh_matrix: Optional[List[List[float]]] = None):
        self.timestamps.append(t)
        self.voltages.append(v)
        self.currents.append(i)
        self.temperatures_max.append(t_max)
        self.temperatures_min.append(t_min)
        self.soc_array.append(soc)

        if temp_matrix is not None:
            self.temp_matrix_frames.append(temp_matrix)
        if soc_matrix is not None:
            self.soc_matrix_frames.append(soc_matrix)
        if soh_matrix is not None:
            self.soh_matrix_frames.append(soh_matrix)
