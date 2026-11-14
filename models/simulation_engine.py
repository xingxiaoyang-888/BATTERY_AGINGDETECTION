# models/simulation_engine.py
import numpy as np
import pandas as pd
import os
import logging
import joblib


from utils.fmu_interface import FMUClient
from models.aging_algorithm import AgingModel
from models.data_structures import SimulationConfig, KPIResult, TimeSeriesData
logger = logging.getLogger(__name__)

class BatteryDigitalTwin:
    """Sodium-ion FMU orchestration with optional thermal compensation and pack metrics."""
    def __init__(self, fmu_path: str, ai_model_path: str = None):

        if not os.path.exists(fmu_path):
            raise FileNotFoundError(f"找不到 FMU 模型文件: {fmu_path}")
        self.fmu_engine = FMUClient(fmu_path)


        self.ai_compensator = None
        if ai_model_path and os.path.exists(ai_model_path):
            try:
                self.ai_compensator = joblib.load(ai_model_path)
                logger.info(" 成功加载 AI 混合驱动权重")
            except Exception as e:
                logger.error(f" 加载 AI 模型失败: {e}")

    def run_profile(self, config: SimulationConfig):
        """Simulate the current profile, assemble spatial results, and calculate engineering metrics."""



        fmu_inputs = {'I_load_external': config.pack_current_a}
        initial_temp_c = (
            config.initial_cell_temp_c
            if config.initial_cell_temp_c is not None else config.env_temp_c
        )
        coolant_temp_c = (
            config.coolant_inlet_temp_c
            if config.coolant_inlet_temp_c is not None else config.env_temp_c
        )
        fmu_parameters = {
            'pack.cellData.SOC_start': config.init_soc / 100.0,
            'pack.cellData.T_start': initial_temp_c + 273.15,
            'pack.cellData.Q_nominal': config.cell_capacity_ah * 3600.0,
            'cooling.data.T_coolant_in': coolant_temp_c + 273.15,
            'cooling.data.m_flow': config.coolant_flow_kg_s,
            'cooling.data.UA_cell': config.cooling_ua_w_per_k,
            'pack.faultMode': config.fault_mode,
            'pack.faultSeriesIndex': config.fault_s_index,
            'pack.faultParallelIndex': config.fault_p_index,
            'pack.faultSeverity': config.fault_severity,
            'cooling.faultMode': config.fault_mode,
            'cooling.faultSeriesIndex': config.fault_s_index,
            'cooling.faultParallelIndex': config.fault_p_index,
            'cooling.faultSeverity': config.fault_severity,
        }


        df_raw, temp_matrix_3d, soc_matrix_3d, soh_matrix_3d = self.fmu_engine.run_simulation(
            stop_time=config.sim_duration_s,
            inputs=fmu_inputs,
            parameters=fmu_parameters,
            current_profile=config.current_profile,
        )

        if df_raw is None or df_raw.empty:
            raise RuntimeError("仿真解算异常：FMU 未返回有效数据")


        if df_raw.isnull().any().any():
            nan_cols = df_raw.columns[df_raw.isnull().any()].tolist()
            logger.warning(f"检测到 NaN 列: {nan_cols}，执行前向填充降级...")

            df_raw = df_raw.ffill().bfill()

            for col in df_raw.columns:
                if df_raw[col].isnull().any():
                    if 'T_' in col:
                        df_raw[col] = df_raw[col].fillna(config.env_temp_c + 273.15)
                    elif 'SOC' in col:
                        df_raw[col] = df_raw[col].fillna(config.init_soc / 100.0)
                    elif 'V_' in col:
                        df_raw[col] = df_raw[col].fillna(3.1 * config.series_num)
                    elif 'I_' in col:
                        df_raw[col] = df_raw[col].fillna(0.0)
                    else:
                        df_raw[col] = df_raw[col].fillna(0.0)
            logger.info("NaN 降级处理完成，仿真可继续。")

            for frames, fallback in [(temp_matrix_3d, config.env_temp_c),
                                      (soc_matrix_3d, config.init_soc / 100.0),
                                      (soh_matrix_3d, config.init_soh / 100.0)]:
                if frames:
                    for frame in frames:
                        for s in range(len(frame)):
                            for p in range(len(frame[s])):
                                if np.isnan(frame[s][p]):
                                    frame[s][p] = fallback


        timestamps = df_raw['time'].to_numpy(dtype=float)
        currents = df_raw['pack.I_pack'].to_numpy(dtype=float)
        voltages = df_raw['pack.V_pack'].to_numpy(dtype=float)
        dt = np.diff(timestamps)
        ah_throughput = float(np.sum(np.abs(currents[:-1]) * dt) / 3600.0) if len(dt) else 0.0
        energy_throughput_kwh = (
            float(np.sum(np.abs(voltages[:-1] * currents[:-1]) * dt) / 3.6e6)
            if len(dt) else 0.0
        )
        efc = ah_throughput / max(2.0 * config.cell_capacity_ah, 1e-9)


        if 'pack.SOH_min' in df_raw and np.isfinite(df_raw['pack.SOH_min']).all():
            step_loss = max(
                0.0,
                float(df_raw['pack.SOH_min'].iloc[0] - df_raw['pack.SOH_min'].iloc[-1]),
            )
        else:
            step_loss = AgingModel.calculate_step_loss(
                temp_c=float(df_raw['pack.T_max'].mean() - 273.15),
                current_a=float(np.mean(currents)),
                soc_pct=config.init_soc,
                dt_seconds=config.sim_duration_s,
                cell_capacity_ah=config.cell_capacity_ah,
            )
        final_soh = max(0.0, config.init_soh - step_loss * 100)


        final_v = df_raw['pack.V_pack'].iloc[-1]
        final_soc = df_raw['pack.SOC_min'].iloc[-1] * 100
        max_temp_c = df_raw['pack.T_max'].max() - 273.15
        delta_t_series = df_raw['pack.T_max'] - df_raw['pack.T_min']


        Ns_fmu = self.fmu_engine.Ns
        Np_fmu = self.fmu_engine.Np
        r_total = 0.003 * Ns_fmu / Np_fmu
        p_dch_peak, p_chg_peak = self._calculate_sop_logic(
            final_v, r_total, max_temp_c, final_soc / 100.0, Ns_fmu
        )


        kpis = KPIResult(
            final_soc=final_soc,
            final_soh=final_soh,
            soh_loss_ppm=step_loss * 1e6,
            max_temp_c=max_temp_c,
            avg_delta_t=float(delta_t_series.mean()),
            max_delta_t=float(delta_t_series.max()),
            ah_throughput=ah_throughput,
            equivalent_full_cycles=efc,
            energy_throughput_kwh=energy_throughput_kwh,
            max_discharge_power_kw=p_dch_peak / 1000.0,
            max_charge_power_kw=p_chg_peak / 1000.0
        )


        if max_temp_c > 50.0: kpis.diagnostic_warnings.append(" 严重热失控风险：电芯温度超标")
        if final_soc < 5.0: kpis.diagnostic_warnings.append(" 低电量预警：请及时充电")
        if config.init_soh < 99.999:
            kpis.diagnostic_warnings.append(" 初始SOH当前作为寿命偏置使用，尚未反馈到FMU内阻")


        ts_data = TimeSeriesData()
        for i in range(len(df_raw)):
            ts_data.add_step(
                t=df_raw['time'].iloc[i],
                v=df_raw['pack.V_pack'].iloc[i],
                i=df_raw['pack.I_pack'].iloc[i],
                t_max=df_raw['pack.T_max'].iloc[i] - 273.15,
                t_min=df_raw['pack.T_min'].iloc[i] - 273.15,
                soc=df_raw['pack.SOC_min'].iloc[i] * 100,
                temp_matrix=temp_matrix_3d[i],
                soc_matrix=soc_matrix_3d[i] if soc_matrix_3d else None,
                soh_matrix=soh_matrix_3d[i] if soh_matrix_3d else None,
            )

        return df_raw, kpis, ts_data

    def _calculate_sop_logic(self, v_pack, r_total, temp_c, soc, series_num):
        """Estimate power limits from the voltage window and temperature derating."""
        V_MAX = 3.95 * series_num
        V_MIN = 2.0 * series_num


        i_max_dch = max(0, (v_pack - V_MIN) / r_total)
        i_max_chg = max(0, (V_MAX - v_pack) / r_total)


        derating = 1.0
        if temp_c > 45: derating *= 0.5
        if temp_c < 0: derating *= 0.2
        if soc < 0.1: derating *= 0.3

        p_dch_peak = v_pack * i_max_dch * derating
        p_chg_peak = v_pack * i_max_chg * derating

        return p_dch_peak, p_chg_peak
