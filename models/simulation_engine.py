# models/simulation_engine.py
import numpy as np
import pandas as pd
import os
import logging
import joblib

# 导入底层驱动与数据协议 
from utils.fmu_interface import FMUClient
from models.aging_algorithm import AgingModel
from models.data_structures import SimulationConfig, KPIResult, TimeSeriesData
logger = logging.getLogger(__name__)

class BatteryDigitalTwin:
    """
    动力电池数字孪生引擎最终版 (FMI物理引擎 + AI补偿 + 寿命闭环)
    适配：钠离子电池体系 (Na-Ion)
    """
    def __init__(self, fmu_path: str, ai_model_path: str = None):
        # 1. 挂载 FMI 物理心脏 
        if not os.path.exists(fmu_path):
            raise FileNotFoundError(f"找不到 FMU 模型文件: {fmu_path}")
        self.fmu_engine = FMUClient(fmu_path)
        
        # 2. 挂载 AI 权重文件 (用于非线性发热补偿或物理引擎失效容灾)
        self.ai_compensator = None
        if ai_model_path and os.path.exists(ai_model_path):
            try:
                self.ai_compensator = joblib.load(ai_model_path)
                logger.info(" 成功加载 AI 混合驱动权重")
            except Exception as e:
                logger.error(f" 加载 AI 模型失败: {e}")

    def run_profile(self, config: SimulationConfig):
        """
        执行全流程解算：物理仿真 -> AI 修正 -> 寿命评估 -> KPI 聚合 
        """
        # --- A. 物理引擎解算 ---
        # 注入信号：FMU 唯一运行时输入是 I_load_external
        # 故障参数为编译时固定，需重新导出 FMU 才能改变
        fmu_inputs = {
            'I_load_external': config.pack_current_a,
        }
        
        # 执行联合仿真（Ns/Np 由 FMU 内部自动检测）
        df_raw, temp_matrix_3d, soc_matrix_3d, soh_matrix_3d = self.fmu_engine.run_simulation(
            stop_time=config.sim_duration_s,
            inputs=fmu_inputs,
        )
        
        if df_raw is None or df_raw.empty:
            raise RuntimeError("仿真解算异常：FMU 未返回有效数据")

        # --- B. NaN 数值保护与降级容灾 ---
        if df_raw.isnull().any().any():
            nan_cols = df_raw.columns[df_raw.isnull().any()].tolist()
            logger.warning(f"检测到 NaN 列: {nan_cols}，执行前向填充降级...")
            # 策略：用前一个有效值填充（forward fill），保持物理连续性
            df_raw = df_raw.ffill().bfill()
            # 二次检查：如果整个列全是 NaN，填入安全默认值
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
            # 相应的空间矩阵也需要做 NaN 处理
            for frames, fallback in [(temp_matrix_3d, config.env_temp_c),
                                      (soc_matrix_3d, config.init_soc / 100.0),
                                      (soh_matrix_3d, config.init_soh / 100.0)]:
                if frames:
                    for frame in frames:
                        for s in range(len(frame)):
                            for p in range(len(frame[s])):
                                if np.isnan(frame[s][p]):
                                    frame[s][p] = fallback

        # --- C. 寿命损耗计算 ---
        # 提取全过程平均参数进行累计损伤评估
        avg_temp_c = df_raw['pack.T_max'].mean() - 273.15
        avg_current = config.pack_current_a
        
        # 调用 NREL 衰减模型
        step_loss = AgingModel.calculate_step_loss(
            temp_c=avg_temp_c,
            current_a=avg_current,
            soc_pct=config.init_soc,
            dt_seconds=config.sim_duration_s,
            cell_capacity_ah=50.0 # 适配 SodiumIonBattery.mo 标定值
        )
        final_soh = max(0.0, config.init_soh - step_loss * 100)

        # --- D. SOP 功率边界计算 (保留 V1.0 的严谨性并适配钠电)  ---
        final_v = df_raw['pack.V_pack'].iloc[-1]
        final_soc = df_raw['pack.SOC_min'].iloc[-1] * 100
        max_temp_c = df_raw['pack.T_max'].max() - 273.15
        
        # 钠离子电池动态内阻估算（使用 FMU 内部实际维度）
        Ns_fmu = self.fmu_engine.Ns
        Np_fmu = self.fmu_engine.Np
        r_total = 0.003 * Ns_fmu / Np_fmu
        p_dch_peak, p_chg_peak = self._calculate_sop_logic(
            final_v, r_total, max_temp_c, final_soc / 100.0, Ns_fmu
        )

        # --- E. 结果装箱与时序封装 ---
        kpis = KPIResult(
            final_soc=final_soc,
            final_soh=final_soh,
            soh_loss_ppm=step_loss * 1e6,
            max_temp_c=max_temp_c,
            avg_delta_t=max_temp_c - config.env_temp_c,
            max_discharge_power_kw=p_dch_peak / 1000.0,
            max_charge_power_kw=p_chg_peak / 1000.0
        )
        
        # 触发告警状态机
        if max_temp_c > 50.0: kpis.diagnostic_warnings.append(" 严重热失控风险：电芯温度超标")
        if final_soc < 5.0: kpis.diagnostic_warnings.append(" 低电量预警：请及时充电")

        # 封装时序对象
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
        """
        SOP (State of Power) 算法：基于电压窗口与温度降额的功率预测
        适配钠离子电池窗口：2.0V ~ 3.95V 
        """
        V_MAX = 3.95 * series_num
        V_MIN = 2.0 * series_num
        
        # 1. 基础物理极限电流
        i_max_dch = max(0, (v_pack - V_MIN) / r_total)
        i_max_chg = max(0, (V_MAX - v_pack) / r_total)
        
        # 2. 动态降额因子 (Derating)
        derating = 1.0
        if temp_c > 45: derating *= 0.5    # 高温限功率保护
        if temp_c < 0: derating *= 0.2     # 低温内阻剧增保护
        if soc < 0.1: derating *= 0.3      # 低 SOC 防止过放
        
        p_dch_peak = v_pack * i_max_dch * derating
        p_chg_peak = v_pack * i_max_chg * derating
        
        return p_dch_peak, p_chg_peak