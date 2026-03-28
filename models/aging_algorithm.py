# models/aging_algorithm.py
import numpy as np

class AgingModel:
    """
    [工业级] 锂电池全生命周期寿命预测模型 (NCM/LFP)
    基于 NREL (美国国家可再生能源实验室) 半经验衰减模型
    
    包含两大机制：
    1. 日历老化 (Calendar Aging): f(Time, Temp, SOC) -> SEI 膜生长
    2. 循环老化 (Cycle Aging): f(Ah-Throughput, C-rate, Temp) -> 机械疲劳与析锂
    """
    
    # === 标定参数 (需根据具体电芯 Lab 数据拟合) ===
    # 1. 日历老化参数 (参考 NCM811)
    CAL_A = 1.5e-4      # 基础衰减速率
    CAL_Ea = 24000      # 活化能 (J/mol)
    CAL_B = 0.05        # SOC 应力系数 (高 SOC 加速衰减)
    R_GAS = 8.314       # 理想气体常数
    
    # 2. 循环老化参数
    CYC_A = 3.0e-5      # 每 Ah 吞吐量的基础损耗
    CYC_C_Rate_Factor = 0.5  # 倍率敏感度 (大电流加速)
    
    @staticmethod
    def calculate_step_loss(temp_c, current_a, soc_pct, dt_seconds, cell_capacity_ah):
        """
        计算单步 (dt) 内的 SOH 损耗量 (无量纲, 0~1)
        """
        # --- A. 预处理 ---
        temp_k = temp_c + 273.15
        soc_norm = soc_pct / 100.0
        c_rate = abs(current_a) / cell_capacity_ah
        
        # --- B. 日历老化 (Calendar Aging) ---
        # 核心公式: Loss_cal ~ t^0.5 * exp(-Ea/RT) * exp(B * SOC)
        # 在微分形式下 (Rate): dLoss/dt ~ 0.5 * t^(-0.5) ... 
        # 为了简化实时计算，我们假设在当前时间步长内是线性的，但引入 SOC 指数惩罚
        
        # Arrhenius 项: 温度越高，衰减越快
        temp_stress = np.exp(-AgingModel.CAL_Ea / (AgingModel.R_GAS * temp_k))
        
        # SOC 项: 高电量(>80%)存放衰减剧增
        soc_stress = np.exp(AgingModel.CAL_B * soc_norm)
        
        # 单步日历损耗
        loss_calendar = AgingModel.CAL_A * temp_stress * soc_stress * (dt_seconds / 3600.0) # 归一化到小时
        
        # --- C. 循环老化 (Cycle Aging) ---
        # 核心机制: 只有当有电流通过时才产生
        if abs(current_a) > 0.01:
            # 机械应力因子: 倍率越大，颗粒破碎风险越高
            rate_stress = np.sqrt(c_rate) # 平方根关系较符合经验
            
            # 低温析锂因子: 低温大倍率充电极其伤电池
            plating_stress = 1.0
            if current_a < 0 and temp_c < 10: # 充电且低温
                plating_stress = 1.0 + 0.5 * (10 - temp_c) # 线性惩罚
            
            # 安时吞吐量 (Ah)
            ah_throughput = abs(current_a) * (dt_seconds / 3600.0)
            
            # 单步循环损耗
            loss_cycle = AgingModel.CYC_A * ah_throughput * rate_stress * plating_stress
        else:
            loss_cycle = 0.0
            
        # --- D. 总损耗 ---
        total_loss = loss_calendar + loss_cycle
        return total_loss