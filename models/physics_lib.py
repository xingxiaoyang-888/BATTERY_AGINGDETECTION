# models/physics_lib.py
import numpy as np

class BatteryPhysicsLibrary:
    """
    动力电池核心物理化学参数与经验公式库
    提供不同材料体系 (NCM, LFP, LTO) 的基础标定数据
    """
    
    # 熵热系数 (Entropic Heat Coefficient) 查找表 (dV/dT)
    # 用于计算可逆热 (Reversible Heat)
    ENTROPIC_COEFFICIENTS = {
        "NCM811": {
            "soc_nodes": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "dU_dT": np.array([0.2, 0.15, 0.05, -0.02, -0.05, -0.08, -0.1, -0.05, 0.02, 0.1, 0.15]) * 1e-3
        },
        "LFP": {
            "soc_nodes": np.array([0.0, 0.2, 0.5, 0.8, 1.0]),
            "dU_dT": np.array([0.05, -0.01, -0.02, -0.01, 0.05]) * 1e-3
        }
    }

    # 传热学参数表 (W/m·K)
    THERMAL_PROPERTIES = {
        "Cell_Core_Specific_Heat": 900.0,       # 电芯核心比热容 (J/kg·K)
        "Cell_Density": 2500.0,                 # 电芯密度 (kg/m^3)
        "Al_Busbar_Conductivity": 237.0,        # 铝排导热系数
        "Cu_Busbar_Conductivity": 401.0,        # 铜排导热系数
        "Thermal_Pad_Conductivity": 3.0         # 导热结构胶导热系数
    }

    @staticmethod
    def calculate_reversible_heat(chemistry: str, soc: float, temp_k: float, current_a: float) -> float:
        """
        计算可逆熵热 (Q_rev = I * T * dU/dT)
        :param chemistry: 电池体系
        :param soc: 0~1 之间的荷电状态
        :param temp_k: 绝对温度 (K)
        :param current_a: 电流 (放电为正)
        :return: 产热功率 (W)
        """
        if chemistry not in BatteryPhysicsLibrary.ENTROPIC_COEFFICIENTS:
            return 0.0
            
        nodes = BatteryPhysicsLibrary.ENTROPIC_COEFFICIENTS[chemistry]["soc_nodes"]
        dU_dT_vals = BatteryPhysicsLibrary.ENTROPIC_COEFFICIENTS[chemistry]["dU_dT"]
        
        # 线性插值求解当前 SOC 下的熵热系数
        current_dU_dT = np.interp(soc, nodes, dU_dT_vals)
        
        # 熵热计算公式
        q_rev = current_a * temp_k * current_dU_dT
        return q_rev

    @staticmethod
    def get_high_fidelity_ocv(soc: float, temp_c: float, chemistry: str = "NCM811") -> float:
        """
        获取高保真开路电压 (OCV)
        考虑了温度对 OCV 曲线的微小偏移影响
        """
        # 基础 OCV (25度基准)
        if chemistry == "NCM811":
            base_ocv = 3.2 + (soc * 0.8) + 0.2 * (soc**0.5) - 0.05 * (soc**3) + 0.15 * np.exp(-20 * (1-soc))
        elif chemistry == "LFP":
            # 铁锂的电压平台特征
            base_ocv = 2.8 + 0.4 * (soc**0.1) + 0.1 * np.exp(-30*(1-soc)) - 0.1 * np.exp(-30*soc)
        else:
            base_ocv = 3.6 # 默认降级值
            
        # 温度补偿系数 (极化修正)
        temp_diff = temp_c - 25.0
        temp_compensation = temp_diff * 0.0005 
        
        return base_ocv + temp_compensation