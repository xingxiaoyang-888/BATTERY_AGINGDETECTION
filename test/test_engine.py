# tests/test_engine.py
import unittest
import numpy as np
from models.aging_algorithm import AgingModel
from models.physics_lib import BatteryPhysicsLibrary

class TestBatteryDigitalTwin(unittest.TestCase):
    """
    数字孪生引擎自动化核心单元测试集
    用于持续集成 (CI) 管道，确保物理公式未被破坏
    """

    def setUp(self):
        """测试前置初始化"""
        self.test_temp_c = 25.0
        self.test_soc_pct = 50.0
        self.test_current_a = 100.0 # 1C 放电假设
        self.test_dt_s = 3600.0     # 运行 1 小时

    def test_aging_model_calendar_loss(self):
        """测试日历老化计算逻辑 (无电流状态)"""
        loss = AgingModel.calculate_step_loss(
            temp_c=self.test_temp_c,
            current_a=0.0, # 静置
            soc_pct=self.test_soc_pct,
            dt_seconds=self.test_dt_s,
            cell_capacity_ah=100.0
        )
        # 日历老化不应为 0，且应为一个极小的值
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 1e-4)

    def test_aging_model_cycle_stress(self):
        """测试循环老化高低温应力对比"""
        # 常温循环
        loss_normal = AgingModel.calculate_step_loss(25.0, 100.0, 50.0, 3600, 100.0)
        # 低温析锂循环 (充电方向为负)
        loss_cold = AgingModel.calculate_step_loss(-10.0, -100.0, 50.0, 3600, 100.0)
        
        # 低温大倍率充电的损伤应显著大于常温放电
        self.assertGreater(loss_cold, loss_normal * 1.5)

    def test_physics_ocv_bounds(self):
        """测试 OCV 物理库的边界合理性"""
        ocv_empty = BatteryPhysicsLibrary.get_high_fidelity_ocv(0.0, 25.0, "NCM811")
        ocv_full = BatteryPhysicsLibrary.get_high_fidelity_ocv(1.0, 25.0, "NCM811")
        
        self.assertGreater(ocv_full, ocv_empty)
        self.assertLess(ocv_empty, 3.5) # NCM 空电应低于 3.5V
        self.assertGreater(ocv_full, 4.1) # NCM 满电应高于 4.1V

    def test_entropic_heat_direction(self):
        """测试熵热在不同 SOC 下的吸放热方向"""
        # 放电且 dU/dT 为负时，熵热应为负 (吸热)
        q_rev_mid_soc = BatteryPhysicsLibrary.calculate_reversible_heat("NCM811", 0.5, 298.15, 50.0)
        self.assertLess(q_rev_mid_soc, 0.0)

if __name__ == '__main__':
    unittest.main()