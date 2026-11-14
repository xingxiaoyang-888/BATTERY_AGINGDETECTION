# tests/test_engine.py
import unittest
import numpy as np
from models.aging_algorithm import AgingModel
from models.physics_lib import BatteryPhysicsLibrary

class TestBatteryDigitalTwin(unittest.TestCase):
    """Test Battery Digital Twin."""

    def setUp(self):
        """Set Up."""
        self.test_temp_c = 25.0
        self.test_soc_pct = 50.0
        self.test_current_a = 100.0
        self.test_dt_s = 3600.0

    def test_aging_model_calendar_loss(self):
        """Test aging model calendar loss."""
        loss = AgingModel.calculate_step_loss(
            temp_c=self.test_temp_c,
            current_a=0.0,
            soc_pct=self.test_soc_pct,
            dt_seconds=self.test_dt_s,
            cell_capacity_ah=100.0
        )

        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 1e-4)

    def test_aging_model_cycle_stress(self):
        """Test aging model cycle stress."""

        loss_normal = AgingModel.calculate_step_loss(25.0, 100.0, 50.0, 3600, 100.0)

        loss_cold = AgingModel.calculate_step_loss(-10.0, -100.0, 50.0, 3600, 100.0)


        self.assertGreater(loss_cold, loss_normal * 1.5)

    def test_physics_ocv_bounds(self):
        """Test physics ocv bounds."""
        ocv_empty = BatteryPhysicsLibrary.get_high_fidelity_ocv(0.0, 25.0, "NCM811")
        ocv_full = BatteryPhysicsLibrary.get_high_fidelity_ocv(1.0, 25.0, "NCM811")

        self.assertGreater(ocv_full, ocv_empty)
        self.assertLess(ocv_empty, 3.5)
        self.assertGreater(ocv_full, 4.1)

    def test_entropic_heat_direction(self):
        """Test entropic heat direction."""

        q_rev_mid_soc = BatteryPhysicsLibrary.calculate_reversible_heat("NCM811", 0.5, 298.15, 50.0)
        self.assertLess(q_rev_mid_soc, 0.0)

if __name__ == '__main__':
    unittest.main()
