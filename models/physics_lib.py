# models/physics_lib.py
import numpy as np

class BatteryPhysicsLibrary:
    """Legacy lithium-ion OCV and reversible-heat approximations."""



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


    THERMAL_PROPERTIES = {
        "Cell_Core_Specific_Heat": 900.0,
        "Cell_Density": 2500.0,
        "Al_Busbar_Conductivity": 237.0,
        "Cu_Busbar_Conductivity": 401.0,
        "Thermal_Pad_Conductivity": 3.0
    }

    @staticmethod
    def calculate_reversible_heat(chemistry: str, soc: float, temp_k: float, current_a: float) -> float:
        """Return reversible heat in watts: Q = I * T * dU/dT. SOC is a fraction; temperature is kelvin."""
        if chemistry not in BatteryPhysicsLibrary.ENTROPIC_COEFFICIENTS:
            return 0.0

        nodes = BatteryPhysicsLibrary.ENTROPIC_COEFFICIENTS[chemistry]["soc_nodes"]
        dU_dT_vals = BatteryPhysicsLibrary.ENTROPIC_COEFFICIENTS[chemistry]["dU_dT"]


        current_dU_dT = np.interp(soc, nodes, dU_dT_vals)


        q_rev = current_a * temp_k * current_dU_dT
        return q_rev

    @staticmethod
    def get_high_fidelity_ocv(soc: float, temp_c: float, chemistry: str = "NCM811") -> float:
        """Return the legacy empirical OCV estimate with temperature correction; no fidelity guarantee is implied."""

        if chemistry == "NCM811":
            base_ocv = 3.2 + (soc * 0.8) + 0.2 * (soc**0.5) - 0.05 * (soc**3) + 0.15 * np.exp(-20 * (1-soc))
        elif chemistry == "LFP":

            base_ocv = 2.8 + 0.4 * (soc**0.1) + 0.1 * np.exp(-30*(1-soc)) - 0.1 * np.exp(-30*soc)
        else:
            base_ocv = 3.6


        temp_diff = temp_c - 25.0
        temp_compensation = temp_diff * 0.0005

        return base_ocv + temp_compensation
