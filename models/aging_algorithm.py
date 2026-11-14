# models/aging_algorithm.py
import numpy as np

class AgingModel:
    """Uncalibrated lithium-ion calendar/cycle aging reference. Constants are not validated for sodium-ion cells."""



    CAL_A = 1.5e-4
    CAL_Ea = 24000
    CAL_B = 0.05
    R_GAS = 8.314


    CYC_A = 3.0e-5
    CYC_C_Rate_Factor = 0.5

    @staticmethod
    def calculate_step_loss(temp_c, current_a, soc_pct, dt_seconds, cell_capacity_ah):
        """Return dimensionless SOH loss over one time step; positive current denotes discharge."""

        temp_k = temp_c + 273.15
        soc_norm = soc_pct / 100.0
        c_rate = abs(current_a) / cell_capacity_ah







        temp_stress = np.exp(-AgingModel.CAL_Ea / (AgingModel.R_GAS * temp_k))


        soc_stress = np.exp(AgingModel.CAL_B * soc_norm)


        loss_calendar = AgingModel.CAL_A * temp_stress * soc_stress * (dt_seconds / 3600.0)



        if abs(current_a) > 0.01:

            rate_stress = np.sqrt(c_rate)


            plating_stress = 1.0
            if current_a < 0 and temp_c < 10:
                plating_stress = 1.0 + 0.5 * (10 - temp_c)


            ah_throughput = abs(current_a) * (dt_seconds / 3600.0)


            loss_cycle = AgingModel.CYC_A * ah_throughput * rate_stress * plating_stress
        else:
            loss_cycle = 0.0


        total_loss = loss_calendar + loss_cycle
        return total_loss
