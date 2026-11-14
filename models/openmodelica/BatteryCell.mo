
model BatteryCell "Simplified High-Fidelity Cell Model"

  parameter Real capacity_Ah = 100 "Cell Capacity";
  parameter Real R_internal = 0.003 "Ohmic Resistance";
  parameter Real C_thermal = 1000 "Thermal Capacitance (J/K)";
  parameter Real h_conv = 5 "Convection Coefficient";
  parameter Real T_env = 298.15 "Environment Temperature (K)";
  parameter Real SOC_init = 0.9 "Initial SOC (0-1)";


  input Real I_load;


  Real SOC(start=SOC_init, fixed=true);
  Real V_terminal;
  Real V_ocv;
  Real T_cell(start=T_env, fixed=true);
  Real heat_gen;
  Real V_p1(start=0, fixed=true);

equation

  der(SOC) = -I_load / (capacity_Ah * 3600);
  V_ocv = 3.0 + 0.9*SOC + 0.3*(SOC^0.5) - 0.1*(SOC^3);
  der(V_p1) = (I_load/5000) - V_p1/20;
  V_terminal = V_ocv - I_load * R_internal - V_p1;


  heat_gen = I_load^2 * R_internal + V_p1*I_load;
  der(T_cell) = (heat_gen - h_conv * (T_cell - T_env)) / C_thermal;
end BatteryCell;
