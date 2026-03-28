within BatterySystem;

model BatteryCell "Coupled Electro-Thermal Model with Active Cooling"
  // === 1. 物理参数 (BOM) ===
  parameter Real capacity_Ah = 100;
  parameter Real R_base = 0.003 "Base Internal Resistance";
  parameter Real mass = 0.9 "Cell Mass (kg)";
  parameter Real Cp = 1000 "Specific Heat Capacity (J/kg.K)";
  
  // === 2. 运行参数 (输入) ===
  parameter Real T_env_input = 298.15;
  parameter Real SOC_init = 0.9;
  parameter Integer cooling_mode = 1; // 由 Python 传入
  
  input Real I_load "Load Current (A)";

  // === 3. 子系统实例化 (关键：调用冷却系统) ===
  CoolingSystem tms(
    mode = cooling_mode,
    T_env = T_env_input,
    T_cell = T_cell
  );

  // === 4. 状态变量 ===
  Real SOC(start=SOC_init, fixed=true);
  Real V_terminal;
  Real V_ocv;
  Real R_dynamic;
  Real T_cell(start=T_env_input, fixed=true);
  Real heat_gen "Joule Heating (W)";

equation
  // --- A. 电气域 (Electro) ---
  der(SOC) = -I_load / (capacity_Ah * 3600);
  
  // 工业级 OCV 拟合 (NCM 曲线特征)
  V_ocv = 3.2 + 0.9*SOC + 0.35*exp(-5*(1-SOC)) - 0.05*exp(-10*SOC);
  
  // 动态内阻 (随温度变化：温度越低，内阻越大)
  // Arrhenius 修正: R = R_base * exp(Ea/kT)
  R_dynamic = R_base * (1 + 0.02 * (298.15 - T_cell));
  
  V_terminal = V_ocv - I_load * R_dynamic;

  // --- B. 热力域 (Thermal) ---
  // 产热
  heat_gen = I_load^2 * R_dynamic;
  
  // 热平衡方程: m*Cp*dT/dt = 产热 - 散热
  // tms.Q_dissipated 来自冷却子系统
  mass * Cp * der(T_cell) = heat_gen - tms.Q_dissipated;

end BatteryCell;