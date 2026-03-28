within BatterySystem;

model CoolingSystem "Thermal Management Subsystem"
  // === 参数 ===
  parameter Integer mode = 1 "1=Natural, 2=Air, 3=Liquid, 4=Heating";
  parameter Real T_env = 298.15 "Environment Temperature (K)";
  parameter Real A_surf = 0.04 "Heat Transfer Surface Area (m2)";
  
  // === 接口 ===
  input Real T_cell "Cell Temperature from Battery (K)";
  output Real Q_dissipated "Heat Power removed by cooling (W)";
  
  // === 内部变量 ===
  Real h_eff "Effective Convection Coefficient (W/m2K)";
  Real T_coolant "Coolant Temperature (K)";

equation
  // 1. 根据模式决定换热系数 (工业查表逻辑)
  if mode == 1 then
    h_eff = 5 + 0.1 * (T_cell - T_env); // 自然对流，温差越大效率略高
    T_coolant = T_env;
  elseif mode == 2 then
    h_eff = 30; // 强迫风冷
    T_coolant = T_env;
  elseif mode == 3 then
    h_eff = 200; // 液冷板 (极高效率)
    T_coolant = T_env - 5; // 假设冷却液比环境更冷
  elseif mode == 4 then
    h_eff = 50; // PTC 加热
    T_coolant = T_env + 30; // 加热模式下介质是热的
  else
    h_eff = 5;
    T_coolant = T_env;
  end if;

  // 2. 牛顿冷却定律
  // Q = h * Area * (T_obj - T_medium)
  // 如果是加热模式，Q 会变成负数（代表热量进入电池）
  Q_dissipated = h_eff * A_surf * (T_cell - T_coolant);

end CoolingSystem;