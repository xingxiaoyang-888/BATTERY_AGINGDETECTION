import os
from OMPython import OMCSessionZMQ

# 1. 定义最简化的物理模型 (去掉 Package 包裹，防止命名空间错误)
mo_content = """
model BatteryCell "Simplified High-Fidelity Cell Model"
  // 参数定义
  parameter Real capacity_Ah = 100 "Cell Capacity";
  parameter Real R_internal = 0.003 "Ohmic Resistance";
  parameter Real C_thermal = 1000 "Thermal Capacitance (J/K)";
  parameter Real h_conv = 5 "Convection Coefficient";
  parameter Real T_env = 298.15 "Environment Temperature (K)";
  parameter Real SOC_init = 0.9 "Initial SOC (0-1)";

  // 输入变量
  input Real I_load;

  // 状态变量
  Real SOC(start=SOC_init, fixed=true);
  Real V_terminal;
  Real V_ocv;
  Real T_cell(start=T_env, fixed=true);
  Real heat_gen;
  Real V_p1(start=0, fixed=true);

equation
  // 电气方程
  der(SOC) = -I_load / (capacity_Ah * 3600);
  V_ocv = 3.0 + 0.9*SOC + 0.3*(SOC^0.5) - 0.1*(SOC^3);
  der(V_p1) = (I_load/5000) - V_p1/20; 
  V_terminal = V_ocv - I_load * R_internal - V_p1;

  // 热力学方程
  heat_gen = I_load^2 * R_internal + V_p1*I_load;
  der(T_cell) = (heat_gen - h_conv * (T_cell - T_env)) / C_thermal;
end BatteryCell;
"""

# 2. 写入文件
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "models", "openmodelica")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

file_path = os.path.join(model_dir, "BatteryCell.mo") # 注意：改名为 BatteryCell.mo

try:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(mo_content)
    print(f"✅ 模型文件已自动重写至: {file_path}")
except Exception as e:
    print(f"❌ 文件写入失败: {e}")
    exit()

# 3. 立即尝试编译验证
print("⚙️ 正在调用 OpenModelica 进行编译测试...")
try:
    omc = OMCSessionZMQ()
    # 加载
    path_fixed = file_path.replace("\\", "/")
    res_load = omc.sendExpression(f'loadFile("{path_fixed}")')
    if not res_load:
        print(f"❌ OpenModelica 加载文件失败! 错误信息: {omc.sendExpression('getErrorString()')}")
        exit()
        
    # 检查类是否存在
    classes = omc.sendExpression("getClassNames()")
    print(f"ℹ️ 已加载的类: {classes}")
    
    if "BatteryCell" in classes:
        print("✅ 验证成功！BatteryCell 模型已就绪。")
        print("👉 请继续执行下一步：更新 simulation_engine.py")
    else:
        print("❌ 验证失败：BatteryCell 未出现在类列表中。")

except Exception as e:
    print(f"❌ 编译环境连接错误: {e}")