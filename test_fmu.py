import os
import fmpy
import pandas as pd

# ==========================================
# 🎯【关键修复】告诉 Python 去哪里找底层的 C/C++ 运行库
# ==========================================
# 请确认你的 OpenModelica 是否安装在这个默认路径，如果不是，请修改
om_bin_path = r"D:\openmodelica\bin"

if os.path.exists(om_bin_path):
    # 将路径加入环境变量
    os.environ['PATH'] = om_bin_path + os.pathsep + os.environ.get('PATH', '')
    # Python 3.8+ 必须显式添加 DLL 目录
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(om_bin_path)
else:
    print(f"⚠️ 警告: 找不到 OpenModelica bin 目录 ({om_bin_path})，仿真可能崩溃！")
# ==========================================

# 你的 FMU 路径
fmu_path = r"F:\Battery_Desktop\fmu_models\SystemForFMI.fmu"

def run_ignition_test():
    print("🔍 步骤 1: 正在扫描 FMU 引擎内部结构...")
    fmpy.dump(fmu_path)
    print("-" * 50)

    print("⚡ 步骤 2: 正在向 FMU 注入 50A 负载电流，启动解算...")
    inputs = {
        'I_load_external': 50.0  # 注入 50A 放电电流
    }

    # 执行联合仿真 (跑 600 秒)
    result = fmpy.simulate_fmu(
        filename=fmu_path,
        start_time=0.0,
        stop_time=600.0,
        output_interval=1.0, 
        start_values=inputs,
        # 提取你想看的变量
        output=['time', 'pack.V_pack', 'pack.T_max'] 
    )

    print("✅ 步骤 3: 解算完成！转换数据格式...")
    df = pd.DataFrame(result)
    
    print("\n📊 仿真最后 5 秒的物理孪生结果：")
    print(df.tail())

if __name__ == '__main__':
    run_ignition_test()