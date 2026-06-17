#!/usr/bin/env python3
"""
钠离子电池包 FMU 系统模型自动生成器

用法:
  python tools/generate_fmu_model.py --ns 96 --np 1
  python tools/generate_fmu_model.py --ns 192 --np 2 --label "800V平台"
  python tools/generate_fmu_model.py --batch          # 批量生成所有常用规格
  python tools/generate_fmu_model.py --list           # 仅列出将要生成的规格

输出文件命名:
  SystemForFMI_NSxNP.mo
"""

import os
import argparse

# ── 模板 (用 __TOKEN__ 避免与 Modelica {} 冲突) ──────
MODEL_TEMPLATE = r'''within ;
model SystemForFMI___NS__x__NP__
  "__DESCRIPTION__"
  import Modelica.Units.SI;

  // === 结构参数 (编译时固定，决定FMU的数组维度) ===
  parameter Integer Ns = __NS__ "串联组数";
  parameter Integer Np = __NP__ "每组并联单体数";

  // === 运行时输入 (FMU唯一输入) ===
  input SI.Current I_load_external "外部负载电流(A, 正值放电, 负值充电)"
    annotation(Dialog(group="External Control"));

  // === 组件实例 ===
  SodiumIonBattery.Packs.SeriesParallelPack pack(
    Ns=Ns,
    Np=Np,
    cellData(
      Q_nominal=50*3600,
      U_min=2.0,
      U_max=3.95,
      U_nominal=3.1,
      R0_ref=0.0008,
      C_th=1050,
      mass=1.05,
      specificEnergy=150,
      SOC_start=0.8,
      T_start=298.15),
    agingData(
      calendarFadePerYear_ref=0.02,
      cycleFadePerEFC_ref=3.2e-5),
    faultMode=SodiumIonBattery.Types.FaultMode.none,
    faultSeverity=0)
    "__NS__s__NP__p 电池包"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}})));

  SodiumIonBattery.Cooling.LiquidColdPlate1D cooling(
    Ns=Ns,
    Np=Np,
    data(
      T_coolant_in=298.15,
      m_flow=0.035,
      cp_coolant=3800,
      UA_cell=2.0))
    "液冷板"
    annotation (Placement(transformation(extent={{-10,-60},{10,-40}})));

  SodiumIonBattery.Sources.ExternalCurrent controlled_load
    "外部控制电流负载"
    annotation (Placement(transformation(extent={{-10,30},{10,50}})));

  SodiumIonBattery.Sources.Ground ground
    "电气参考地"
    annotation (Placement(transformation(extent={{40,-30},{60,-10}})));

  SodiumIonBattery.Monitors.PackStatusSummary monitor(
    V_pack=pack.V_pack,
    I_pack=pack.I_pack,
    T_max=pack.T_max,
    T_min=pack.T_min,
    SOC_min=pack.SOC_min,
    SOH_min=pack.SOH_min,
    platingRiskMax=pack.platingRiskMax,
    faultSeverityMax=pack.faultSeverityMax,
    faultCount=pack.faultCount)
    "状态监视器"
    annotation (Placement(transformation(extent={{50,-70},{70,-50}})));

equation
  // 将FMU外部输入信号接入电流负载
  controlled_load.i_ext = I_load_external;

  // 电气回路：负载 → 电池包 → 接地
  connect(controlled_load.p, pack.p);
  connect(controlled_load.n, pack.n);
  connect(pack.n, ground.p);

  // 热回路：电池包热端口 → 冷却板
  connect(pack.heatPort, cooling.cellPort);

  annotation (
    experiment(StartTime=0, StopTime=1800, Tolerance=1e-6, Interval=1),
    Documentation(info="<html>
<p><strong>SystemForFMI___NS__x__NP__</strong> 是一个预配置的钠离子电池包系统模型，专用于 FMU 导出。</p>
<ul>
<li><b>拓扑</b>：先并后串 (SeriesParallel) — __NP__ 个电芯并联成组，__NS__ 组串联</li>
<li><b>电芯数量</b>：__TOTAL_CELLS__ (__NS__×__NP__)</li>
<li><b>额定电压</b>：约 __NOMINAL_VOLTAGE__ V (__NS__ × 3.1 V)</li>
<li><b>额定能量</b>：约 __NOMINAL_ENERGY__ kWh (50 Ah × 3.1 V × __NP__P × __NS__S)</li>
<li><b>适用场景</b>：__USE_CASE__</li>
<li><b>运行时输入</b>：<code>I_load_external</code> (A, 正值放电)</li>
<li><b>FMU输出关键变量</b>：<code>pack.V_pack</code>、<code>pack.TCell[s,j]</code>、<code>pack.SOCCell[s,j]</code> 等</li>
</ul>
<p>在 OpenModelica 中打开此文件，需先加载 <code>SodiumIonBattery_backup.mo</code> 库文件，然后导出 FMU。</p>
</html>"));
end SystemForFMI___NS__x__NP__;
'''


def classify_use_case(ns: int, np: int, voltage: float, energy: float) -> str:
    """根据电压等级和电芯数自动推断适用场景"""
    total_cells = ns * np

    if voltage <= 48:
        if total_cells <= 20:
            return "微型储能模组、两轮电动车、便携式电源"
        elif total_cells <= 40:
            return "轻型电动车(LEV)、小型AGV、通信基站备电"
        else:
            return "低压储能、高尔夫球车、小型物流车"
    elif voltage <= 72:
        return "轻型商用车、48V/72V微混系统、户用储能"
    elif voltage <= 160:
        return "小型乘用车(150V平台)、户用储能、轻型物流车"
    elif voltage <= 350:
        return "乘用车动力电池(400V平台)、工商业储能"
    elif voltage <= 450:
        return "乘用车动力电池(400V+平台)、大型工商业储能"
    else:
        if total_cells <= 250:
            return "乘用车动力电池(800V高压平台)、快充车型"
        else:
            return "大型乘用车/SUV(800V平台)、重型商用车、电网级储能"


def make_description(ns: int, np: int, label: str, use_case: str) -> str:
    if label:
        return f"{label} — {ns}串{np}并 ({ns * np}电芯)，先并后串，适用于{use_case}"
    return f"{ns}串{np}并 ({ns * np}电芯)，先并后串，适用于{use_case}"


def generate_model(ns: int, np: int, label: str = None,
                   use_case: str = None, output_dir: str = ".") -> str:
    """生成单个系统模型 .mo 文件，返回文件路径"""
    total_cells = ns * np
    nominal_voltage = round(ns * 3.1, 1)
    nominal_energy = round(50.0 * 3.1 * np * ns / 1000.0, 1)

    if use_case is None:
        use_case = classify_use_case(ns, np, nominal_voltage, nominal_energy)

    description = make_description(ns, np, label, use_case)
    filename = f"SystemForFMI_{ns}x{np}.mo"
    filepath = os.path.join(output_dir, filename)

    content = MODEL_TEMPLATE
    content = content.replace("__NS__", str(ns))
    content = content.replace("__NP__", str(np))
    content = content.replace("__TOTAL_CELLS__", str(total_cells))
    content = content.replace("__NOMINAL_VOLTAGE__", str(nominal_voltage))
    content = content.replace("__NOMINAL_ENERGY__", str(nominal_energy))
    content = content.replace("__USE_CASE__", use_case)
    content = content.replace("__DESCRIPTION__", description)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return filepath


# ── 预定义常用规格 ──────────────────────────────────
COMMON_CONFIGS = [
    (ns, np, label, None)  # use_case=None 让脚本自动推断
    for ns, np, label in [
        # --- 微型/低压 (≤48V) ---
        (8,   2,  "微型模组"),
        (12,  2,  "小型模组"),
        (16,  1,  "低压串联模组"),
        (16,  3,  "中型低压模组"),
        (20,  2,  "低压储能模组"),

        # --- 中压储能/商用车 (48V~160V) ---
        (16,  4,  "储能单簇"),
        (24,  2,  "轻商用车模组"),
        (32,  2,  "中型储能模组"),
        (48,  1,  "小型乘用车"),
        (48,  2,  "小型乘用车高容"),

        # --- 400V 乘用车平台 (160V~350V) ---
        (96,  1,  "乘用车标准包"),
        (96,  2,  "乘用车大容量包"),
        (96,  3,  "乘用车高续航包"),
        (100, 1,  "400V标准平台"),
        (100, 2,  "400V大容量平台"),
        (108, 2,  "400V长续航平台"),
        # 钠离子电池 400V 等效（串联数更多以补偿单体低压）
        (128, 1,  "钠电400V标准包"),
        (128, 2,  "钠电400V大容量包"),

        # --- 800V 高压平台 (≥450V) ---
        (192, 1,  "800V标准平台"),
        (192, 2,  "800V高性能平台"),
        (200, 1,  "800V通用平台"),
        (216, 1,  "800V长续航平台"),
        # 钠离子电池高压等效
        (256, 1,  "钠电800V标准包"),
        (288, 1,  "钠电900V高压包"),

        # --- 电网级储能 (≥1000V) ---
        (480, 1,  "钠电1500V储能包"),
    ]
]


def batch_generate(output_dir: str = ".", dry_run: bool = False):
    """批量生成全部常用规格"""
    results = []
    for ns, np, label, use_case in COMMON_CONFIGS:
        params = {
            'ns': ns, 'np': np,
            'total': ns * np,
            'voltage': round(ns * 3.1, 1),
            'energy': round(50.0 * 3.1 * np * ns / 1000.0, 1),
            'label': label,
        }
        if dry_run:
            results.append(params)
        else:
            path = generate_model(ns, np, label, use_case, output_dir)
            results.append((path, params))
    return results


def print_list():
    """打印所有将要生成的规格表格"""
    print("将要生成的 FMU 模型文件:\n")
    print(f"{'规格':<16} {'电芯数':<8} {'额定电压':<10} {'额定能量':<10} {'适用场景'}")
    print("-" * 100)
    for ns, np, label, _ in COMMON_CONFIGS:
        total = ns * np
        voltage = round(ns * 3.1, 1)
        energy = round(50.0 * 3.1 * np * ns / 1000.0, 1)
        tag = f" [{label}]" if label else ""
        print(f"{ns:>3}s{np:>2}p{tag:<12} {total:<8} "
              f"{voltage:<8.1f}V {energy:<8.1f}kWh "
              f"{classify_use_case(ns, np, voltage, energy)}")


# ── CLI ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="钠离子电池包 FMU 系统模型生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/generate_fmu_model.py --ns 96 --np 1
  python tools/generate_fmu_model.py --ns 192 --np 2 --label "800V平台"
  python tools/generate_fmu_model.py --batch
  python tools/generate_fmu_model.py --list
        """,
    )
    parser.add_argument("--ns", type=int, help="串联数 (Ns)")
    parser.add_argument("--np", type=int, help="并联数 (Np)")
    parser.add_argument("--label", type=str, default=None, help="自定义标签")
    parser.add_argument("--output", "-o", type=str, default="mo_system_models", help="输出目录 (默认 mo_system_models/)")
    parser.add_argument("--batch", action="store_true", help="批量生成所有常用规格")
    parser.add_argument("--list", action="store_true", help="仅列出将要生成的规格，不实际写入")

    args = parser.parse_args()

    if args.list:
        print_list()
        return

    if args.batch:
        print(f"批量生成 {len(COMMON_CONFIGS)} 个系统模型文件...\n")
        results = batch_generate(args.output)
        for path, params in results:
            label = params.get('label', '')
            tag = f"  [{label}]" if label else ""
            print(f"  ✓ {os.path.basename(path)}{tag} "
                  f"({params['total']}电芯, {params['voltage']}V, {params['energy']}kWh)")
        print(f"\n共生成 {len(results)} 个文件 → {os.path.abspath(args.output)}")
        return

    if args.ns is None or args.np is None:
        parser.error("请指定 --ns 和 --np，或使用 --batch 批量生成")

    ns, np = args.ns, args.np
    if ns < 1 or np < 1:
        parser.error("Ns 和 Np 必须 >= 1")

    path = generate_model(ns, np, args.label, output_dir=args.output)
    cells = ns * np
    volt = round(ns * 3.1, 1)
    energy = round(50.0 * 3.1 * np * ns / 1000.0, 1)
    print(f"✓ 已生成: {path}")
    print(f"  规格: {ns}s{np}p ({cells}电芯)")
    print(f"  额定电压: ~{volt}V")
    print(f"  额定能量: ~{energy}kWh")


if __name__ == "__main__":
    main()
