within ;
model SystemForFMI_16x3
  "中型低压模组 — 16串3并 (48电芯)，先并后串，适用于轻型商用车、48V/72V微混系统、户用储能"
  import Modelica.Units.SI;

  // === 结构参数 (编译时固定，决定FMU的数组维度) ===
  parameter Integer Ns = 16 "串联组数";
  parameter Integer Np = 3 "每组并联单体数";

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
    "16s3p 电池包"
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
<p><strong>SystemForFMI_16x3</strong> 是一个预配置的钠离子电池包系统模型，专用于 FMU 导出。</p>
<ul>
<li><b>拓扑</b>：先并后串 (SeriesParallel) — 3 个电芯并联成组，16 组串联</li>
<li><b>电芯数量</b>：48 (16×3)</li>
<li><b>额定电压</b>：约 49.6 V (16 × 3.1 V)</li>
<li><b>额定能量</b>：约 7.4 kWh (50 Ah × 3.1 V × 3P × 16S)</li>
<li><b>适用场景</b>：轻型商用车、48V/72V微混系统、户用储能</li>
<li><b>运行时输入</b>：<code>I_load_external</code> (A, 正值放电)</li>
<li><b>FMU输出关键变量</b>：<code>pack.V_pack</code>、<code>pack.TCell[s,j]</code>、<code>pack.SOCCell[s,j]</code> 等</li>
</ul>
<p>在 OpenModelica 中打开此文件，需先加载 <code>SodiumIonBattery_backup.mo</code> 库文件，然后导出 FMU。</p>
</html>"));
end SystemForFMI_16x3;
