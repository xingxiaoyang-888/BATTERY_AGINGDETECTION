import { useEffect, useState } from "react";
import {
  ArrowDownToLine,
  Battery,
  Gauge,
  Loader2,
  Thermometer,
  Zap,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type { Work } from "../api";
import { Badge, Metric, Notes, PackGraphic, Panel } from "../components";
import type { FmuSimulation, Simulation, SimulationInput } from "../types";

export default function TwinStudio({
  busy,
  work,
  onComplete,
  exportRun,
}: {
  busy: boolean;
  work: Work;
  onComplete: () => Promise<void>;
  exportRun: (id: string) => void;
}) {
  const [settings, setSettings] = useState<SimulationInput>({
    series: 16,
    parallel: 4,
    capacity_ah: 50,
    current_a: 40,
    duration_s: 1800,
    ambient_c: 25,
    initial_soc_pct: 80,
    initial_soh_pct: 95,
    cooling_w_k: 2,
  });
  const [result, setResult] = useState<Simulation | null>(null);
  const [fmuResult, setFmuResult] = useState<FmuSimulation | null>(null);
  const [engine, setEngine] = useState("lumped-rc");
  const [configs, setConfigs] = useState<
    { series: number; parallel: number; platforms: string[] }[]
  >([]);
  const [dirty, setDirty] = useState(false);
  const [configError, setConfigError] = useState("");
  useEffect(() => {
    api<typeof configs>("/fmu/configurations")
      .then(setConfigs)
      .catch((e) => setConfigError(e.message));
  }, []);
  const fields: {
    key: keyof SimulationInput;
    label: string;
    min: number;
    max: number;
    step?: number;
  }[] = [
    {
      key: "capacity_ah",
      label: "Capacity (Ah)",
      min: 0.1,
      max: 500,
      step: 0.1,
    },
    { key: "current_a", label: "Pack current (A)", min: -500, max: 500 },
    { key: "duration_s", label: "Duration (s)", min: 10, max: 7200 },
    { key: "ambient_c", label: "Ambient (°C)", min: -20, max: 60 },
    { key: "initial_soc_pct", label: "Initial SOC (%)", min: 5, max: 95 },
    { key: "initial_soh_pct", label: "Initial SOH (%)", min: 50, max: 100 },
    {
      key: "cooling_w_k",
      label: "Cooling (W/K)",
      min: 0.1,
      max: 20,
      step: 0.1,
    },
  ];
  async function run() {
    await work(async () => {
      if (engine === "fmu") {
        const response = await api<FmuSimulation>("/simulations/fmu", settings);
        setFmuResult(response);
        setResult(null);
      } else {
        const response = await api<Simulation>("/simulations", settings);
        setResult(response);
        setFmuResult(null);
      }
      setDirty(false);
      await onComplete();
    });
  }
  return (
    <div className="studio-grid">
      <Panel
        title="Configure your pack"
        subtitle="Positive current means discharge."
        className="settings-panel"
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            run();
          }}
          onChange={() => setDirty(true)}
        >
          <label>
            Solver
            <select value={engine} onChange={(e) => setEngine(e.target.value)}>
              <option value="lumped-rc">Equivalent circuit · portable</option>
              <option value="fmu">Modelica / FMU · native runtime</option>
            </select>
          </label>
          <label>
            Pack topology
            <select
              value={`${settings.series}x${settings.parallel}`}
              onChange={(e) => {
                const [series, parallel] = e.target.value
                  .split("x")
                  .map(Number);
                setSettings({ ...settings, series, parallel });
              }}
            >
              {(configs.length
                ? configs
                : [
                    { series: 16, parallel: 4 },
                    { series: 8, parallel: 2 },
                  ]
              ).map((c) => (
                <option
                  key={`${c.series}x${c.parallel}`}
                  value={`${c.series}x${c.parallel}`}
                >
                  {c.series}S × {c.parallel}P · {c.series * c.parallel} cells
                </option>
              ))}
            </select>
          </label>
          <div className="field-pair">
            {fields.map((f) => (
              <label key={f.key}>
                {f.label}
                <input
                  type="number"
                  required
                  min={f.min}
                  max={f.max}
                  step={f.step || 1}
                  value={settings[f.key]}
                  onChange={(e) =>
                    setSettings({ ...settings, [f.key]: +e.target.value })
                  }
                />
              </label>
            ))}
          </div>
          <button className="button primary wide" disabled={busy}>
            {busy ? <Loader2 className="spin" size={16} /> : <Zap size={16} />}
            Run simulation
          </button>
        </form>
        {configError && <Notes notes={[configError]} />}
      </Panel>
      <div>
        <Panel
          title="Module perspective"
          subtitle={`${result?.series || settings.series} series · ${result?.parallel || settings.parallel} parallel`}
          action={
            <Badge tone="neutral">
              {engine === "fmu" ? "Native solver" : "Lumped model"}
            </Badge>
          }
        >
          <PackGraphic
            temperature={
              result?.summary.peak_temperature_c ??
              fmuResult?.payload.summary.max_temp ??
              settings.ambient_c
            }
          />
          <div className="pack-caption">
            <span>
              <i />
              Schematic · 24 representative cells
            </span>
            <span>
              {engine === "fmu"
                ? "Illustrative layout"
                : "Uniform cells assumed"}
            </span>
          </div>
          {dirty && (result || fmuResult) && (
            <Notes
              notes={[
                "Settings changed. Run again to update the displayed result.",
              ]}
            />
          )}
        </Panel>
        {result && (
          <>
            <div className="metrics three">
              <Metric
                label="Peak temperature"
                value={result.summary.peak_temperature_c.toFixed(1)}
                unit="°C"
                note="Across simulated duration"
                icon={<Thermometer size={16} />}
              />
              <Metric
                label="Final state of charge"
                value={result.summary.final_soc_pct.toFixed(1)}
                unit="%"
                note={
                  result.summary.cutoff_reached
                    ? "Protection cutoff reached"
                    : "Within SOC limits"
                }
                icon={<Battery size={16} />}
              />
              <Metric
                label="Terminal voltage"
                value={result.summary.final_voltage_v.toFixed(1)}
                unit="V"
                note="At final simulated step"
                icon={<Gauge size={16} />}
              />
            </div>
            <Panel
              title="Thermal response"
              action={
                <button
                  className="text-button"
                  onClick={() => exportRun(result.id)}
                >
                  Export run
                  <ArrowDownToLine size={15} />
                </button>
              }
            >
              <div className="chart-wrap short">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={result.trajectory}>
                    <CartesianGrid vertical={false} stroke="#edf0ec" />
                    <XAxis
                      dataKey="time_s"
                      tickLine={false}
                      axisLine={false}
                      fontSize={11}
                    />
                    <YAxis
                      domain={["dataMin - 1", "dataMax + 1"]}
                      tickFormatter={(value: number) => value.toFixed(1)}
                      tickLine={false}
                      axisLine={false}
                      fontSize={11}
                    />
                    <Tooltip />
                    <Line
                      dataKey="temperature_c"
                      name="Temperature (°C)"
                      stroke="#377b5f"
                      strokeWidth={2.5}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Panel>
            <Notes notes={result.warnings} />
          </>
        )}
        {fmuResult && (
          <>
            <div className="metrics three">
              <Metric
                label="Peak temperature"
                value={fmuResult.payload.summary.max_temp.toFixed(1)}
                unit="°C"
                note="Native Modelica result"
                icon={<Thermometer size={16} />}
              />
              <Metric
                label="Final state of charge"
                value={fmuResult.payload.summary.soc.toFixed(2)}
                unit="%"
                note="At final simulated step"
                icon={<Battery size={16} />}
              />
              <Metric
                label="Energy throughput"
                value={(
                  fmuResult.payload.summary.energy_throughput_kwh * 1000
                ).toFixed(2)}
                unit="Wh"
                note="Across simulated duration"
                icon={<Zap size={16} />}
              />
            </div>
            <Panel
              title="Native thermal response"
              subtitle="Temperature envelope across the simulated pack."
              action={
                <button
                  className="text-button"
                  onClick={() => exportRun(fmuResult.id)}
                >
                  Export complete JSON <ArrowDownToLine size={15} />
                </button>
              }
            >
              <div className="chart-wrap short">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={fmuResult.payload.time_series.time.map((time, i) => ({
                      time_s: time,
                      maximum: fmuResult.payload.time_series.t_max[i],
                      minimum: fmuResult.payload.time_series.t_min[i],
                    }))}
                  >
                    <CartesianGrid vertical={false} stroke="#edf0ec" />
                    <XAxis
                      dataKey="time_s"
                      tickLine={false}
                      axisLine={false}
                      fontSize={11}
                    />
                    <YAxis
                      domain={["dataMin - 1", "dataMax + 1"]}
                      tickFormatter={(value: number) => value.toFixed(1)}
                      tickLine={false}
                      axisLine={false}
                      fontSize={11}
                    />
                    <Tooltip />
                    <Line
                      dataKey="maximum"
                      name="Maximum (°C)"
                      stroke="#377b5f"
                      strokeWidth={2.5}
                      dot={false}
                    />
                    <Line
                      dataKey="minimum"
                      name="Minimum (°C)"
                      stroke="#92a77d"
                      strokeDasharray="5 4"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {fmuResult.payload.diagnostics && (
                <details className="solver-diagnostics">
                  <summary>Original solver diagnostics</summary>
                  <p>{fmuResult.payload.diagnostics}</p>
                </details>
              )}
            </Panel>
            <Notes
              notes={[
                "This result uses the original Modelica parameterization. Initial SOH adjusts the aging calculation but is not coupled to FMU electrical resistance.",
              ]}
            />
          </>
        )}
        {!result && !fmuResult && (
          <Notes
            notes={[
              "The portable solver uses an uncalibrated equivalent circuit and lumped heat balance. It does not resolve spatial thermal gradients.",
              "The FMU solver loads the original Modelica binaries. Compatible platform binaries and runtime libraries must be installed.",
            ]}
          />
        )}
      </div>
    </div>
  );
}
