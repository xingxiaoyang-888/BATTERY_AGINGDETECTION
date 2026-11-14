export type Cell = {
  c_rate: number;
  id: string;
  cycles: number;
  soh_pct: number;
  temperature_c: number;
  capacity_ah: number;
  observations: number;
  status: string;
};
export type Dataset = {
  id: string;
  name: string;
  chemistry: string;
  provenance: string;
  rows: number;
  cell_count: number;
  cells: Cell[];
};
export type Observation = { cycle_index: number; soh_pct: number };
export type Forecast = {
  id: string;
  cell_id: string;
  model: string;
  model_label: string;
  provenance: string;
  current_soh_pct: number;
  current_cycle: number;
  rul_cycles: number | null;
  rul_censored: boolean;
  horizon: number;
  eol_pct: number;
  rate_pct_per_100_cycles: number;
  history: Observation[];
  trajectory: {
    cycle_index: number;
    median: number;
    lower: number;
    upper: number;
  }[];
  uncertainty: { calibrated: boolean; kind: string; coverage: number | null };
  warnings: string[];
};
export type Model = {
  id: string;
  name: string;
  family: string;
  description: string;
  tags: string[];
  status: string;
  metrics: Record<string, number> | null;
  reason?: string;
};
export type Run = {
  id: string;
  kind: string;
  created_at: string;
  request: Record<string, unknown>;
};
export type SimulationInput = {
  series: number;
  parallel: number;
  capacity_ah: number;
  current_a: number;
  duration_s: number;
  ambient_c: number;
  initial_soc_pct: number;
  initial_soh_pct: number;
  cooling_w_k: number;
};
export type Simulation = {
  id: string;
  engine: string;
  label: string;
  series: number;
  parallel: number;
  trajectory: {
    time_s: number;
    voltage_v: number;
    temperature_c: number;
    soc_pct: number;
  }[];
  summary: {
    peak_temperature_c: number;
    final_soc_pct: number;
    final_voltage_v: number;
    cutoff_reached: boolean;
  };
  warnings: string[];
};

export type FmuSimulation = {
  id: string;
  payload: {
    summary: {
      soc: number;
      max_temp: number;
      energy_throughput_kwh: number;
    };
    time_series: {
      time: number[];
      t_max: number[];
      t_min: number[];
    };
    diagnostics: string;
  };
};
