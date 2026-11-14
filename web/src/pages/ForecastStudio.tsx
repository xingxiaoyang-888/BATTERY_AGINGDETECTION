import { useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  FlaskConical,
  Loader2,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { Badge, HealthChart, Notes, Panel } from "../components";
import type { Cell, Dataset, Forecast, Model, Observation } from "../types";

export default function ForecastStudio({
  dataset,
  cell,
  models,
  history,
  forecast,
  busy,
  run,
  exportRun,
}: {
  dataset: Dataset;
  cell: Cell;
  models: Model[];
  history: Observation[];
  forecast: Forecast | null;
  busy: boolean;
  run: (body: unknown) => Promise<boolean>;
  exportRun: (id: string) => void;
}) {
  const [horizon, setHorizon] = useState(600);
  const [temperature, setTemperature] = useState(cell.temperature_c);
  const [rate, setRate] = useState(cell.c_rate);
  const [eol, setEol] = useState(80);
  const [model, setModel] = useState("robust-trend");
  const [dirty, setDirty] = useState(false);
  return (
    <div className="studio-grid">
      <Panel
        title="Set your scenario"
        subtitle="Make the assumptions explicit."
        className="settings-panel"
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            run({
              dataset_id: dataset.id,
              cell_id: cell.id,
              horizon,
              temperature_c: temperature,
              c_rate: rate,
              eol_pct: eol,
              model,
            }).then((success) => {
              if (success) setDirty(false);
            });
          }}
          onChange={() => setDirty(true)}
        >
          <label>
            Forecast model
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => (
                <option
                  key={m.id}
                  value={m.id}
                  disabled={!["available", "validated"].includes(m.status)}
                >
                  {m.id === "robust-trend" ? "Robust history baseline" : m.name}
                  {m.status === "untrained" ? " · untrained" : ""}
                </option>
              ))}
            </select>
          </label>
          <label>
            Horizon <span>{horizon} cycles</span>
            <input
              type="range"
              min="20"
              max="3000"
              step="20"
              value={horizon}
              onChange={(e) => setHorizon(+e.target.value)}
            />
            <div className="range-labels">
              <span>20</span>
              <span>3,000</span>
            </div>
          </label>
          <div className="field-pair">
            <label>
              Temperature (°C)
              <input
                type="number"
                min="-20"
                max="60"
                step="0.1"
                required
                value={temperature}
                onChange={(e) => setTemperature(+e.target.value)}
              />
            </label>
            <label>
              C-rate
              <input
                type="number"
                min="0.1"
                max="5"
                step="0.1"
                required
                value={rate}
                onChange={(e) => setRate(+e.target.value)}
              />
            </label>
          </div>
          <label>
            End-of-life threshold (%)
            <input
              type="number"
              min="50"
              max="95"
              step="1"
              required
              value={eol}
              onChange={(e) => setEol(+e.target.value)}
            />
          </label>
          <div className="form-note">
            <ShieldCheck size={18} />
            <p>
              Untrained models remain unavailable. Baseline uncertainty is a
              sensitivity estimate.
            </p>
          </div>
          <button className="button primary wide" disabled={busy}>
            {busy ? (
              <Loader2 className="spin" size={16} />
            ) : (
              <Sparkles size={16} />
            )}
            Run forecast
            <ArrowRight size={16} />
          </button>
        </form>
      </Panel>
      <div>
        <Panel
          title="The road ahead"
          subtitle={`${cell.id} · Observations and projected degradation`}
          action={
            forecast && (
              <button
                className="button secondary small"
                onClick={() => exportRun(forecast.id)}
              >
                <ArrowDownToLine size={14} />
                Export
              </button>
            )
          }
        >
          <div className="chart-legend">
            <span>
              <i className="legend-line" />
              Observed
            </span>
            <span>
              <i className="legend-line dashed" />
              Forecast
            </span>
            <span>
              <i className="legend-box" />
              Uncertainty
            </span>
            {dirty && forecast && (
              <Badge tone="amber">Settings changed · rerun to update</Badge>
            )}
          </div>
          <HealthChart history={history} forecast={forecast} />
          {forecast ? (
            <div className="forecast-summary">
              <div>
                <span>Remaining useful life</span>
                <strong>
                  {forecast.rul_cycles === null
                    ? `> ${forecast.horizon.toLocaleString()}`
                    : Math.round(forecast.rul_cycles).toLocaleString()}
                  <small>cycles</small>
                </strong>
              </div>
              <div>
                <span>Degradation trend</span>
                <strong>
                  {forecast.rate_pct_per_100_cycles.toFixed(2)}
                  <small>pp / 100 cycles</small>
                </strong>
              </div>
              <div>
                <span>Evidence status</span>
                <Badge
                  tone={forecast.uncertainty.calibrated ? "green" : "amber"}
                >
                  {forecast.uncertainty.calibrated
                    ? "Calibrated"
                    : "Exploratory baseline"}
                </Badge>
              </div>
            </div>
          ) : (
            <div className="chart-footer">
              <span>Set a scenario and run your first forecast.</span>
              <Badge tone="neutral">Observed history only</Badge>
            </div>
          )}
        </Panel>
        {forecast && <Notes notes={forecast.warnings} />}
        <div className="quiet-card">
          <FlaskConical size={22} />
          <div>
            <h3>Make uncertainty part of the decision.</h3>
            <p>
              Read the interval alongside the trajectory. A lifetime beyond the
              forecast horizon is reported as censored, never as an invented
              exact date.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
