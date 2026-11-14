import {
  ArrowUpRight,
  BatteryCharging,
  ChevronRight,
  Info,
} from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Forecast, Observation } from "./types";
import type { ReactNode } from "react";

export function Badge({
  children,
  tone = "green",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return (
    <span className={`badge ${tone}`}>
      <span className="badge-dot" />
      {children}
    </span>
  );
}

export function Metric({
  label,
  value,
  unit,
  note,
  icon,
}: {
  label: string;
  value: string;
  unit?: string;
  note: string;
  icon: ReactNode;
}) {
  return (
    <article className="metric">
      <div className="metric-label">
        {label}
        <span>{icon}</span>
      </div>
      <div className="metric-value">
        {value}
        <small>{unit}</small>
      </div>
      <p>{note}</p>
    </article>
  );
}

export function Panel({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function HealthChart({
  history,
  forecast,
}: {
  history: Observation[];
  forecast?: Forecast | null;
}) {
  const points: {
    cycle_index: number;
    observed?: number;
    median?: number;
    band?: number[];
  }[] = history.map((p) => ({
    cycle_index: p.cycle_index,
    observed: p.soh_pct,
  }));
  if (forecast) {
    const last = points[points.length - 1];
    if (last) {
      last.median = forecast.current_soh_pct;
      last.band = [forecast.current_soh_pct, forecast.current_soh_pct];
    }
    points.push(
      ...forecast.trajectory.slice(1).map((p) => ({
        cycle_index: p.cycle_index,
        median: p.median,
        band: [p.lower, p.upper],
      })),
    );
  }
  const floor = forecast
    ? Math.min(75, forecast.eol_pct - 2)
    : Math.max(
        65,
        Math.floor(Math.min(...history.map((p) => p.soh_pct), 90) / 5) * 5 - 5,
      );
  return (
    <div
      className="chart-wrap"
      role="img"
      aria-label="Battery state of health by cycle. Forecast and uncertainty are shown only after a prediction is run."
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={points}
          margin={{ top: 12, right: 15, bottom: 5, left: -22 }}
        >
          <defs>
            <linearGradient id="healthFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#5a977d" stopOpacity={0.17} />
              <stop offset="100%" stopColor="#5a977d" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            vertical={false}
            stroke="#e9ede9"
            strokeDasharray="4 4"
          />
          <XAxis
            dataKey="cycle_index"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickLine={false}
            axisLine={false}
            fontSize={11}
            tickMargin={12}
            minTickGap={45}
          />
          <YAxis
            domain={[floor, 105]}
            tickFormatter={(v) => `${v}%`}
            tickLine={false}
            axisLine={false}
            fontSize={11}
          />
          <Tooltip
            contentStyle={{
              borderRadius: 12,
              border: "1px solid #e5eae4",
              fontSize: 12,
            }}
            labelFormatter={(v) => `Cycle ${v}`}
            formatter={(value, name) => [
              Array.isArray(value)
                ? value.map((v) => Number(v).toFixed(2)).join(" – ") + "%"
                : Number(value).toFixed(2) + "%",
              name,
            ]}
          />
          <Area
            dataKey="observed"
            type="monotone"
            stroke="none"
            fill="url(#healthFill)"
            name="Observed SOH"
            isAnimationActive={false}
          />
          {forecast && (
            <Area
              dataKey="band"
              stroke="none"
              fill="#dbe9da"
              name="Uncertainty envelope"
              isAnimationActive={false}
            />
          )}
          <Line
            dataKey="observed"
            type="monotone"
            stroke="#377b5f"
            strokeWidth={2.7}
            dot={false}
            name="Observed SOH"
            isAnimationActive={false}
          />
          <Line
            dataKey="median"
            type="monotone"
            stroke="#86a358"
            strokeWidth={2.2}
            strokeDasharray="6 5"
            dot={false}
            name="Forecast median"
            isAnimationActive={false}
          />
          {forecast && (
            <ReferenceLine
              y={forecast.eol_pct}
              stroke="#c1a57c"
              strokeDasharray="4 4"
              label={{
                value: "EOL threshold",
                position: "insideBottomRight",
                fill: "#aa8b61",
                fontSize: 10,
              }}
            />
          )}
          {forecast && (
            <ReferenceLine
              x={forecast.current_cycle}
              stroke="#a4b3aa"
              strokeDasharray="3 3"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PackGraphic({ temperature = 27.5 }: { temperature?: number }) {
  return (
    <div
      className="pack-visual"
      aria-label={`Illustrative battery module. Uniform temperature ${temperature.toFixed(1)} degrees Celsius.`}
    >
      <div className="pack-glow" />
      <div className="pack-base">
        <div className="cell-grid">
          {Array.from({ length: 24 }, (_, i) => (
            <div className="battery-cell" key={i}>
              <span />
              <i />
            </div>
          ))}
        </div>
      </div>
      <span className="pack-tag">
        <span className="live-dot" />
        {temperature.toFixed(1)}°C <small>illustrative module</small>
      </span>
      <span className="pack-axis">
        X <span>↗</span> Y
      </span>
    </div>
  );
}

export function Empty({
  title,
  text,
  action,
}: {
  title: string;
  text: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <BatteryCharging size={32} />
      <h3>{title}</h3>
      <p>{text}</p>
      {action}
    </div>
  );
}

export function Notes({ notes }: { notes: string[] }) {
  return (
    <div className="notes">
      {notes.map((note) => (
        <p key={note}>
          <Info size={15} />
          <span>{note}</span>
        </p>
      ))}
    </div>
  );
}

export function TextLink({
  children,
  onClick,
}: {
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button className="text-button" onClick={onClick}>
      {children}
      <ChevronRight size={15} />
    </button>
  );
}

export function ModelPipeline() {
  return (
    <div className="pipeline">
      {[
        "Cycle history",
        "Temporal encoder",
        "Physical constraints",
        "Calibrated forecast",
      ].map((name, i) => (
        <div key={name}>
          <span>0{i + 1}</span>
          <strong>{name}</strong>
          {i < 3 && <ArrowUpRight size={16} />}
        </div>
      ))}
    </div>
  );
}
