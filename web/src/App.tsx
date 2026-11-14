import { useEffect, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Battery,
  Box,
  Check,
  ChevronDown,
  CircleHelp,
  Cpu,
  Database,
  FileText,
  Layers,
  LayoutDashboard,
  Loader2,
  Menu,
  Plus,
  Search,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X,
  Zap,
} from "lucide-react";
import { api, downloadJson } from "./api";
import {
  Badge,
  Empty,
  HealthChart,
  Metric,
  ModelPipeline,
  Notes,
  PackGraphic,
  Panel,
  TextLink,
} from "./components";
import type { Cell, Dataset, Forecast, Model, Observation, Run } from "./types";

import ForecastStudio from "./pages/ForecastStudio";
import TwinStudio from "./pages/TwinStudio";
import DataWorkspace from "./pages/DataWorkspace";

type Page = "overview" | "forecast" | "twin" | "data" | "models" | "reports";
const navigation = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "forecast", label: "Forecast studio", icon: Activity },
  { id: "twin", label: "Digital twin", icon: Box },
  { id: "data", label: "Data workspace", icon: Database },
  { id: "models", label: "Model registry", icon: Cpu },
  { id: "reports", label: "Reports & runs", icon: FileText },
] as const;

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState("demo");
  const [cellId, setCellId] = useState("NA-001");
  const [history, setHistory] = useState<Observation[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [query, setQuery] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  const dataset = datasets.find((d) => d.id === datasetId);
  const cell = dataset?.cells.find((c) => c.id === cellId);

  async function refresh() {
    const [ds, ms, rs] = await Promise.all([
      api<Dataset[]>("/datasets"),
      api<Model[]>("/models"),
      api<Run[]>("/runs"),
    ]);
    setDatasets(ds);
    setModels(ms);
    setRuns(rs);
  }
  useEffect(() => {
    refresh()
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);
  useEffect(() => {
    let active = true;
    setHistory([]);
    setForecast(null);
    api<Observation[]>(
      `/datasets/${encodeURIComponent(datasetId)}/cells/${encodeURIComponent(cellId)}`,
    )
      .then((data) => {
        if (active) setHistory(data);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [datasetId, cellId]);
  function navigate(next: Page) {
    setPage(next);
    setMobileNav(false);
    setError("");
  }
  function chooseDataset(id: string) {
    const next = datasets.find((d) => d.id === id);
    if (next) {
      setDatasetId(id);
      setCellId(next.cells[0].id);
    }
  }
  async function work<T>(fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      return await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
      return undefined;
    } finally {
      setBusy(false);
    }
  }
  async function exportRun(id: string) {
    await work(async () => {
      downloadJson(
        `sodium-run-${id.slice(0, 8)}.json`,
        await api(`/runs/${id}`),
      );
      setNotice("Run exported with inputs, provenance, and results.");
    });
  }
  const average = dataset
    ? dataset.cells.reduce((sum, c) => sum + c.soh_pct, 0) / dataset.cell_count
    : 0;
  const reviewCount =
    dataset?.cells.filter((c) => c.status === "review").length ?? 0;

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("overview");
          }}
        >
          <div className="brand-mark">
            <Zap size={23} fill="currentColor" />
          </div>
          <span>
            SODIUM<span className="brand-light">TWIN</span>
            <small>BATTERY INTELLIGENCE</small>
          </span>
        </a>
        <div className="workspace-switch">
          <div className="workspace-icon">
            <Layers size={17} />
          </div>
          <div>
            <strong>Research workspace</strong>
            <span>Local environment</span>
          </div>
          <ChevronDown size={14} />
        </div>
        <p className="nav-caption">WORKSPACE</p>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              className={page === item.id ? "nav-item active" : "nav-item"}
              key={item.id}
              onClick={() => navigate(item.id)}
            >
              <item.icon size={18} />
              {item.label}
              {item.id === "models" && <span className="nav-count">4</span>}
              {page === item.id && <span className="nav-active-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <div>
            <Sparkles size={18} />
            <span>Built for the next cycle.</span>
          </div>
          <p>
            Physics meets learning.
            <br />
            Every prediction has a provenance.
          </p>
          <button onClick={() => navigate("models")}>
            Explore the architecture <ArrowUpRight size={15} />
          </button>
        </div>
        <div className="sidebar-bottom">
          <a href="/docs" target="_blank" rel="noreferrer">
            <CircleHelp size={17} />
            API documentation
            <ArrowUpRight size={13} />
          </a>
          <div className="profile">
            <div className="avatar">ST</div>
            <span>
              <strong>Sodium Twin Lab</strong>
              <small>Research edition · v3.0</small>
            </span>
            <span className="live-dot" />
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              aria-label="Toggle navigation"
              onClick={() => setMobileNav(!mobileNav)}
            >
              <Menu size={20} />
            </button>
            <span>Workspace</span>
            <span>/</span>
            <strong>{navigation.find((n) => n.id === page)?.label}</strong>
          </div>
          <div className="topbar-right">
            <span className="local-status">
              <span className="live-dot" />
              Local workspace
            </span>
            <div className="topbar-divider" />
            <span className="top-avatar">ST</span>
          </div>
        </header>
        <main>
          {error && (
            <div className="message error" role="alert">
              <InfoIcon />
              {error}
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div className="message success" role="status">
              <Check size={16} />
              {notice}
              <button
                aria-label="Dismiss notification"
                onClick={() => setNotice("")}
              >
                <X size={16} />
              </button>
            </div>
          )}
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <span />{" "}
                {page === "overview"
                  ? "THE BIG PICTURE"
                  : "SODIUM TWIN WORKBENCH"}
              </div>
              <h1>
                {
                  {
                    overview: "Battery intelligence",
                    forecast: "Forecast studio",
                    twin: "Your battery. In perspective.",
                    data: "A home for every cycle.",
                    models: "Intelligence, by design.",
                    reports: "From insight to evidence.",
                  }[page]
                }
              </h1>
              <p>
                {
                  {
                    overview:
                      "Understand degradation. Explore what comes next.",
                    forecast:
                      "Turn observed history into an uncertainty-aware outlook.",
                    twin: "Explore the electrical and thermal behavior of your pack.",
                    data: "Bring your observations into one traceable workspace.",
                    models:
                      "Modern sequence models, grounded in battery physics.",
                    reports: "Revisit, compare, and export your research runs.",
                  }[page]
                }
              </p>
            </div>
            <div className="heading-actions">
              {page === "overview" ? (
                <>
                  <button
                    className="button secondary"
                    onClick={() => navigate("data")}
                  >
                    <UploadCloud size={16} />
                    Import data
                  </button>
                  <button
                    className="button primary"
                    onClick={() => navigate("forecast")}
                  >
                    <Plus size={16} />
                    New forecast
                  </button>
                </>
              ) : (
                <Badge tone="neutral">Research edition</Badge>
              )}
            </div>
          </div>
          {loading ? (
            <div className="loading">
              <Loader2 className="spin" />
              Loading your workspace…
            </div>
          ) : (
            <>
              {["overview", "forecast"].includes(page) && dataset && (
                <div className="context-bar">
                  <div className="context-select">
                    <Database size={15} />
                    <select
                      aria-label="Active dataset"
                      value={datasetId}
                      onChange={(e) => chooseDataset(e.target.value)}
                    >
                      {datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name}
                        </option>
                      ))}
                    </select>
                    <span className="context-divider" />
                    <select
                      aria-label="Active cell"
                      value={cellId}
                      onChange={(e) => setCellId(e.target.value)}
                    >
                      {dataset.cells.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.id}
                        </option>
                      ))}
                    </select>
                  </div>
                  <Badge
                    tone={
                      dataset.provenance === "synthetic" ? "amber" : "green"
                    }
                  >
                    {dataset.provenance === "synthetic"
                      ? "Synthetic demo data"
                      : "Imported observations"}
                  </Badge>
                </div>
              )}
              {page === "overview" && dataset && cell && (
                <>
                  <div className="metrics">
                    <Metric
                      label="Fleet health"
                      value={average.toFixed(1)}
                      unit="%"
                      note="Mean of latest observed SOH"
                      icon={<Battery size={17} />}
                    />
                    <Metric
                      label="Cells in workspace"
                      value={String(dataset.cell_count).padStart(2, "0")}
                      note={`${dataset.rows.toLocaleString()} cycle observations`}
                      icon={<Layers size={17} />}
                    />
                    <Metric
                      label="Observed cycles"
                      value={cell.cycles.toLocaleString()}
                      note={`Latest observation · ${cell.id}`}
                      icon={<Activity size={17} />}
                    />
                    <Metric
                      label="Cells to review"
                      value={String(reviewCount).padStart(2, "0")}
                      note="Latest SOH below 90%"
                      icon={<ShieldCheck size={17} />}
                    />
                  </div>
                  <div className="overview-grid">
                    <Panel
                      title="The health of every cycle"
                      subtitle={`${cell.id} · State of health over observed lifetime`}
                      action={<span className="small-chip">Cycle history</span>}
                    >
                      <div className="chart-legend">
                        <span>
                          <i className="legend-line" />
                          Observed SOH
                        </span>
                        {forecast && (
                          <>
                            <span>
                              <i className="legend-line dashed" />
                              Forecast
                            </span>
                            <span>
                              <i className="legend-box" />
                              Sensitivity envelope
                            </span>
                          </>
                        )}
                        <span className="legend-right">SOH (%)</span>
                      </div>
                      <HealthChart history={history} forecast={forecast} />
                      <div className="chart-footer">
                        <span>
                          <span className="live-dot" /> {history.length}{" "}
                          observations ·{" "}
                          {dataset.provenance === "synthetic"
                            ? "Demonstration series"
                            : "Imported series"}
                        </span>
                        <TextLink onClick={() => navigate("forecast")}>
                          Explore forecast
                        </TextLink>
                      </div>
                    </Panel>
                    <section className="insight-card">
                      <div className="insight-top">
                        <span className="eyebrow">CELL SPOTLIGHT</span>
                        <ArrowUpRight size={18} />
                      </div>
                      <h2>
                        More life.
                        <br />
                        More understanding.
                      </h2>
                      <p>A closer look at {cell.id}.</p>
                      <div
                        className="health-ring"
                        style={
                          {
                            "--health": `${cell.soh_pct * 3.6}deg`,
                          } as React.CSSProperties
                        }
                      >
                        <div>
                          <Battery size={21} />
                          <strong>
                            {cell.soh_pct.toFixed(1)}
                            <small>%</small>
                          </strong>
                          <span>STATE OF HEALTH</span>
                        </div>
                      </div>
                      <div className="spotlight-info">
                        <span>
                          Latest temperature
                          <strong>{cell.temperature_c}°C</strong>
                        </span>
                        <span>
                          Nominal capacity<strong>{cell.capacity_ah} Ah</strong>
                        </span>
                      </div>
                      <button onClick={() => navigate("forecast")}>
                        Look ahead
                        <ArrowRight size={16} />
                      </button>
                    </section>
                  </div>
                  <div className="bottom-grid">
                    <Panel
                      title="Your cells, at a glance"
                      subtitle="Choose a cell to inspect its history."
                      action={
                        <TextLink onClick={() => navigate("data")}>
                          View dataset
                        </TextLink>
                      }
                    >
                      <div className="table-search">
                        <Search size={15} />
                        <input
                          aria-label="Search cells"
                          value={query}
                          onChange={(e) => setQuery(e.target.value)}
                          placeholder="Find a cell…"
                        />
                      </div>
                      <CellTable
                        cells={dataset.cells
                          .filter((c) =>
                            c.id.toLowerCase().includes(query.toLowerCase()),
                          )
                          .slice(0, 5)}
                        selected={cellId}
                        select={setCellId}
                      />
                    </Panel>
                    <Panel
                      title="A twin with a purpose"
                      subtitle="An interactive view of your battery system."
                      action={<Box size={19} />}
                    >
                      <PackGraphic temperature={cell.temperature_c} />
                      <div className="chart-footer">
                        <span>Electrical + thermal exploration</span>
                        <TextLink onClick={() => navigate("twin")}>
                          Open twin
                        </TextLink>
                      </div>
                    </Panel>
                  </div>
                </>
              )}
              {page === "forecast" && dataset && cell && (
                <ForecastStudio
                  key={`${datasetId}-${cellId}`}
                  dataset={dataset}
                  cell={cell}
                  models={models}
                  history={history}
                  forecast={forecast}
                  busy={busy}
                  run={async (body) => {
                    const success = await work(async () => {
                      const result = await api<Forecast>("/forecasts", body);
                      setForecast(result);
                      await refresh();
                      return true;
                    });
                    return success === true;
                  }}
                  exportRun={exportRun}
                />
              )}
              {page === "twin" && (
                <TwinStudio
                  busy={busy}
                  work={work}
                  onComplete={refresh}
                  exportRun={exportRun}
                />
              )}
              {page === "data" && (
                <DataWorkspace
                  datasets={datasets}
                  busy={busy}
                  work={work}
                  onImport={async (d) => {
                    await refresh();
                    setDatasetId(d.id);
                    setCellId(d.cells[0].id);
                    setNotice(
                      `${d.rows} observations imported. Your dataset is ready.`,
                    );
                  }}
                  open={(d) => {
                    setDatasetId(d.id);
                    setCellId(d.cells[0].id);
                    navigate("overview");
                  }}
                />
              )}
              {page === "models" && (
                <>
                  <div className="model-hero">
                    <div className="model-hero-icon">
                      <Cpu size={30} />
                    </div>
                    <div>
                      <span className="eyebrow">
                        A MULTI-MODEL RESEARCH STACK
                      </span>
                      <h2>Different architectures. One rigorous contract.</h2>
                      <p>
                        Every learned model predicts ordered SOH quantiles.
                        Serving requires measured data, independent calibration,
                        and held-out evaluation.
                      </p>
                    </div>
                    <Badge tone="amber">
                      {models.some((m) => m.status === "validated")
                        ? "Validated artifacts loaded"
                        : "Training not run"}
                    </Badge>
                  </div>
                  <ModelPipeline />
                  <div className="model-grid">
                    {models.map((model, i) => (
                      <article className="panel model-card" key={model.id}>
                        <div className="model-card-top">
                          <span className="model-number">0{i + 1}</span>
                          <Badge
                            tone={
                              ["available", "validated"].includes(model.status)
                                ? "green"
                                : "neutral"
                            }
                          >
                            {model.status === "untrained"
                              ? "Ready to train"
                              : model.status}
                          </Badge>
                        </div>
                        <span className="eyebrow">{model.family}</span>
                        <h2>{model.name}</h2>
                        <p>{model.description}</p>
                        <div className="tags">
                          {model.tags.map((tag) => (
                            <span key={tag}>{tag}</span>
                          ))}
                        </div>
                        <div className="model-evidence">
                          <ShieldCheck size={16} />
                          <span>
                            {model.metrics
                              ? `Held-out MAE: ${model.metrics.mae_pp.toFixed(3)} pp`
                              : model.id === "robust-trend"
                                ? "History-based · No learned weights"
                                : "No checkpoint · No accuracy claimed"}
                          </span>
                        </div>
                        {model.reason && <Notes notes={[model.reason]} />}
                      </article>
                    ))}
                  </div>
                  <Notes
                    notes={[
                      "Patch Transformer, Inverted Transformer, and Selective State Space are battery-specific research implementations inspired by published architectures. The selective scan is a portable reference, not the fused official Mamba-2 implementation.",
                      "Training is a separate command-line workflow and is never triggered from this workspace.",
                    ]}
                  />
                </>
              )}
              {page === "reports" && (
                <Panel
                  title="Research runbook"
                  subtitle="Every run preserves its settings, source, and result."
                  action={
                    <Badge tone="neutral">{runs.length} recent runs</Badge>
                  }
                >
                  {runs.length ? (
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>RUN</th>
                            <th>TYPE</th>
                            <th>CREATED</th>
                            <th>SUBJECT</th>
                            <th>EXPORT</th>
                          </tr>
                        </thead>
                        <tbody>
                          {runs.map((r) => (
                            <tr key={r.id}>
                              <td className="mono">{r.id.slice(0, 8)}</td>
                              <td>
                                <Badge tone="neutral">{r.kind}</Badge>
                              </td>
                              <td>{new Date(r.created_at).toLocaleString()}</td>
                              <td>
                                {String(
                                  r.request.cell_id ||
                                    `${r.request.series}S × ${r.request.parallel}P`,
                                )}
                              </td>
                              <td>
                                <div className="inline-actions">
                                  <button
                                    className="text-button"
                                    disabled={busy}
                                    onClick={() => exportRun(r.id)}
                                  >
                                    JSON
                                    <ArrowDownToLine size={14} />
                                  </button>
                                  {r.kind !== "fmu" && (
                                    <a
                                      className="text-button"
                                      href={`/api/v2/runs/${r.id}/export.csv`}
                                    >
                                      CSV
                                      <ArrowDownToLine size={14} />
                                    </a>
                                  )}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <Empty
                      title="Your next insight starts here"
                      text="Run a forecast or simulation to create a reproducible record."
                      action={
                        <button
                          className="button primary"
                          onClick={() => navigate("forecast")}
                        >
                          Create a forecast
                          <ArrowRight size={15} />
                        </button>
                      }
                    />
                  )}
                </Panel>
              )}
              {!dataset && ["overview", "forecast"].includes(page) && (
                <Empty
                  title="Workspace unavailable"
                  text="Check the API connection and reload the workspace."
                  action={
                    <button
                      className="button primary"
                      onClick={() => work(refresh)}
                    >
                      Retry connection
                    </button>
                  }
                />
              )}
            </>
          )}
          <footer className="page-footer">
            <span>
              <Zap size={13} />
              SODIUM TWIN <span className="footer-dot">·</span> Built for better
              battery decisions.
            </span>
            <span>
              Research workspace <span className="footer-dot">/</span> v3.0
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}

function InfoIcon() {
  return <CircleHelp size={17} />;
}

function CellTable({
  cells,
  selected,
  select,
}: {
  cells: Cell[];
  selected: string;
  select: (id: string) => void;
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>CELL ID</th>
            <th>HEALTH</th>
            <th>CYCLES</th>
            <th>STATUS</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {cells.map((cell) => (
            <tr
              key={cell.id}
              className={selected === cell.id ? "selected-row" : ""}
            >
              <td>
                <button className="cell-link" onClick={() => select(cell.id)}>
                  <span className="cell-icon">
                    <Battery size={14} />
                  </span>
                  {cell.id}
                </button>
              </td>
              <td>
                <div className="mini-health">
                  <span>{cell.soh_pct.toFixed(1)}%</span>
                  <i>
                    <b style={{ width: `${Math.min(100, cell.soh_pct)}%` }} />
                  </i>
                </div>
              </td>
              <td>{cell.cycles}</td>
              <td>
                <Badge tone={cell.status === "review" ? "amber" : "green"}>
                  {cell.status === "review" ? "Review" : "Stable"}
                </Badge>
              </td>
              <td>
                <button
                  className="icon-button"
                  aria-label={`Inspect ${cell.id}`}
                  onClick={() => select(cell.id)}
                >
                  <ArrowUpRight size={15} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {cells.length === 0 && (
        <p className="no-results">No cells match your search.</p>
      )}
    </div>
  );
}
