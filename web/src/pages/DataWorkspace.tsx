import { useState } from "react";
import {
  ArrowDownToLine,
  ArrowUpRight,
  Check,
  Database,
  Loader2,
  Plus,
  UploadCloud,
} from "lucide-react";
import { api } from "../api";
import type { Work } from "../api";
import { Badge, Panel } from "../components";
import type { Dataset } from "../types";

export default function DataWorkspace({
  datasets,
  busy,
  work,
  onImport,
  open,
}: {
  datasets: Dataset[];
  busy: boolean;
  work: Work;
  onImport: (d: Dataset) => Promise<void>;
  open: (d: Dataset) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [chemistry, setChemistry] = useState("sodium-ion");
  async function importData() {
    if (!file) return;
    await work(async () => {
      if (file.size > 2_000_000) throw new Error("CSV exceeds the 2 MB limit.");
      const d = await api<Dataset>("/datasets", {
        name: name || file.name.replace(/\.csv$/i, ""),
        chemistry,
        csv: await file.text(),
      });
      await onImport(d);
      setFile(null);
      setName("");
    });
  }
  return (
    <>
      <div className="data-grid">
        <Panel
          title="Bring your own observations"
          subtitle="A strict schema keeps every cycle traceable."
        >
          <form
            onSubmit={(e) => {
              e.preventDefault();
              importData();
            }}
          >
            <label className="upload-zone">
              <UploadCloud size={32} />
              <strong>{file ? file.name : "Choose a cycle CSV"}</strong>
              <span>Up to 2 MB · 20,000 rows · 200 cells</span>
              <input
                type="file"
                accept=".csv,text/csv"
                aria-label="Choose cycle CSV"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </label>
            <div className="field-pair">
              <label>
                Dataset name
                <input
                  maxLength={80}
                  placeholder="e.g. Sodium aging cohort A"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </label>
              <label>
                Chemistry
                <select
                  value={chemistry}
                  onChange={(e) => setChemistry(e.target.value)}
                >
                  <option value="sodium-ion">Sodium-ion</option>
                  <option value="lithium-ion">Lithium-ion</option>
                </select>
              </label>
            </div>
            <div className="import-actions">
              <a href="/api/v2/template.csv" className="text-button">
                <ArrowDownToLine size={15} />
                Download template
              </a>
              <button className="button primary" disabled={!file || busy}>
                {busy ? (
                  <Loader2 size={16} className="spin" />
                ) : (
                  <Plus size={16} />
                )}
                Validate & import
              </button>
            </div>
          </form>
        </Panel>
        <div className="schema-card">
          <Database size={27} />
          <h2>
            Good models start
            <br />
            with good data.
          </h2>
          <p>
            Each row represents one observed cycle. Health is expressed in
            percentage points.
          </p>
          <div className="schema-fields">
            {[
              "cell_id",
              "cycle_index",
              "soh_pct",
              "temperature_c",
              "c_rate",
              "capacity_ah",
            ].map((f) => (
              <code key={f}>
                {f}
                <Check size={13} />
              </code>
            ))}
          </div>
          <p className="small-text">
            Missing values, duplicate cycles, and invalid units are rejected. A
            cell requires at least 8 observations. Learned models require
            consecutive cycles.
          </p>
        </div>
      </div>
      <Panel
        title="Your datasets"
        subtitle="Synthetic demonstrations and imported data remain explicitly labeled."
      >
        <div className="dataset-list">
          {datasets.map((d) => (
            <button className="dataset-row" key={d.id} onClick={() => open(d)}>
              <div className="dataset-symbol">
                <Database size={21} />
              </div>
              <div>
                <strong>{d.name}</strong>
                <span>
                  {d.chemistry} · {d.cell_count} cells ·{" "}
                  {d.rows.toLocaleString()} observations
                </span>
              </div>
              <Badge tone={d.provenance === "synthetic" ? "amber" : "green"}>
                {d.provenance === "synthetic" ? "Synthetic demo" : "Imported"}
              </Badge>
              <ArrowUpRight size={18} />
            </button>
          ))}
        </div>
      </Panel>
    </>
  );
}
