# Cycle-level data contract

CSV must be UTF-8 with a header and these six columns:

| Field | Meaning | Range |
|---|---|---|
| `cell_id` | Stable cell identifier | 1–80 ASCII letters/digits/periods/underscores/hyphens |
| `cycle_index` | Physical cycle number | Integer 0–1,000,000 |
| `soh_pct` | Observed capacity-based SOH | 1–105 percent |
| `temperature_c` | Operating temperature | -40–100 °C |
| `c_rate` | Observed cycling rate | 0.01–15 C |
| `capacity_ah` | Nominal beginning-of-life capacity | 0.01–2,000 Ah |

Additional CSV columns are not persisted. Duplicate cell/cycle pairs and missing/non-finite values are rejected. Valid rows are sorted, never silently averaged or interpolated. Chemistry is explicitly selected at import. Minimum eight observations per cell; browser imports are limited to 2 MB, 20,000 rows, and 200 cells. The offline training CLI applies the same measurement validation without these web upload limits; its window construction still requires enough local memory.

Normalize SOH using a defensible beginning-of-life reference without future observations. Never normalize using maximum capacity over the full lifetime, use centered smoothing, or fit a scaler before splitting cells.

## Legacy conversion

The new importer does not automatically load `models/data/processed/feature_table.parquet`: its provenance/chemistry differ from the sodium-only path expected by the old lifetime service. Audit metadata first. An explicit conversion can map `soh * 100` to `soh_pct`, `nominal_capacity_ah` to `capacity_ah`, and a documented operating-rate definition to `c_rate`.

The old service requires `models/data/processed/sodium_ion/feature_table.parquet`, or `SODIUM_LIFETIME_FEATURE_PATH`. This artifact is absent from the supplied snapshot. The new demo is labeled synthetic rather than presented as a measured RWTH cohort.

## Sampling and independence

The trend baseline uses actual cycle spacing. Learned models require consecutive observed cycles; one forecast step always means one physical cycle. Do not silently fill gaps. Deep-model dry-runs need at least 20 cells with at least `lookback + horizon` observations each. Shared pack/batch dependencies require higher-level grouping before the split.
