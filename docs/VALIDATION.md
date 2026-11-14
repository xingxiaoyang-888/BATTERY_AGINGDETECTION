# Validation record

Validation date: **2026-11-14**. This record describes checks actually performed on the local Windows environment. It does not report trained-model accuracy.

## Automated checks

| Check | Observed outcome |
|---|---|
| `python -m pytest tests test -q` | **91 passed in 17.42 seconds**, no failures or skips |
| `python tools/check_english.py` | Passed: source comments/docstrings and project documentation use English |
| `npm --prefix web run build` | Passed: TypeScript compilation and Vite production build |
| `git diff --check` | Passed; Git only reported Windows line-ending normalization notices |
| Frontend dependency audit during installation | 0 reported vulnerabilities at installation time |

The Python total includes **51 new tests** and **40 legacy regression tests**. Tests exercise:

- Strict CSV/API contracts, duplicate cycles, non-finite JSON values, unknown fields, upload limits, and explicit offline data limits.
- Persistence across application instances, run provenance, full record retrieval, and trajectory CSV export.
- Baseline scenario response, ordered monotonic forecasts, and no fabricated finite RUL beyond the requested horizon.
- Electrical current direction, SOC/voltage cutoffs, finite temperatures under extreme bounded inputs, and preservation of initial conditions.
- All four learned encoders: output dimensions, batch isolation, finite gradients, ordered quantiles, and monotonic degradation.
- Independent attention-block initialization and masked reconstruction without overlap leakage.
- Cell-disjoint partitions, train-only scaling, consecutive-cycle windows, finite-sample conformal ranks, and analytic censored survival likelihood.
- A dry-run that cannot call an optimizer step, corrupted weight checksums, overlapping split metadata, and unavailable/untrained model behavior.
- Legacy simulation contracts, aging calculations, lifetime/censoring regressions, sodium data preparation, residual models, and account handling without default credentials.

Gradient checks use randomly initialized models to verify differentiability. They do **not** update parameters. No training job, benchmark fit, optimizer step, or new trained checkpoint was produced for this release.

## Browser integration checks

The compiled application was served by FastAPI on `127.0.0.1:8000` and checked in Microsoft Edge:

| User flow | Observed outcome |
|---|---|
| Overview and cell navigation | Fleet totals, observed SOH chart, cell selection, and synthetic-data label rendered |
| Forecast studio | A real API request produced and persisted a baseline forecast; the horizon-censored RUL and uncalibrated envelope were labeled |
| Data workspace | A 40-row, one-cell synthetic CSV fixture imported successfully and appeared in the dataset selector |
| Portable digital twin | A real simulation produced temperature/SOC/voltage metrics and a thermal chart |
| Native FMU integration | 8S2P, 10 A, 10 seconds executed through the new UI and original Modelica adapter |
| Model registry | Baseline available; four learned architectures explicitly untrained with no invented accuracy metrics |
| Reports | Forecast, portable simulation, and native simulation records persisted; a downloaded JSON file was parsed and verified |
| Responsive navigation | Mobile viewport checked at 390 × 844; menu opened/closed and the document had no horizontal overflow |
| Browser console | No captured application warnings/errors in the checked flows |

The browser download event hook timed out for the generated JSON blob, but the file was present on disk and its contents were successfully parsed. The exported native record included request settings, solver identity, and all 11 time samples.

Screenshots: [overview](assets/dashboard.jpg) and [native digital twin](assets/digital-twin.jpg).

## Native FMU observation

The supplied repository contains 26 compiled topologies. Only **8S2P** was executed. The test used 50 Ah cells, 25 °C ambient, 80% initial SOC, 95% initial SOH, and 2 W/K cooling. The adapter returned:

| Quantity | Observed value |
|---|---|
| Time samples | 11 |
| Final SOC | 79.97% |
| Peak temperature | 25.0 °C |
| Energy throughput | 0.000763 kWh |
| Reported SOH | 95% |

These values establish execution and serialization, not physical accuracy. The original solver warns that initial SOH is applied to the aging calculation and is not coupled to FMU electrical resistance. The new UI exposes that limitation. The legacy thermal pickle was incompatible with the available numerical stack and was left disabled; no replacement model was trained.

## Environment and remaining validation

Local versions: Python 3.10.11, PyTorch 2.0.1 CPU, NumPy 1.24.4, pandas 2.2.2, FastAPI 0.135.1, Pydantic 2.12.5, pytest 8.2.0, HTTPX 0.28.1, Node.js 24.13.0, npm 11.6.2, and Vite 7.3.8. Native execution used the installed OpenModelica runtime discovered through its environment setting.

The following are supplied or implemented but **not validated by this record**:

- Predictive accuracy, calibration coverage on measured batteries, chemistry transfer, out-of-distribution robustness, and improvement over the old learned models.
- Full training execution, GPU behavior, large-cohort memory/performance, or published-paper reproduction.
- The remaining 25 native FMU configurations, native fault/spatial fidelity, or non-Windows FMU execution.
- Docker image execution, remote CI execution, multi-user authorization, or public deployment.

The original sodium-only lifetime feature table is absent from the supplied snapshot. The dashboard therefore starts with a labeled synthetic fleet. Acceptance of the software checks should precede a separate, measured-data evaluation plan.
