# Architecture and operating boundaries

`sodium_twin` separates contracts, ingestion, storage, forecasting, simulation, and optional learning. Its application factory accepts explicit settings, allowing isolated temporary test workspaces. Importing the API does not load PyTorch or instantiate FMUs.

`web/src/App.tsx` owns navigation and dataset/cell selection. Page modules own forecast, simulation, and import controls. Shared components implement charts, evidence labels, metrics, and the schematic pack. Frontend data comes from the API; predictions are not fabricated in JavaScript.

## Persistence and API

SQLite stores validated observations and immutable run records. Connections are scoped and closed. Queries use bound parameters. The deterministic demo is inserted once with `synthetic` provenance. Uploads receive opaque IDs; names never become filesystem paths. The application is a single-user local workspace and does not reuse legacy accounts.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/v2/health` | Version and workspace status |
| GET / POST | `/api/v2/datasets` | List / validate and import |
| GET | `/api/v2/datasets/{id}/cells/{cell}` | Observed history |
| GET | `/api/v2/template.csv` | Import template |
| GET | `/api/v2/models` | Architecture/artifact availability |
| POST | `/api/v2/forecasts` | Baseline or guarded learned inference |
| POST | `/api/v2/simulations` | Portable electrical/thermal solver |
| GET | `/api/v2/fmu/configurations` | FMU files and compiled platforms |
| POST | `/api/v2/simulations/fmu` | Native FMU adapter |
| GET | `/api/v2/runs` | Latest 100 runs |
| GET | `/api/v2/runs/{id}` | Full JSON export |
| GET | `/api/v2/runs/{id}/export.csv` | Portable/forecast trajectory export |

The native adapter preserves the old structured JSON response and does not expose a v2 trajectory CSV. Both API generations reject untrusted Host headers, non-loopback peers without configured authentication, and cross-origin browser requests. Request bodies are bounded by actual streamed bytes and deadlines, independently of Content-Length. Request IDs and timing headers support diagnosis. Pydantic rejects unknown fields and non-finite values.

## Concurrency and native runtime

Synchronous handlers run in FastAPI's thread pool. The legacy async simulation endpoint now explicitly offloads initialization and native computation. A process-local lock serializes FMU runtime access and environment changes. This is not a durable distributed queue. FMI simulation has a 120-second solver timeout and a bounded output sampling grid. Native DLL faults may still terminate the process; production isolation requires a dedicated worker process.

Runtime discovery uses `OPENMODELICA_BIN`, `OPENMODELICAHOME/bin`, or `omc` on PATH. Stored FMUs retain their compiled topology and host-platform requirements. Changing topology or structural model parameters requires re-export.

## Physical scope

Positive current denotes discharge. Pack current divides by parallel count; series count scales terminal voltage. Effective capacity depends on initial SOH. A polarization state and temperature-dependent resistance determine terminal voltage/losses. A lumped thermal balance exchanges heat with ambient through a cooling coefficient. One-second integration is sampled into an economical output grid.

Legacy `aging_algorithm.py` and `physics_lib.py` contain lithium-ion approximations. They remain for compatibility and are not claimed as calibrated sodium-ion physics. Modelica source constants are preserved; modifying them requires calibration and recompilation where relevant.

## Production work remaining

Single-operator authentication, request limits, and database quotas are implemented. Per-user authorization, multi-tenancy, durable workers, formal storage migrations, and a privacy policy remain outside this local release. Container port publication is loopback-only. Network binding requires a key; use TLS and explicit allowed hostnames. Local mode trusts other loopback clients, so enable a key on shared computers. SQLite is not encrypted at rest.
