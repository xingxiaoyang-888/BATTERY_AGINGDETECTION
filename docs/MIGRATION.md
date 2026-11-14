# Migration from the original platform

Base revision: `ddd6e97` from `xingxiaoyang-888/BATTERY_AGINGDETECTION`. The modernization is developed on `feat/sodium-twin-v2`; the package/product release is version 3.0.0. The branch name is not an API/version contract.

| Before | Current behavior |
|---|---|
| Streamlit-only primary console | React/TypeScript primary UI; Streamlit retained |
| Monolithic v1 service | Separate v2 package with explicit contracts and factory configuration |
| Native work directly inside an async handler | Thread-pool offload and serialized native access |
| Fixed OpenModelica drive path | Environment/PATH runtime discovery |
| Automatic legacy thermal pickle loading | Opt-in `SODIUM_THERMAL_MODEL` |
| Default `admin / 1234` provisioning | No default password; explicit environment provisioning only |
| Tracked local database and bytecode | Removed from the current tree; generated files ignored |
| Missing sodium artifacts treated as an external prerequisite | Explicit synthetic workspace and missing-artifact model state |
| Extensive localized comments and stale prose | Concise English documentation; legacy runtime UI text retained |

Run the new application with `python -m sodium_twin`. Keep `python server.py` and `streamlit run main.py` for the old workflow, in separate terminals. Do not run both backends on port 8000 simultaneously.

The old `.db` is not migrated into the new workspace. Existing user data should be backed up independently before any future migration. The checked-in historical database is removed from this revision, not retroactively erased from Git history.

To provision a legacy administrator, set `SODIUM_ADMIN_USER` and `SODIUM_ADMIN_PASSWORD` before initialization. Passwords must be 12–72 UTF-8 bytes. Old accounts in an existing external database are not changed automatically. The new local research console has no account system and must remain loopback/private.

The old sodium-only cohort artifacts were absent in the repository. Set `SODIUM_LIFETIME_FEATURE_PATH` after recovering the audited feature table. Do not substitute mixed-chemistry historical data. Native FMUs and Modelica equations are retained; this release does not claim a new electrochemical calibration.

Legacy comment cleanup was checked against executable Python ASTs with docstrings excluded. Explicit concurrency/configuration fixes are separately covered by regression tests. Original source remains recoverable from Git history.
