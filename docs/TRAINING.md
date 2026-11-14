# Offline training workflow

Training has **not** been run for this release. The commands below document the implemented workflow for later use.

## Inspect first

Audit chemistry, units, cell independence, ordering, and the beginning-of-life reference. Prepare the [CSV contract](DATA_CONTRACT.md), install `python -m pip install -e ".[ml,test]"`, and inspect each architecture:

```powershell
python -m sodium_twin.ml.train --csv data/measured_cycles.csv --architecture patchtst
python -m sodium_twin.ml.train --csv data/measured_cycles.csv --architecture itransformer
python -m sodium_twin.ml.train --csv data/measured_cycles.csv --architecture selective-ssm
python -m sodium_twin.ml.train --csv data/measured_cycles.csv --architecture tcn
```

Defaults are 64 history cycles, a 128-cycle horizon, seed 42, and **no optimization**. Reports show exact cell partitions, window/parameter counts, and zero optimizer steps. The 12-cell, five-cycle-sampled UI demo is intentionally unsuitable as a training cohort.

## Execute only when training is requested

```powershell
python -m sodium_twin.ml.train --csv data/measured_cycles.csv --architecture patchtst --chemistry sodium-ion --provenance measured --epochs 80 --device cpu --output artifacts/run_001 --execute
```

Use CUDA only with a compatible PyTorch build. Existing artifact directories are not overwritten. Saved artifacts contain state tensors, training history, data/weight checksums, independent cell partitions, scaler, calibration, and evaluation metrics.

`--masked-weight 0` disables the Patch Transformer reconstruction auxiliary for ablation. Other architectures ignore that objective. The hazard auxiliary uses `--eol-pct` (default 80); its survival outputs are not calibrated or served as operational probabilities.

## Evaluate and activate

Validation MAE selects the checkpoint. Calibration uses its own cells afterward. Test metrics include median MAE/RMSE, pointwise coverage, interval width, and persistence MAE. Failing to beat persistence marks the artifact rejected. This is a minimal evidence gate, not certification or external validation.

Set `SODIUM_MODEL_DIR` to a selected run root such as `artifacts/run_001`. The registry expects `<root>/<architecture>/manifest.json` and `weights.pt`. Synthetic artifacts cannot activate operational inference. Only trusted local weights should be installed. The workflow never uploads data to an external service.

Compare robust trend and all learned models on the same untouched cells/horizons. Reserve external data before repeated architecture tuning. Report event coverage and uncertainty, not only point-error metrics.
