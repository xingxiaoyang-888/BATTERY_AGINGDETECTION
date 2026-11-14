# Legacy sodium training workflow

See [TRAINING.md](TRAINING.md) for the v3 workflow. The original research pipeline remains available:

- `models/soh_ai/run_pipeline.py`: preparation through the original data adapters.
- `models/soh_ai/train.py`: XGBoost, BiLSTM, Transformer, and ensemble training.
- `utils/audit_sodium_data.py`: sodium-only audit and provenance checks.
- `utils/build_rwth_lifetime_labels.py`: observed and right-censored EOL labels.
- `utils/evaluate_lifetime_backtest.py`: held-out-cell lifetime backtesting.

Inspect each command with `--help`. These old commands can train immediately; the v3 CLI's dry-run default does not change their behavior. No old training commands were executed during modernization.

The old service expects audited sodium features at `models/data/processed/sodium_ion/feature_table.parquet`. That artifact is absent from the supplied snapshot. Never substitute the legacy mixed-chemistry table. Recover the original audited sodium data before reproducing earlier cohort results.

Preserve dataset licenses, chemistry labels, chronological ordering, and independent evaluation partitions. Old joblib artifacts require compatible NumPy/scikit-learn versions and must come from a trusted source.
