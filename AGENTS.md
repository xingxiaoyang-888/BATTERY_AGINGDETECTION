# Development conventions

- Keep comments, docstrings, and documentation in English; use the user's language in conversation.
- The primary application is `python -m sodium_twin`. React sources are in `web/src`.
- Preserve the `server.py` import contract and legacy tests when modifying FMU adapters.
- Never train, download weights, or fit preprocessing as an import side effect. The offline CLI defaults to dry-run.
- Label synthetic data, observed data, baseline forecasts, and trained predictions explicitly.
- Fit preprocessing only on training cells. Keep calibration and test groups independent.
- Do not claim accuracy gains without measured results and an evaluation protocol.
- Configure model paths and native runtimes through the environment. Never add default credentials.
- Run relevant tests in `tests/` and `test/`, the English source check, and the frontend build after substantive changes.
- Keep runtime databases, model weights, caches, and generated runs out of tracked source.
