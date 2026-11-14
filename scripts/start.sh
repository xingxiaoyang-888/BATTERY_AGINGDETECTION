#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
pushd web >/dev/null
npm ci
npm run build
popd >/dev/null
exec .venv/bin/python -m sodium_twin --port "${PORT:-8000}"
