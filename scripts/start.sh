#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
if [ ! -x .venv/bin/python ]; then
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else "Python 3.11+ is required")'
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -c constraints-security.txt -e '.[test]'
pushd web >/dev/null
npm ci
npm run build
popd >/dev/null
exec .venv/bin/python -m sodium_twin --port "${PORT:-8000}"
