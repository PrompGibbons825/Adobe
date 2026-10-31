#!/usr/bin/env bash
# One-command local setup: virtual environment, dependencies, model files,
# then a run over the three sample designs in input/.
#
#   ./setup.sh
#
# Safe to re-run: every step skips what is already in place.
set -euo pipefail
cd "$(dirname "$0")"

# Python 3.11 is what the Docker image uses and what every pinned dependency
# supports. 3.12 works too. On 3.13+ rapidocr-onnxruntime's metadata refuses
# to install (its code runs fine), so pip is told to ignore that check.
PY=""
for c in python3.11 python3.12 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3 not found. Install Python 3.11 from https://www.python.org/downloads/"; exit 1; }
MINOR=$("$PY" -c 'import sys; print(sys.version_info.minor)')
[ "$("$PY" -c 'import sys; print(sys.version_info.major)')" = "3" ] && [ "$MINOR" -ge 10 ] || { echo "Need Python 3.10 or newer (found $("$PY" --version))."; exit 1; }
PIP_FLAGS=""
[ "$MINOR" -ge 13 ] && PIP_FLAGS="--ignore-requires-python"
echo "==> Using $("$PY" --version) ($PY)"

if [ ! -d .venv ]; then
  echo "==> Creating .venv"
  "$PY" -m venv .venv
fi

echo "==> Installing dependencies (about a minute)"
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q $PIP_FLAGS -r requirements.txt

echo "==> Fetching model files (about 150 MB, once)"
.venv/bin/python tools/fetch_models.py

echo "==> Running the three sample designs in input/"
.venv/bin/python -m engine.main 2>&1 | grep -v "net_impl_backend"

echo
echo "Done. Open the PNGs in output/ to see the results."
echo "Next time:  .venv/bin/python -m engine.main"
