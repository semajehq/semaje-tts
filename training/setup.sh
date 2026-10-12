#!/usr/bin/env bash
# One-time setup for voice training. Idempotent.
#   training/setup.sh            # venv at $SEMAJE_HOME/venv-train
#
# piper-tts 1.8.0's wheel ships monotonic_align/setup.py but not core.pyx, so
# training cannot import piper.train.vits.monotonic_align.monotonic_align.core
# from a plain pip install. We fetch core.pyx from the matching piper1-gpl tag
# (same GPL-3.0 source) and compile it in place.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEMAJE_HOME="${SEMAJE_HOME:-$HOME/.local/share/semaje}"
VENV="${TRAIN_VENV:-$SEMAJE_HOME/venv-train}"
PIPER_VERSION="1.8.0"
PYTHON="${PYTHON:-python3.11}"

[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip uv
UV_HTTP_TIMEOUT=120 "$VENV/bin/uv" pip install --python "$VENV/bin/python" --quiet \
  -r "$ROOT/requirements.txt" -r "$ROOT/training/requirements-train.txt"

# find_spec locates the package without executing its broken __init__.
MA="$("$VENV/bin/python" -c 'import importlib.util as u; print(u.find_spec("piper.train.vits").submodule_search_locations[0] + "/monotonic_align")')"
if ! "$VENV/bin/python" -c 'import piper.train.vits.monotonic_align' 2>/dev/null; then
  echo "==> building monotonic_align (core.pyx from piper1-gpl v$PIPER_VERSION)"
  # Build outside the package so Cython names the module plain "core".
  BUILD="$(mktemp -d)"
  curl -fsSL "https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/v$PIPER_VERSION/src/piper/train/vits/monotonic_align/core.pyx" -o "$BUILD/core.pyx"
  cat > "$BUILD/setup.py" <<'PY'
import numpy
from Cython.Build import cythonize
from setuptools import setup
setup(ext_modules=cythonize("core.pyx", language_level=3), include_dirs=[numpy.get_include()])
PY
  (cd "$BUILD" && "$VENV/bin/python" setup.py build_ext --inplace >/dev/null 2>&1)
  mkdir -p "$MA/monotonic_align" && cp "$BUILD"/core*.so "$MA/monotonic_align/" && touch "$MA/monotonic_align/__init__.py"
  rm -rf "$BUILD"
fi
"$VENV/bin/python" -c 'import piper.train.vits.monotonic_align, torch, lightning; print("training ready · torch", torch.__version__, "· cuda", torch.cuda.is_available())'
