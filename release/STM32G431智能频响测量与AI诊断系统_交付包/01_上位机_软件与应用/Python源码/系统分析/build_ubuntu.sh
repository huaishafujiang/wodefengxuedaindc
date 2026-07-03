#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Install it with: sudo apt install python3 python3-venv python3-tk"
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-linux.txt pyinstaller

pyinstaller --noconfirm --clean --onefile --windowed \
  --name "stm32g431-ai-frequency-response" \
  --add-data "system_analysis/data/diagnosis_knowledge_base.json:system_analysis/data" \
  --icon "assets/icons/frequency_response_icon.png" \
  main.py

echo
echo "Build complete:"
echo "$APP_DIR/dist/stm32g431-ai-frequency-response"
