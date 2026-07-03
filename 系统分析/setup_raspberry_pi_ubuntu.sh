#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${STM32G431_UPPER_VENV:-$HOME/.local/share/stm32g431_upper_host/.venv}"
cd "$APP_DIR"

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required on Raspberry Pi Ubuntu."
  exit 1
fi

sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-tk \
  python3-numpy \
  python3-scipy \
  python3-matplotlib \
  python3-serial

mkdir -p "$(dirname "$VENV_DIR")"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python - <<'PY'
import matplotlib
import numpy
import scipy
import serial

print("Python dependencies are available.")
PY

echo
echo "Setup complete. Run the app with:"
echo "  bash ./run_raspberry_pi_ubuntu.sh"
echo
echo "If the serial port cannot be opened, run:"
echo "  sudo usermod -aG dialout \$USER"
echo "Then reboot or log out and log back in."
