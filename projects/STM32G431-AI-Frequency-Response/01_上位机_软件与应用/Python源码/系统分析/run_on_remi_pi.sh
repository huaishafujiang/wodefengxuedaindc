#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -z "${DISPLAY:-}" ]; then
  echo "当前没有检测到图形桌面 DISPLAY。"
  echo "请在瑞米派本机桌面终端运行，或用 ssh -X/远程桌面运行。"
  exit 1
fi

python3 main.py
