#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "未找到 apt-get。请确认瑞米派正在运行 Debian/Ubuntu 系统。"
  exit 1
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

echo "安装稳频仪上位机依赖：python3 / tkinter / numpy / matplotlib / pyserial"
$SUDO apt-get update
$SUDO apt-get install -y \
  python3 \
  python3-tk \
  python3-numpy \
  python3-matplotlib \
  python3-serial \
  python3-pip

if [ "${1:-}" = "--with-scipy" ]; then
  echo "安装可选依赖 scipy，用于 Savitzky-Golay 平滑。"
  $SUDO apt-get install -y python3-scipy
else
  echo "已跳过 scipy。需要更接近 MATLAB 的平滑时，可运行："
  echo "  ./install_remi_pi_deps.sh --with-scipy"
fi

echo "依赖安装完成。"
