#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="stm32g431-ai-frequency-response"
ICON_SRC="$APP_DIR/assets/icons/frequency_response_icon.png"
DESKTOP_SRC="$APP_DIR/stm32g431-ai-frequency-response.desktop"
ICON_DIR="$HOME/.local/share/icons/hicolor/512x512/apps"
APP_DIR_DESKTOP="$HOME/.local/share/applications"

mkdir -p "$ICON_DIR" "$APP_DIR_DESKTOP"
cp "$ICON_SRC" "$ICON_DIR/$APP_NAME.png"
sed "s|Icon=frequency_response_icon|Icon=$APP_NAME|g" "$DESKTOP_SRC" > "$APP_DIR_DESKTOP/$APP_NAME.desktop"
chmod +x "$APP_DIR/run_ubuntu.sh" "$APP_DIR/build_ubuntu.sh" "$APP_DIR/install_desktop_ubuntu.sh"
chmod +x "$APP_DIR_DESKTOP/$APP_NAME.desktop"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APP_DIR_DESKTOP" || true
fi

echo "Installed desktop launcher:"
echo "$APP_DIR_DESKTOP/$APP_NAME.desktop"
