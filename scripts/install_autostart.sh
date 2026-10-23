#!/usr/bin/env bash
# Install a login/boot autostart for the CADX updater daemon — the daemon is
# the supervisor, so starting it on login is enough for the full stack to
# come up (it respawns mesh-server + heals compose services).
#
# Usage:
#   scripts/install_autostart.sh [--uninstall] [--no-color] [-h|--help]
#
#     macOS:  ~/Library/LaunchAgents/ai.cadx.updater.plist
#     Linux:  systemd user unit ~/.config/systemd/user/cadx-updater.service
#             (falls back to a @reboot cron entry when systemd --user is off)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/ui.sh

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

UNINSTALL=0
while [ $# -gt 0 ]; do
  if ui_common_flags "$1"; then shift; continue; fi
  case "$1" in
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "unknown flag: $1"; usage >&2; exit 2 ;;
  esac
done

ROOT="$(pwd)"
PY="$(command -v python3)"
LABEL="ai.cadx.updater"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UNIT="$HOME/.config/systemd/user/cadx-updater.service"
CRON_MARK="# cadx-updater-autostart"

install_macos() {
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$PY</string><string>$ROOT/scripts/update_server.py</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>StandardOutPath</key><string>$ROOT/logs/updater.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/updater.log</string>
</dict></plist>
EOF
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  ok "launchd agent installed: $PLIST (keepalive, throttled 15s)"
}

install_systemd() {
  mkdir -p "$(dirname "$UNIT")"
  cat > "$UNIT" <<EOF
[Unit]
Description=CADX updater/supervisor daemon
After=docker.service network-online.target

[Service]
Type=simple
WorkingDirectory=$ROOT
ExecStart=$PY $ROOT/scripts/update_server.py
Restart=always
RestartSec=15
StandardOutput=append:$ROOT/logs/updater.log
StandardError=append:$ROOT/logs/updater.log

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now cadx-updater.service
  ok "systemd user unit installed: $UNIT (Restart=always)"
  info "for headless boot without login: loginctl enable-linger $USER"
}

install_cron() {
  local line="@reboot cd $ROOT && nohup $PY scripts/update_server.py >> logs/updater.log 2>&1 & $CRON_MARK"
  if crontab -l 2>/dev/null | grep -qF "$CRON_MARK"; then
    info "cron entry already present"
    return
  fi
  ( crontab -l 2>/dev/null | grep -vF "$CRON_MARK"; echo "$line" ) | crontab -
  ok "@reboot cron entry installed"
}

uninstall() {
  if [ -f "$PLIST" ]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    ok "removed $PLIST"
  fi
  if [ -f "$UNIT" ]; then
    systemctl --user disable --now cadx-updater.service 2>/dev/null || true
    rm -f "$UNIT"
    systemctl --user daemon-reload 2>/dev/null || true
    ok "removed $UNIT"
  fi
  if crontab -l 2>/dev/null | grep -qF "$CRON_MARK"; then
    crontab -l 2>/dev/null | grep -vF "$CRON_MARK" | crontab -
    ok "removed cron entry"
  fi
}

if [ "$UNINSTALL" = 1 ]; then
  say "removing cadx autostart"
  uninstall
  exit 0
fi

say "installing cadx autostart ($ROOT)"
mkdir -p logs run
case "$(uname -s)" in
  Darwin) install_macos ;;
  Linux)
    if command -v systemctl >/dev/null 2>&1 && systemctl --user status >/dev/null 2>&1; then
      install_systemd
    else
      warn "systemd --user unavailable — falling back to @reboot cron"
      install_cron
    fi
    ;;
  *) err "unsupported OS: $(uname -s)"; exit 1 ;;
esac
info "the updater daemon supervises the stack — login starts supervision"
