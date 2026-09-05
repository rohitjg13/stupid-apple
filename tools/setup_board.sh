#!/usr/bin/env bash
# One-time board setup. Run as root on the PYNQ over SSH.
#
#   sudo tools/setup_board.sh
set -euo pipefail

APP_DIR=${APP_DIR:-/home/xilinx/retail}
STATE_DIR=/var/lib/retail

echo "== python deps (board set only: no torch, no pandas) =="
python3 -m pip install --no-cache-dir \
  "numpy" "opencv-python-headless" "fastapi" "uvicorn" "pyyaml" "scipy"

echo "== state dir =="
mkdir -p "$STATE_DIR"

echo "== service =="
install -m 644 "$APP_DIR/retail.service" /etc/systemd/system/retail.service
systemctl daemon-reload
systemctl enable retail.service

echo "== log rotation (SD cards fill up and then the demo dies) =="
cat > /etc/systemd/journald.conf.d/retail.conf <<'CONF'
[Journal]
SystemMaxUse=200M
RuntimeMaxUse=50M
CONF
systemctl restart systemd-journald

echo "== no NTP is fine, but use it when it is there =="
timedatectl set-ntp true 2>/dev/null || echo "  (timedatectl unavailable; clock falls back to core/clock.py)"

echo
echo "done. start with:  systemctl start retail && journalctl -u retail -f"
