#!/usr/bin/env bash
# Push the working tree to the board, restart, and tail the log.
#
#   tools/deploy.sh                 # uses BOARD=xilinx@192.168.2.99
#   BOARD=xilinx@10.0.0.5 tools/deploy.sh --no-follow
set -euo pipefail

BOARD=${BOARD:-xilinx@192.168.2.99}
APP_DIR=${APP_DIR:-/home/xilinx/retail}
FOLLOW=1
[ "${1:-}" = "--no-follow" ] && FOLLOW=0

echo "== rsync -> $BOARD:$APP_DIR =="
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'footage' --exclude 'hls/proj' --exclude '*.egg-info' \
  ./ "$BOARD:$APP_DIR/"

echo "== restart =="
ssh "$BOARD" "sudo systemctl restart retail && sleep 2 && systemctl is-active retail"

if [ "$FOLLOW" = "1" ]; then
  ssh "$BOARD" "journalctl -u retail -f -n 50"
fi
