#!/usr/bin/env bash
# One-time setup on a Jetson Orin Nano. Run once, with a network; everything
# after that works offline.
#
#   sudo tools/setup_jetson.sh
#
# What it does NOT do: install torch. JetPack ships a CUDA build of torch and
# pip would happily replace it with a CPU wheel, so that stays your call:
#   https://developer.nvidia.com/embedded/jetson-linux  (torch for your JetPack)
set -euo pipefail

APP_DIR=${APP_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
STATE_DIR=${STATE_DIR:-/var/lib/retail}
MODEL_DIR=${MODEL_DIR:-$APP_DIR/models}
YOLO_MODEL=${YOLO_MODEL:-yolov8n.pt}

echo "== python deps =="
python3 -m pip install --no-cache-dir \
  numpy opencv-python-headless fastapi "uvicorn[standard]" pyyaml scipy \
  python-multipart

echo "== detector =="
if python3 -c "import torch" 2>/dev/null; then
  python3 -m pip install --no-cache-dir ultralytics
  mkdir -p "$MODEL_DIR"
  # Fetch the weights now, while there is a network. RETAIL_YOLO_MODEL points
  # at the file so ultralytics never tries to download during a demo.
  python3 - "$MODEL_DIR/$YOLO_MODEL" <<'PY'
import shutil, sys
from pathlib import Path
from ultralytics import YOLO
dest = Path(sys.argv[1])
if not dest.exists():
    src = Path(YOLO(dest.name).ckpt_path if hasattr(YOLO(dest.name), "ckpt_path") else dest.name)
    if src.exists() and src.resolve() != dest.resolve():
        shutil.copy(src, dest)
print(f"weights: {dest} ({'ok' if dest.exists() else 'MISSING — copy one here'})")
PY
else
  echo "!! torch not importable, so no YOLO. The box will fall back to"
  echo "!! --backend reference (OpenCV background subtraction, CPU)."
fi

echo "== dashboard =="
if command -v npm >/dev/null; then
  (cd "$APP_DIR/web" && npm ci && npm run build)
else
  echo "-- no npm; using the web/dist committed in the repo"
fi

echo "== state dir =="
mkdir -p "$STATE_DIR" "$APP_DIR/data"

echo "== service =="
sed -e "s|@APP_DIR@|$APP_DIR|g" -e "s|@MODEL@|$MODEL_DIR/$YOLO_MODEL|g" \
    "$APP_DIR/retail-web.service" > /etc/systemd/system/retail-web.service
systemctl daemon-reload
systemctl enable retail-web.service
systemctl restart retail-web.service

echo
echo "Open http://$(hostname -I | awk '{print $1}'):8000 and upload a clip."
