#!/usr/bin/env bash
# Normalise footage to what the pipeline expects: 640x480, MJPEG in AVI.
#
#   tools/transcode.sh in.mp4 footage/overhead_01.avi
#   tools/transcode.sh --raw in.mp4 footage/overhead_01.npy   # decode-free benchmark clip
#
# --raw writes a BGR uint8 .npy memmap so tools/perf.py can time the PL without
# paying for JPEG decode on the board's CPU.
set -euo pipefail

RAW=0
if [ "${1:-}" = "--raw" ]; then RAW=1; shift; fi
IN="${1:?usage: transcode.sh [--raw] <in> <out>}"
OUT="${2:?usage: transcode.sh [--raw] <in> <out>}"
SECS="${SECS:-20}"

if [ "$RAW" = "1" ]; then
  python3 - "$IN" "$OUT" "$SECS" <<'PY'
import sys
import cv2, numpy as np
src, dst, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
n = int(fps * secs)
frames = []
while len(frames) < n:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.resize(f, (640, 480), interpolation=cv2.INTER_AREA))
cap.release()
if not frames:
    raise SystemExit(f"no frames decoded from {src}")
np.save(dst, np.asarray(frames, dtype=np.uint8))
print(f"{dst}: {len(frames)} frames, {np.asarray(frames).nbytes/1e6:.0f} MB")
PY
else
  ffmpeg -hide_banner -loglevel warning -y -i "$IN" \
    -vf "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:-1:-1:color=black,fps=15" \
    -c:v mjpeg -q:v 5 -an "$OUT"
  echo "wrote $OUT"
fi
