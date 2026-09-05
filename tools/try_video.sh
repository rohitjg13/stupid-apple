#!/usr/bin/env bash
# Run any video through the shopper pipeline and open the annotated result.
#
#   tools/try_video.sh vid/arcade1.mp4
#   tools/try_video.sh vid/arcade1.mp4 --bg-lr 0.0005 --morph 2     # extra knobs pass through
#   tools/try_video.sh vid/arcade1.mp4 --detector yolo               # person detector (Jetson)
#
# Writes out/<name>_tracked.mp4 and out/<name>_tracked.json, prints the report,
# then opens the video. Layout-free: no door, no zones, no floor calibration --
# it measures what needs none of those: who was there, how long they stood
# still, and where (the heatmap painted over the frame).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
  echo "usage: tools/try_video.sh <video> [viz_tracks options]" >&2
  exit 2
fi
video="$1"; shift
[ -f "$video" ] || { echo "no such file: $video" >&2; exit 1; }

name="$(basename "${video%.*}")"
mkdir -p out
out="out/${name}_tracked.mp4"

.venv/bin/python -m tools.viz_tracks --video "$video" --out "$out" "$@"

echo
echo "wrote $out"
command -v open >/dev/null && open "$out"
