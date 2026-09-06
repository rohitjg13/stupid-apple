#!/usr/bin/env bash
# Run any video through the shopper pipeline; get a report folder per video.
#
#   tools/try_video.sh vid/arcade1.mp4 --detector yolo
#   tools/try_video.sh vid/arcade1.mp4                          # background subtraction
#   tools/try_video.sh vid/arcade1.mp4 --detector yolo --config config/mystore
#
# Writes out/<name>/ containing:
#   tracked.mp4        annotated video
#   heatmap.png        4. dwell heatmap: green = walked through, red = stood still (also in the video)
#   dwell.csv          3. per person: present, standing still, distance walked, region
#   footfall.csv       2. people present / arrivals / departures per time bucket
#   entries_exits.csv  1. door crossings with --config, else a labelled frame-edge proxy
#   summary.txt/.json  the headline numbers
# then prints summary.txt and opens the video and the heatmap.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
  echo "usage: tools/try_video.sh <video> [--detector yolo] [--config <clipset>] [viz_tracks options]" >&2
  exit 2
fi
video="$1"; shift
[ -f "$video" ] || { echo "no such file: $video" >&2; exit 1; }

name="$(basename "${video%.*}")"
report="out/${name}"

.venv/bin/python -m tools.viz_tracks --video "$video" --report "$report" "$@" 2>&1 \
  | grep -vE "^INFO|tracker params|clip exhausted|^$" || true

echo
echo "report: $report/"
ls -1 "$report"
if command -v open >/dev/null; then
  open "$report/tracked.mp4"
  open "$report/heatmap.png"
fi
