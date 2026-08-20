#!/usr/bin/env bash
set -euo pipefail

DIR=$(mktemp -d)
trap 'find "$DIR" -type f -delete; rmdir "$DIR"' EXIT
GATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/maps-visual-gate.sh"

magick -size 1280x720 canvas:'#86cee8' "$DIR/empty.png"
if "$GATE" --before "$DIR/empty.png" --after "$DIR/empty.png" \
    --settled "$DIR/empty.png" >/dev/null 2>&1; then
  echo "self-test: empty canvas was accepted" >&2
  exit 1
fi

magick -size 1280x720 canvas:white \
  -fill '#8bc6e8' -draw 'polygon 230,80 560,80 710,260 590,500 230,620' \
  -fill '#263238' -font DejaVu-Sans -pointsize 28 \
  -annotate +760+140 'North District' -annotate +760+240 'River Park' \
  -annotate +760+340 'Central Station' -annotate +760+440 'South Village' \
  "$DIR/map.png"
"$GATE" --before "$DIR/map.png" --after "$DIR/map.png" \
  --settled "$DIR/map.png"

echo "self-test: pass"
