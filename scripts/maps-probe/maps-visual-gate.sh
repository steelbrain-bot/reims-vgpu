#!/usr/bin/env bash
# maps-visual-gate.sh — did Maps render the declared geographic workload?
#
# The Maps probe deliberately opens a dense city viewport. This gate checks the
# resulting contract without recognizing a particular city, label, colour, or
# screenshot size: the map area must not be dominated by one fill colour, and
# OCR must find several label-shaped text lines. Both are required in every
# frame, because geography without labels and labels over an empty fill are two
# different partial renders. A failed gate invalidates performance results.
set -euo pipefail
export LC_ALL=C

BEFORE=""
AFTER=""
SETTLED=""

while [ $# -gt 0 ]; do
  case "$1" in
    --before) BEFORE="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --settled) SETTLED="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "maps-visual-gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

for image in "$BEFORE" "$AFTER" "$SETTLED"; do
  [ -s "$image" ] || { echo "maps-visual-gate: missing captured frame" >&2; exit 2; }
done
command -v magick >/dev/null || {
  echo "maps-visual-gate: ImageMagick is required" >&2; exit 2; }
command -v tesseract >/dev/null || {
  echo "maps-visual-gate: tesseract is required" >&2; exit 2; }

WORK=$(mktemp -d)
trap 'find "$WORK" -type f -delete; rmdir "$WORK"' EXIT
failed=0
index=0

for image in "$BEFORE" "$AFTER" "$SETTLED"; do
  index=$((index + 1))
  read -r width height <<<"$(magick identify -format '%w %h' "$image")"

  # Keep only the central map viewport. The sidebar, title/toolbar, scale bar,
  # compass and zoom buttons all contain text, but none proves that Maps drew
  # its geographic label layer.
  x=$((width * 20 / 100))
  y=$((height * 10 / 100))
  crop_width=$((width * 75 / 100))
  crop_height=$((height * 80 / 100))
  crop="$WORK/frame-$index.png"
  magick "$image" -crop "${crop_width}x${crop_height}+${x}+${y}" +repage "$crop"

  pixels=$((crop_width * crop_height))
  dominant=$(magick "$crop" -colors 16 -format %c histogram:info:- |
    awk -F: 'BEGIN { max=0 } { gsub(/ /, "", $1); if ($1 + 0 > max) max=$1 + 0 } END { print max }')
  dominant_fraction=$(awk -v count="$dominant" -v total="$pixels" \
    'BEGIN { printf "%.4f", count / total }')
  # OCR at a fixed two-pixel sample for every captured pixel. The guest's map
  # labels are substantially smaller than the host's toolbar text at the
  # wider zoom levels this workload reaches; feeding their native screenshot
  # size to Tesseract loses readable labels before the rendered layer itself is
  # absent. Scaling the already-isolated viewport changes only the instrument.
  ocr="$WORK/ocr-$index.png"
  magick "$crop" -resize 200% "$ocr"
  labels=$(tesseract "$ocr" stdout --psm 11 tsv 2>/dev/null |
    awk -F '\t' 'NR > 1 && $12 ~ /[[:alpha:]][[:alpha:]][[:alpha:]]/ {
      count++
    } END { print count + 0 }')

  printf 'maps-visual-gate: %s dominant=%s labels=%s\n' \
    "$(basename "$image")" "$dominant_fraction" "$labels"

  # Quantizing to 16 colours makes this a large-scale canvas test rather than
  # an antialiasing/noise test. In the controlled empty-grid capture the fill
  # owns 0.8721 of the interior; in the widest valid driven capture it owns
  # 0.5915. The 0.80 boundary sits between those measured populations with
  # room on both sides. Three OCR words is below the ordinary toolbar-free
  # label population but above a fully label-free geography frame.
  if awk -v fraction="$dominant_fraction" 'BEGIN { exit !(fraction >= 0.80) }'; then
    echo "maps-visual-gate: INVALID — geographic layers do not cover the map interior"
    failed=1
  fi
  if [ "$labels" -lt 4 ]; then
    echo "maps-visual-gate: INVALID — the map interior does not contain its label layer"
    failed=1
  fi
done

[ "$failed" -eq 0 ] || exit 1
echo "maps-visual-gate: valid geographic content and labels in every frame"
