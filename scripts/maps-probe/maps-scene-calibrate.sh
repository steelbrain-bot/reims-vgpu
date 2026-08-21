#!/usr/bin/env bash
# maps-scene-calibrate.sh — find a Maps scene whose labels OCR can actually read.
#
#   maps-scene-calibrate.sh <outdir>
#
# The gate's label test asks whether Maps drew its type. On the scene the probe
# declared it could not: at that zoom the map interior carries no legible
# glyphs, so every word OCR returned was antialiasing garbage at 27-43 %
# confidence, and the test passed and failed the same frame at random.
#
# A threshold cannot fix that. What fixes it is a scene where correct rendering
# produces type OCR reads with confidence, so that a passing count is positive
# evidence the text layer rendered and a zero is a real absence. This walks a
# set of candidate scenes on one boot and reports, for each, how many words
# survive a confidence floor -- which is the number the gate's threshold has to
# have margin against in both directions.
#
# One boot for the whole sweep: each scene is an `open -a Maps <url>`, which
# re-aims the existing window without restarting anything.
set -uo pipefail
export LC_ALL=C

OUT="${1:?usage: maps-scene-calibrate.sh <outdir>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAIL="${RAIL:-macos-13}"
SHOT="$REPO/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh"
export QMP_SOCK="${QMP_SOCK:-$REPO/vm/disks/run/qmp.sock}"
Q="$REPO/scripts/qmp/qmp.py"
CONF_FLOOR="${CONF_FLOOR:-60}"
mkdir -p "$OUT"
say() { echo "maps-calibrate: $*"; }

# Manhattan at a range of zooms, and one deliberate negative: mid-ocean, which
# has no type at any zoom and is what a working test must score at zero.
SCENES="
z12 http://maps.apple.com/?ll=40.7128,-74.0060&z=12
z14 http://maps.apple.com/?ll=40.7128,-74.0060&z=14
z16 http://maps.apple.com/?ll=40.7128,-74.0060&z=16
z18 http://maps.apple.com/?ll=40.7128,-74.0060&z=18
ocean http://maps.apple.com/?ll=35.0000,-45.0000&z=12
"

pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
for _ in $(seq 1 30); do
  pgrep -f 'qemu-system-x86_6[4].*reims-vgpu' >/dev/null || break
  sleep 1
done
rm -f /tmp/reims-vgpu-fail.log

say "booting rail=$RAIL"
TESTING_TIMEOUT="${TESTING_TIMEOUT:-1500}" \
  "$REPO/vm/boot-x86.sh" --device reims-vgpu-pci --rail "$RAIL" --testing \
  >"$OUT/boot-stdout.log" 2>&1 &
BOOT_PID=$!

for _ in $(seq 1 240); do
  [ -f /tmp/reims-vgpu-fail.log ] && break
  kill -0 "$BOOT_PID" 2>/dev/null || break
  sleep 2
done
[ -f /tmp/reims-vgpu-fail.log ] || { say "device never came up"; exit 4; }

timeout 300 "$REPO/vm/guest-authorize.sh" >"$OUT/authorize.log" 2>&1 || true
for _ in $(seq 1 90); do
  timeout 20 ssh -o BatchMode=yes macos-vm 'pgrep -x Dock >/dev/null' 2>/dev/null && break
  sleep 4
done
say "desktop up"
sleep 8

timeout 60 ssh -o BatchMode=yes macos-vm "open -a Maps" 2>/dev/null || true
# Maps' first-run sheet, in the probe's own focus order.
sleep 20
"$Q" key tab spc >/dev/null 2>&1; sleep 2
"$Q" key tab spc >/dev/null 2>&1; sleep 10
"$Q" key ctrl+meta_l+f >/dev/null 2>&1; sleep 6

printf '%-8s %8s %8s  %s\n' scene words conf_ge_$CONF_FLOOR sample >"$OUT/calibration.txt"
echo "$SCENES" | while read -r name url; do
  [ -n "$name" ] || continue
  # `-n` because this loop is fed by a pipe and ssh would otherwise read the
  # remaining scenes off stdin, ending the sweep after its first entry.
  timeout 60 ssh -n -o BatchMode=yes macos-vm "open -a Maps '$url'" 2>/dev/null || true
  # Tiles and type arrive independently and asynchronously; this is setup, not a
  # measured window, so the wait is generous rather than tuned.
  sleep 30
  shot="$OUT/$name.png"
  REIMS_SHOT_NATIVE=1 "$SHOT" -o "$shot" >/dev/null 2>&1 || { say "$name: capture failed"; continue; }

  read -r w h <<<"$(magick identify -format '%w %h' "$shot")"
  crop="$OUT/$name-crop.png"
  magick "$shot" -crop "$((w*75/100))x$((h*80/100))+$((w*20/100))+$((h*10/100))" +repage "$crop"
  magick "$crop" -resize 200% "$OUT/$name-ocr.png"

  tsv=$(tesseract "$OUT/$name-ocr.png" stdout --psm 11 tsv 2>/dev/null)
  all=$(echo "$tsv" | awk -F'\t' 'NR>1 && $12 ~ /[[:alpha:]][[:alpha:]][[:alpha:]]/ {c++} END{print c+0}')
  good=$(echo "$tsv" | awk -F'\t' -v f="$CONF_FLOOR" \
    'NR>1 && $12 ~ /[[:alpha:]][[:alpha:]][[:alpha:]]/ && $11+0>=f {c++} END{print c+0}')
  sample=$(echo "$tsv" | awk -F'\t' -v f="$CONF_FLOOR" \
    'NR>1 && $12 ~ /[[:alpha:]][[:alpha:]][[:alpha:]]/ && $11+0>=f {printf "%s ", $12}' | cut -c1-60)
  printf '%-8s %8s %8s  %s\n' "$name" "$all" "$good" "$sample" >>"$OUT/calibration.txt"
  say "$name words=$all conf>=$CONF_FLOOR:$good  $sample"
done

cat "$OUT/calibration.txt"
pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
wait "$BOOT_PID" 2>/dev/null

# Keep this boot's fail log beside its captures. The device appends to one
# well-known path and the next sweep here truncates it, so a log read after the
# fact belongs to whichever sweep ran last -- which is how a capability gate
# (`host_pointer_import=`) or an audit counter ends up quoted from a different
# boot than the frames it is being used to explain. Copying makes the pair
# self-contained, and the `vk_caps` count printed here is what proves the copy
# holds exactly one boot.
cp -f /tmp/reims-vgpu-fail.log "$OUT/fail.log" 2>/dev/null &&
  say "fail log -> $OUT/fail.log (boots=$(grep -c vk_caps "$OUT/fail.log"))"
say "done -> $OUT/calibration.txt"
