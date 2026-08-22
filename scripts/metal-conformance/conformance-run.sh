#!/usr/bin/env bash
# Boot one x86 rail and run the Metal conformance battery inside the guest.
#
#   conformance-run.sh <outdir>
#
# The binary is built on a native Apple host (see the header of
# conformance.swift) and cross-compiled for x86_64, so the guest needs no
# developer tools -- `AGENTS.md` records that a guest-side build does not
# degrade gracefully, it simply fails, and reports a build error that reads like
# noise.
#
# Environment passes through, so an arm is `REIMS_VGPU_X=y conformance-run.sh ...`.
set -uo pipefail
export LC_ALL=C
OUT="${1:?usage: conformance-run.sh <outdir>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAIL="${RAIL:-macos-13}"
BIN="$REPO/scripts/metal-conformance/conformance-x86_64"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "no cross-built binary at $BIN"; exit 2; }

pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
for _ in $(seq 1 30); do
  pgrep -f 'qemu-system-x86_6[4].*reims-vgpu' >/dev/null || break
  sleep 1
done
rm -f /tmp/reims-vgpu-fail.log

echo "conformance: booting rail=$RAIL"
TESTING_TIMEOUT="${TESTING_TIMEOUT:-1800}" \
  "$REPO/vm/boot-x86.sh" --device reims-vgpu-pci --rail "$RAIL" --testing \
  >"$OUT/boot-stdout.log" 2>&1 &
BOOT_PID=$!

# Only a live device writes the fail log, so this is what says the boot is ours
# and not a survivor holding 2222.
for _ in $(seq 1 240); do
  [ -f /tmp/reims-vgpu-fail.log ] && break
  kill -0 "$BOOT_PID" 2>/dev/null || break
  sleep 2
done
[ -f /tmp/reims-vgpu-fail.log ] || { echo "device never came up"; exit 1; }

"$REPO/vm/guest-authorize.sh" >"$OUT/authorize.log" 2>&1
for _ in $(seq 1 60); do
  timeout 20 ssh -o BatchMode=yes macos-vm true 2>/dev/null && break
  sleep 5
done

# Prefer building in the guest where the rail has `swiftc`: the battery is
# edited far more often than a rail is added, and a guest-side build removes the
# cross-build round trip from that loop. Where the rail has no compiler -- which
# `AGENTS.md` records is the normal case, not the exception -- the cross-built
# binary is what runs, and it is what makes the two hosts comparable.
timeout 60 scp -o BatchMode=yes -q "$REPO/scripts/metal-conformance/conformance.swift" \
  macos-vm:/tmp/conformance.swift
if timeout 120 ssh -o BatchMode=yes macos-vm 'command -v swiftc >/dev/null' 2>/dev/null; then
  echo "conformance: building in the guest"
  timeout 600 ssh -o BatchMode=yes macos-vm \
    'cd /tmp && swiftc -O conformance.swift -o conformance' >"$OUT/build.log" 2>&1 || {
      echo "guest build failed; see $OUT/build.log"; }
fi
timeout 120 ssh -o BatchMode=yes macos-vm 'test -x /tmp/conformance' 2>/dev/null || {
  timeout 60 scp -o BatchMode=yes -q "$BIN" macos-vm:/tmp/conformance || {
    echo "could not copy the battery into the guest"; exit 1; }
}
timeout 600 ssh -o BatchMode=yes macos-vm 'chmod +x /tmp/conformance && /tmp/conformance' \
  >"$OUT/conformance.txt" 2>&1
rc=$?
echo "conformance rc=$rc"
cp /tmp/reims-vgpu-fail.log "$OUT/device.log" 2>/dev/null

grep -q 'guest kernel panic' "$OUT/boot-stdout.log" && echo "PANIC" || echo "no panic"
echo "--- results ---"
cat "$OUT/conformance.txt"
pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
exit $rc
