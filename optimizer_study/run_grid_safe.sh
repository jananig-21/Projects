#!/usr/bin/env bash
# Duplicate-proof grid runner.
#   ./run_grid_safe.sh <jobs-file> <parallelism>
# A cell is skipped if (a) it already finished (run_meta.json exists) or (b) any live
# process is computing it right now (e.g. a run started by another scheduler).
# A per-cell claim directory (mkdir is atomic) stops this runner's own workers from racing.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/user/qaoa-env/bin/python}"
mkdir -p "$HERE/logs" "$HERE/.claims"
cell_is_live() {   # $1..$4 = P I O T ; true if a python process is running exactly this cell
  for pid in $(pgrep -x python); do
    tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q -- "--problem $1 --init $2 --optimizer $3 --trial $4 " && return 0
  done; return 1
}
run_one() {
  read -r P I O T B <<<"$1"
  [ -f "$HERE/results/$P/$I/$O/trial$T/run_meta.json" ] && return 0
  cell_is_live "$P" "$I" "$O" "$T" && { echo "[skip-live] $1"; return 0; }
  mkdir "$HERE/.claims/${P}_${I}_${O}_${T}" 2>/dev/null || { echo "[skip-claimed] $1"; return 0; }
  OMP_NUM_THREADS=1 "$PY" -W ignore "$HERE/run_study.py" --problem "$P" --init "$I" --optimizer "$O" \
      --trial "$T" --budget "$B" > "$HERE/logs/${P}_${I}_${O}_t${T}.log" 2>&1
  grep -h "\[done\]" "$HERE/logs/${P}_${I}_${O}_t${T}.log" || echo "[FAIL] $1"
}
export -f run_one cell_is_live; export HERE PY
grep -v '^\s*#' "$1" | grep -v '^\s*$' | xargs -P "$2" -I{} bash -c 'run_one "{}"'
