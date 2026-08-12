#!/usr/bin/env bash
# inner-tuner benchmark cell dispatcher (PLAN §八).
#
# Dynamically consumes the FIRST unfinished, unclaimed job from the jobs
# file — workers re-read the file before every job, so you can reorder,
# trim, or extend it at any time and every worker notices immediately.
# One worker per GPU (§八: one cell per GPU at a time):
#
#   export MODEL='deepseek-v4-flash[1m]'   # pin, per production run_metadata
#   tools/inner_benchmark/sweep.sh jobs.txt 0 &
#   tools/inner_benchmark/sweep.sh jobs.txt 1 &
#   wait
#
# Jobs file lines (whitespace-separated, # comments and blanks allowed):
#   <checkpoint_dir> <arm> <seed>
#
# Per-job state under the cells root (default ./cells-inner-v1):
#   <name>/result.json   finished (any status — arm_error is an outcome)
#   <name>.claim         taken by some worker (atomic mkdir)
#   <name>.log           cell stdout/stderr
# Recovery: an interrupted cell has a claim but no result.json and stays
# skipped; `rm -rf cells-inner-v1/*.claim` to requeue all such cells.
#
# Env: MODEL (required), MACHINE (default hostname), CELLS_CMD (test seam).

set -u
JOBS=${1:?usage: sweep.sh <jobs-file> <gpu> [cells-root]}
GPU=${2:?usage: sweep.sh <jobs-file> <gpu> [cells-root]}
CELLS=${3:-cells-inner-v1}
MODEL=${MODEL:?export MODEL=<pinned model id> first}
MACHINE=${MACHINE:-$(hostname)}
CELL_CMD=${CELL_CMD:-uv run python tools/inner_benchmark/cell.py}
cd "$(dirname "$0")/../.." || exit 2  # repo root
mkdir -p "$CELLS"

next_job() {
  # First line whose cell has neither a result.json nor a live claim.
  while IFS= read -r line; do
    case "$line" in ''|\#*) continue;; esac
    set -- $line
    [ -z "${1:-}" ] && continue
    name="$(basename "$1")__${2}__s${3}"
    [ -f "$CELLS/$name/result.json" ] && continue
    [ -e "$CELLS/$name.claim" ] && continue
    printf '%s\n' "$line"
    return 0
  done < "$JOBS"
  return 1
}

while job=$(next_job); do
  set -- $job
  ck=$1; arm=$2; seed=$3
  name="$(basename "$ck")__${arm}__s${seed}"
  out="$CELLS/$name"
  mkdir "$out.claim" 2>/dev/null || continue  # lost the race: re-scan
  rm -rf "$out"  # write-once manifest forbids restarting into a half cell
  echo "[$(date +%H:%M:%S)] gpu$GPU -> $name"
  CUDA_VISIBLE_DEVICES=$GPU $CELL_CMD \
    --arm "$arm" --checkpoint "$ck" --seed "$seed" \
    --out "$out" --model "$MODEL" --machine "$MACHINE" \
    > "$out.log" 2>&1
done
echo "[$(date +%H:%M:%S)] gpu$GPU: queue drained"
