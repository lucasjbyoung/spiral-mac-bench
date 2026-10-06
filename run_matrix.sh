#!/usr/bin/env bash
# Reproduce the spiral-fit scaling matrix with one command.
#
#   SPIRAL_DIR=~/villa/spiral-fitting SPIRAL_DATASET=~/spiral_datasets/PHercParis4 \
#   PYTHON=~/villa/spiral-fitting/.venv/bin/python RUNS_DIR=./runs ./run_matrix.sh [run ...]
#
# Runs, in order (default: all):
#   mps_vponly_z1000  verified patches only, 1,000 slices (the day-one input set)
#   cpu_full_z1000    full inputs, 1,000 slices, on the CPU
#   mps_full_z1000 mps_full_z2000 mps_full_z4000 mps_full_z8000 mps_full_z13000
# Each run gets RUNS_DIR/<run>/{run.log,bench.json,driver_result.json,mps_memory_samples.tsv}.
# Checkpoints are deleted after each run unless KEEP_CHECKPOINTS=1.
# Then: $PYTHON scripts/make_table.py RUNS_DIR results/
set -uo pipefail

if [ "${1:-}" = -h ] || [ "${1:-}" = --help ]; then
  sed -n '2,13s/^# \{0,1\}//p' "$0"; exit 0
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
: "${SPIRAL_DIR:?set SPIRAL_DIR to villa/spiral-fitting at the commit under test}"
: "${SPIRAL_DATASET:?set SPIRAL_DATASET to the PHercParis4 spiral-input root}"
: "${PYTHON:=python}"
: "${RUNS_DIR:=$HERE/runs}"
: "${SPIRAL_CACHE:=$RUNS_DIR/cache}"
export SPIRAL_DIR SPIRAL_DATASET SPIRAL_CACHE

ALL=(mps_vponly_z1000 cpu_full_z1000 mps_full_z1000 mps_full_z2000 mps_full_z4000 mps_full_z8000 mps_full_z13000)
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=("${ALL[@]}")

mkdir -p "$RUNS_DIR"
for run in "${RUNS[@]}"; do
  device="${run%%_*}"            # mps | cpu
  spec="$HERE/specs/${run#*_}.json"
  dir="$RUNS_DIR/$run"
  if [ -f "$dir/bench.json" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "== $run: already done (FORCE=1 to redo)"; continue
  fi
  rm -rf "$dir/out"; mkdir -p "$dir"
  echo "== $run: $(date '+%F %T') spec=$spec device=$device commit=$(git -C "$SPIRAL_DIR" rev-parse HEAD)" | tee "$dir/run.log"
  # PYTORCH_ENABLE_MPS_FALLBACK stays unset: an op MPS lacks fails loudly instead of silently running on the CPU.
  env -u PYTORCH_ENABLE_MPS_FALLBACK FIT_SPIRAL_DEVICE="$device" TQDM_MININTERVAL=30 \
    /usr/bin/time -l "$PYTHON" "$HERE/scripts/bench_fit.py" "$spec" "$dir" >> "$dir/run.log" 2>&1
  echo "EXIT $?" | tee -a "$dir/run.log"
  [ "${KEEP_CHECKPOINTS:-0}" = 1 ] || find "$dir/out" -name '*.ckpt' -delete 2>/dev/null
done
