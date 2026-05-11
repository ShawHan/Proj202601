#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="/Users/master/miniconda3/envs/use/bin/python"
fi

"$PYTHON_BIN" main.py --method XGBBaseline,PINN20260414,PINN20260511 --runs 10 --seed 42 --torch-threads 1
