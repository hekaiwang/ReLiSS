#!/usr/bin/env bash
# Source before inference/evaluation commands.
reliss_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export RELISS_DATA_ROOT="${RELISS_DATA_ROOT:-$reliss_project_root/data}"
export nnUNet_raw="$RELISS_DATA_ROOT/inputs"
export nnUNet_preprocessed="$RELISS_DATA_ROOT/cache"
export nnUNet_results="$RELISS_DATA_ROOT/weights"
export PYTHONPATH="$reliss_project_root${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_compile=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
