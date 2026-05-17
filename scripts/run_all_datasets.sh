#!/usr/bin/env bash
set -u
set -o pipefail

# Usage:
#   DRY_RUN=1 ROOT=/path/to/datasets PYTHON=python3 ./scripts/run_all_datasets.sh
#
# By default this uses ROOT=/home/prakh/projects/CoOp/datasets_root

DRY_RUN=${DRY_RUN:-0}
ROOT=${ROOT:-/home/prakh/projects/CoOp/datasets_root}
PYTHON=${PYTHON:-python}

TRAINER_COOP_CONFIG="configs/trainers/CoOp/rn50_ep50.yaml"
TRAINER_GRAPH_CONFIG="configs/trainers/CoOp/rn50_ep50_graph.yaml"

DATASETS=(oxford_pets dtd caltech101 oxford_flowers fgvc_aircraft ucf101)
MULTI_SHOTS=(1 2 4 8 16)

run_cmd() {
  local -a cmd=("$@")
  echo "+ ${cmd[*]}"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  "${cmd[@]}"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "Warning: command failed with exit code $rc — continuing"
  fi
  return $rc
}

echo "Starting runs for datasets: ${DATASETS[*]}"

for ds in "${DATASETS[@]}"; do
  ds_cfg="configs/datasets/${ds}.yaml"
  if [ ! -f "$ds_cfg" ]; then
    echo "Skipping $ds — dataset config not found at $ds_cfg"
    continue
  fi

  echo "\n=== $ds: baseline CoOp (16 shots) ==="
  run_cmd $PYTHON train.py --root "$ROOT" --trainer CoOp --dataset-config-file "$ds_cfg" --config-file "$TRAINER_COOP_CONFIG" --output-dir "output/baseline/${ds}/16shots" DATASET.NUM_SHOTS 16 TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC False TRAINER.COOP.CLASS_TOKEN_POSITION end

  echo "\n--- $ds: graph variant — smooth_only (16 shots)"
  run_cmd $PYTHON train.py --root "$ROOT" --trainer CoOpGraph --dataset-config-file "$ds_cfg" --config-file "$TRAINER_GRAPH_CONFIG" --output-dir "output/graph/${ds}/smooth_only/16shots" DATASET.NUM_SHOTS 16 TRAINER.COOP.ALPHA_SMOOTH 0.1 TRAINER.COOP.LAMBDA_LAP 0.0 TRAINER.COOP.USE_GCN False

  echo "\n--- $ds: graph variant — lap_only (16 shots)"
  run_cmd $PYTHON train.py --root "$ROOT" --trainer CoOpGraph --dataset-config-file "$ds_cfg" --config-file "$TRAINER_GRAPH_CONFIG" --output-dir "output/graph/${ds}/lap_only/16shots" DATASET.NUM_SHOTS 16 TRAINER.COOP.ALPHA_SMOOTH 0.0 TRAINER.COOP.LAMBDA_LAP 0.01 TRAINER.COOP.USE_GCN False

  echo "\n--- $ds: graph variant — smooth_lap (shots: ${MULTI_SHOTS[*]})"
  for SHOTS in "${MULTI_SHOTS[@]}"; do
    run_cmd $PYTHON train.py --root "$ROOT" --trainer CoOpGraph --dataset-config-file "$ds_cfg" --config-file "$TRAINER_GRAPH_CONFIG" --output-dir "output/graph/${ds}/smooth_lap/${SHOTS}shots" DATASET.NUM_SHOTS ${SHOTS} TRAINER.COOP.ALPHA_SMOOTH 0.1 TRAINER.COOP.LAMBDA_LAP 0.01 TRAINER.COOP.USE_GCN False
  done

  echo "\n--- $ds: graph variant — full_graph (16 shots)"
  run_cmd $PYTHON train.py --root "$ROOT" --trainer CoOpGraph --dataset-config-file "$ds_cfg" --config-file "$TRAINER_GRAPH_CONFIG" --output-dir "output/graph/${ds}/full_graph/16shots" DATASET.NUM_SHOTS 16 TRAINER.COOP.ALPHA_SMOOTH 0.1 TRAINER.COOP.LAMBDA_LAP 0.01 TRAINER.COOP.USE_GCN True

done

echo "\nAll dataset jobs processed."
