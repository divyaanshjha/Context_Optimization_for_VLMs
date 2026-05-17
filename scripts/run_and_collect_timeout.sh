#!/usr/bin/env bash
set -u
set -o pipefail

# Run each training command once with a timeout to catch early errors.
# Adjust ROOT or PYTHON environment variables if needed.

ROOT=${ROOT:-/home/prakh/projects/CoOp/datasets_root}
PYTHON=${PYTHON:-python3}
TIMEOUT_CMD=${TIMEOUT_CMD:-timeout 300}

TRAINER_COOP_CONFIG="configs/trainers/CoOp/rn50_ep50.yaml"
TRAINER_GRAPH_CONFIG="configs/trainers/CoOp/rn50_ep50_graph.yaml"

DATASETS=(oxford_pets dtd caltech101 oxford_flowers fgvc_aircraft ucf101)
MULTI_SHOTS=(1 2 4 8 16)

mkdir -p run_logs

failures=()

run_and_log() {
  name="$1"
  shift
  cmd="$*"
  echo "=== RUN: $name ==="
  echo "CMD: $cmd"
  log="run_logs/${name}.log"
  # Run with timeout; capture combined stdout/stderr
  $TIMEOUT_CMD bash -lc "$cmd" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  echo "EXITCODE: $rc"
  if [ $rc -ne 0 ]; then
    failures+=("$name:$rc")
  fi
  echo "----TAIL----"
  tail -n 200 "$log" || true
  echo "----END----"
  echo
}

cd "$(dirname "$0")/.." || exit 1

for ds in "${DATASETS[@]}"; do
  ds_cfg="configs/datasets/${ds}.yaml"
  if [ ! -f "$ds_cfg" ]; then
    echo "SKIP $ds - missing $ds_cfg"
    continue
  fi

  run_and_log "${ds}_baseline_16" $PYTHON train.py --root "$ROOT" --trainer CoOp --dataset-config-file "$ds_cfg" --config-file "$TRAINER_COOP_CONFIG" --output-dir "output/baseline/${ds}/16shots" DATASET.NUM_SHOTS 16 TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC False TRAINER.COOP.CLASS_TOKEN_POSITION end OPTIM.MAX_EPOCH 1

  run_and_log "${ds}_smooth_only_16" $PYTHON train.py --root "$ROOT" --trainer CoOpGraph --dataset-config-file "$ds_cfg" --config-file "$TRAINER_GRAPH_CONFIG" --output-dir "output/graph/${ds}/smooth_only/16shots" DATASET.NUM_SHOTS 16 TRAINER.COOP.ALPHA_SMOOTH 0.1 TRAINER.COOP.LAMBDA_LAP 0.0 TRAINER.COOP.USE_GCN False OPTIM.MAX_EPOCH 1

  run_and_log "${ds}_lap_only_16" $PYTHON train.py --root "$ROOT" --trainer CoOpGraph --dataset-config-file "$ds_cfg" --config-file "$TRAINER_GRAPH_CONFIG" --output-dir "output/graph/${ds}/lap_only/16shots" DATASET.NUM_SHOTS 16 TRAINER.COOP.ALPHA_SMOOTH 0.0 TRAINER.COOP.LAMBDA_LAP 0.01 TRAINER.COOP.USE_GCN False OPTIM.MAX_EPOCH 1

  for SHOTS in "${MULTI_SHOTS[@]}"; do
    run_and_log "${ds}_smooth_lap_${SHOTS}shots" $PYTHON train.py --root "$ROOT" --trainer CoOpGraph --dataset-config-file "$ds_cfg" --config-file "$TRAINER_GRAPH_CONFIG" --output-dir "output/graph/${ds}/smooth_lap/${SHOTS}shots" DATASET.NUM_SHOTS ${SHOTS} TRAINER.COOP.ALPHA_SMOOTH 0.1 TRAINER.COOP.LAMBDA_LAP 0.01 TRAINER.COOP.USE_GCN False OPTIM.MAX_EPOCH 1
  done

  run_and_log "${ds}_full_graph_16" $PYTHON train.py --root "$ROOT" --trainer CoOpGraph --dataset-config-file "$ds_cfg" --config-file "$TRAINER_GRAPH_CONFIG" --output-dir "output/graph/${ds}/full_graph/16shots" DATASET.NUM_SHOTS 16 TRAINER.COOP.ALPHA_SMOOTH 0.1 TRAINER.COOP.LAMBDA_LAP 0.01 TRAINER.COOP.USE_GCN True OPTIM.MAX_EPOCH 1

done

echo
echo "SUMMARY: failures=${#failures[@]}"
for f in "${failures[@]}"; do
  echo "$f"
  tail -n 200 "run_logs/${f%%:*}.log" || true
  echo "----"
done

echo Done.
