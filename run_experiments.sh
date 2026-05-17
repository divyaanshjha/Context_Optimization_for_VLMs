#!/bin/bash
# =============================================================================
# Graph-CoOp Full Experiment Script
# Runs 4-shot and 16-shot training for all 5 configurations
# across 6 datasets in increasing order of dataset size
#
# Datasets (ordered by training split size):
#   DTD          : 2,820 train images
#   OxfordPets   : 2,944 train images
#   FGVCAircraft : 3,334 train images
#   Flowers102   : 4,093 train images
#   Caltech101   : 4,128 train images
#   UCF101       : 7,639 train images
#
# Configurations per dataset per shot count:
#   1. baseline    : original CoOp (CoOp trainer)
#   2. smooth_only : alpha=0.1, lambda=0.0, gcn=False
#   3. lap_only    : alpha=0.0, lambda=0.01, gcn=False
#   4. smooth_lap  : alpha=0.1, lambda=0.01, gcn=False
#   5. full_graph  : alpha=0.1, lambda=0.01, gcn=True
#
# Total runs: 6 datasets × 2 shots × 5 configs = 60 runs
# =============================================================================

ROOT="/home/prakh/projects/CoOp/datasets_root"
GRAPH_CFG="configs/trainers/CoOp/rn50_ep50_graph.yaml"
BASE_CFG="configs/trainers/CoOp/rn50_ep50.yaml"
LOG_DIR="output/run_logs"
RESULTS_FILE="output/all_results.txt"

mkdir -p "$LOG_DIR"
echo "" > "$RESULTS_FILE"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Prints a clearly visible section header
print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# Prints a single-line progress marker
print_run() {
    echo ""
    echo "------------------------------------------------------------"
    echo "  RUNNING: Dataset=$1 | Shots=$2 | Config=$3"
    echo "  Output : $4"
    echo "------------------------------------------------------------"
}

# Extracts the final test accuracy from a log file and appends to results
extract_and_log() {
    DATASET=$1
    SHOTS=$2
    CONFIG=$3
    OUTPUT_DIR=$4
    LOG_FILE="$LOG_DIR/${DATASET}_${CONFIG}_${SHOTS}shot.txt"

    # CoOp logs accuracy as "* accuracy: XX.XX%"
    ACC=$(grep "accuracy:" "$LOG_FILE" | tail -1 | awk '{print $3}')

    if [ -z "$ACC" ]; then
        ACC="FAILED_OR_INCOMPLETE"
    fi

    echo "$DATASET | $SHOTS-shot | $CONFIG | $ACC" | tee -a "$RESULTS_FILE"
}

# Runs one training job. Args: dataset, cfg_file, trainer, output_dir,
#                               shots, alpha, lambda, gcn
run_job() {
    DATASET=$1
    CFG_FILE=$2
    TRAINER=$3
    OUTPUT_DIR=$4
    SHOTS=$5
    ALPHA=$6
    LAMBDA=$7
    GCN=$8
    LOG_FILE=$9

    mkdir -p "$OUTPUT_DIR"

    python train.py \
        --root "$ROOT" \
        --trainer "$TRAINER" \
        --dataset-config-file "configs/datasets/${DATASET}.yaml" \
        --config-file "$CFG_FILE" \
        --output-dir "$OUTPUT_DIR" \
        DATASET.NUM_SHOTS "$SHOTS" \
        TRAINER.COOP.ALPHA_SMOOTH "$ALPHA" \
        TRAINER.COOP.LAMBDA_LAP "$LAMBDA" \
        TRAINER.COOP.USE_GCN "$GCN" \
        2>&1 | tee "$LOG_FILE"

    # Check exit code
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Run failed. Check $LOG_FILE for details." \
            | tee -a "$RESULTS_FILE"
        return 1
    fi
    return 0
}

# =============================================================================
# DATASET AND SHOT CONFIGURATION
# =============================================================================

# Datasets in increasing order of training split size
DATASETS=("dtd" "oxford_pets" "fgvc_aircraft" "oxford_flowers" \
          "caltech101" "ucf101")

# Only 4-shot and 16-shot as requested
SHOTS_LIST=(4 16)

# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

TOTAL_RUNS=$((${#DATASETS[@]} * ${#SHOTS_LIST[@]} * 5))
CURRENT_RUN=0
START_TIME=$(date +%s)

echo "Starting experiment: $TOTAL_RUNS total runs"
echo "Start time: $(date)"
echo "Results will be written to: $RESULTS_FILE"
echo ""
echo "Dataset | Shots | Config | Accuracy" | tee -a "$RESULTS_FILE"
echo "--------|-------|--------|----------" | tee -a "$RESULTS_FILE"

for DATASET in "${DATASETS[@]}"; do
    for SHOTS in "${SHOTS_LIST[@]}"; do

        print_header "DATASET: $DATASET | SHOTS: $SHOTS"

        # ── 1. BASELINE ──────────────────────────────────────────────────────
        CURRENT_RUN=$((CURRENT_RUN + 1))
        CONFIG="baseline"
        OUT="output/graph/${DATASET}/${CONFIG}/${SHOTS}shots"
        LOG="$LOG_DIR/${DATASET}_${CONFIG}_${SHOTS}shot.txt"

        print_run "$DATASET" "$SHOTS" "$CONFIG" "$OUT"
        echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"

        python train.py \
            --root "$ROOT" \
            --trainer CoOp \
            --dataset-config-file "configs/datasets/${DATASET}.yaml" \
            --config-file "$BASE_CFG" \
            --output-dir "$OUT" \
            DATASET.NUM_SHOTS "$SHOTS" \
            TRAINER.COOP.N_CTX 16 \
            TRAINER.COOP.CSC False \
            TRAINER.COOP.CLASS_TOKEN_POSITION end \
            2>&1 | tee "$LOG"

        extract_and_log "$DATASET" "$SHOTS" "$CONFIG" "$OUT"

        # ── 2. GRAPH LABEL SMOOTHING ONLY ────────────────────────────────────
        CURRENT_RUN=$((CURRENT_RUN + 1))
        CONFIG="smooth_only"
        OUT="output/graph/${DATASET}/${CONFIG}/${SHOTS}shots"
        LOG="$LOG_DIR/${DATASET}_${CONFIG}_${SHOTS}shot.txt"

        print_run "$DATASET" "$SHOTS" "$CONFIG" "$OUT"
        echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"

        run_job "$DATASET" "$GRAPH_CFG" "CoOpGraph" "$OUT" \
                "$SHOTS" "0.1" "0.0" "False" "$LOG"

        extract_and_log "$DATASET" "$SHOTS" "$CONFIG" "$OUT"

        # ── 3. LAPLACIAN REGULARIZATION ONLY ─────────────────────────────────
        CURRENT_RUN=$((CURRENT_RUN + 1))
        CONFIG="lap_only"
        OUT="output/graph/${DATASET}/${CONFIG}/${SHOTS}shots"
        LOG="$LOG_DIR/${DATASET}_${CONFIG}_${SHOTS}shot.txt"

        print_run "$DATASET" "$SHOTS" "$CONFIG" "$OUT"
        echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"

        run_job "$DATASET" "$GRAPH_CFG" "CoOpGraph" "$OUT" \
                "$SHOTS" "0.0" "0.01" "False" "$LOG"

        extract_and_log "$DATASET" "$SHOTS" "$CONFIG" "$OUT"

        # ── 4. SMOOTHING + LAPLACIAN COMBINED ────────────────────────────────
        CURRENT_RUN=$((CURRENT_RUN + 1))
        CONFIG="smooth_lap"
        OUT="output/graph/${DATASET}/${CONFIG}/${SHOTS}shots"
        LOG="$LOG_DIR/${DATASET}_${CONFIG}_${SHOTS}shot.txt"

        print_run "$DATASET" "$SHOTS" "$CONFIG" "$OUT"
        echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"

        run_job "$DATASET" "$GRAPH_CFG" "CoOpGraph" "$OUT" \
                "$SHOTS" "0.1" "0.01" "False" "$LOG"

        extract_and_log "$DATASET" "$SHOTS" "$CONFIG" "$OUT"

        # ── 5. FULL GRAPH (WITH GCN) ──────────────────────────────────────────
        CURRENT_RUN=$((CURRENT_RUN + 1))
        CONFIG="full_graph"
        OUT="output/graph/${DATASET}/${CONFIG}/${SHOTS}shots"
        LOG="$LOG_DIR/${DATASET}_${CONFIG}_${SHOTS}shot.txt"

        print_run "$DATASET" "$SHOTS" "$CONFIG" "$OUT"
        echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"

        run_job "$DATASET" "$GRAPH_CFG" "CoOpGraph" "$OUT" \
                "$SHOTS" "0.1" "0.01" "True" "$LOG"

        extract_and_log "$DATASET" "$SHOTS" "$CONFIG" "$OUT"

    done
done

# =============================================================================
# FINAL SUMMARY
# =============================================================================

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================================"
echo "  ALL RUNS COMPLETE"
echo "  Total time: ${ELAPSED} minutes"
echo "  Results summary: $RESULTS_FILE"
echo "============================================================"
echo ""
echo "Final results table:"
cat "$RESULTS_FILE"
