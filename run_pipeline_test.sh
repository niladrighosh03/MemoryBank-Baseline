#!/usr/bin/env bash
set -e  # exit on any error

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

OUTPUT_DIR="output/session_split_v1"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/output.log"

# Define interpreters and paths
# # Used during verification to solve Faiss/Library and dependency issues
# PY_INFERENCE="/DATA/rohan_kirti/miniconda3/bin/python"
# PY_EVAL="/usr/bin/python3.8"
# export LD_LIBRARY_PATH=/DATA/rohan_kirti/miniconda3/envs/llama_factory/lib:$LD_LIBRARY_PATH

{
# -- Step 4: Run inference -------------------------------------
echo "[Step 4/6] Running MemoryBank inference ..."
python run_inference.py --top_k 3
echo ""

# -- Step 6: Evaluate -----------------------------------------
echo "[Step 6/6] Evaluating results (BLEU / ROUGE / BERTScore) ..."
python evaluation.py
echo ""

echo "============================================================"
echo "  Pipeline complete!"
echo "  Inference Results: $BASE_DIR/$OUTPUT_DIR/inference_results.json"
echo "  Evaluation Results: $BASE_DIR/$OUTPUT_DIR/evaluation.csv"
echo "============================================================"
} 2>&1 | tee -a "$LOG_FILE"
