#!/usr/bin/env bash
# ============================================================
# MemoryBank Baseline — OpenAI-Compatible Pipeline Runner
# ============================================================
# Runs the full pipeline with an OpenAI-compatible chat model
# and stores all artifacts in a separate run directory.
#
# Usage:
#   bash run_pipeline_openai.sh
#   OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... bash run_pipeline_openai.sh
# ============================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:-BRK3HINQGTBXH5FYUGQ4XO63MZJ32FCIL6WQ}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.vultrinference.com/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-zai-org/GLM-5.1-FP8}"

RUN_NAME="${RUN_NAME:-openai_glm_5_1_fp8}"
RUN_DIR="$BASE_DIR/runs/$RUN_NAME"
MEMORY_DIR="$RUN_DIR/memory_bank"
OUTPUT_DIR="$RUN_DIR/output"
LOG_FILE="$OUTPUT_DIR/output_openai.log"

mkdir -p "$MEMORY_DIR" "$OUTPUT_DIR"

{
echo "============================================================"
echo "  MemoryBank Baseline Pipeline"
echo "  Model: all-MiniLM-L6-v2 (embeddings) + $OPENAI_MODEL (LLM)"
echo "  Backend: OpenAI-compatible API"
echo "  Base URL: $OPENAI_BASE_URL"
echo "  Run folder: $RUN_DIR"
echo "  Logs saved to: $LOG_FILE"
echo "============================================================"
echo ""

echo "[Step 1/6] Converting sorted_conversations.json ..."
python convert_to_memorybank_format.py \
  --output_dir "$MEMORY_DIR"
echo ""

echo "[Step 2/6] Summarizing memory with $OPENAI_MODEL ..."
python summarize_memory.py \
  --backend openai \
  --model_name "$OPENAI_MODEL" \
  --base_url "$OPENAI_BASE_URL" \
  --memory_file "$MEMORY_DIR/memory.json"
echo ""

echo "[Step 3/6] Building MiniLM+FAISS memory index ..."
python build_memory_index.py \
  --memory_file "$MEMORY_DIR/memory.json" \
  --index_dir "$MEMORY_DIR/faiss_index"
echo ""

echo "[Step 4/6] Running MemoryBank inference ..."
python run_inference.py \
  --backend openai \
  --model_name "$OPENAI_MODEL" \
  --base_url "$OPENAI_BASE_URL" \
  --memory_file "$MEMORY_DIR/memory.json" \
  --query_file "$MEMORY_DIR/query_set.json" \
  --index_dir "$MEMORY_DIR/faiss_index" \
  --output_file "$OUTPUT_DIR/inference_results.json" \
  --top_k 3
echo ""

echo "[Step 5/6] Converting inference_results.json → CSV ..."
python json_to_csv.py \
  --input_json "$OUTPUT_DIR/inference_results.json" \
  --output_csv "$OUTPUT_DIR/inference_results.csv"
echo ""

echo "[Step 6/6] Evaluating results (BLEU / ROUGE / BERTScore / Distinct / METEOR) ..."
python evaluation.py \
  --input_csv "$OUTPUT_DIR/inference_results.csv" \
  --output_csv "$OUTPUT_DIR/evaluation.csv" \
  --skip_ppl
echo ""

echo "============================================================"
echo "  Pipeline complete!"
echo "  Memory:  $MEMORY_DIR/memory.json"
echo "  Results: $OUTPUT_DIR/inference_results.json"
echo "  CSV:     $OUTPUT_DIR/inference_results.csv"
echo "  Metrics: $OUTPUT_DIR/evaluation.csv"
echo "============================================================"
} 2>&1 | tee -a "$LOG_FILE"
