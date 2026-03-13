#!/usr/bin/env bash
# ============================================================
# MemoryBank Baseline — Full Pipeline Runner
# ============================================================
# Runs the full 5-step pipeline on a small subset (3 personas)
# with 80-20% train/test split.
#
# Usage:
#   bash run_pipeline.sh
#
# To switch from SUBSET → FULL dataset:
#   In convert_to_memorybank_format.py, set N_PERSONAS = None
# ============================================================

set -e  # exit on any error

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

mkdir -p output
LOG_FILE="output/output.log"

{
echo "============================================================"
echo "  MemoryBank Baseline Pipeline"
echo "  Model: BERT-base-uncased (embeddings) + Qwen 2.5 3B (LLM)"
echo "  Dataset: 100% History & Inference Overlap"
echo "  Logs saved to: $LOG_FILE"
echo "============================================================"
echo ""

# ── Step 1: Convert sorted_conversations.json → memory.json ──
echo "[Step 1/5] Converting sorted_conversations.json → memory.json ..."
python convert_to_memorybank_format.py
echo ""

# ── Step 2: Summarize memory using Qwen 2.5 3B ───────────────
# NOTE: This step uses the GPU. Pre-seeded year_summary fields
#       from the json will be skipped (already done).
echo "[Step 2/5] Summarizing memory with Qwen 2.5 3B ..."
python summarize_memory.py
echo ""

# ── Step 3: Build BERT + FAISS index ─────────────────────────
echo "[Step 3/5] Building BERT+FAISS memory index ..."
python build_memory_index.py
echo ""

# ── Step 4: Run inference ─────────────────────────────────────
echo "[Step 4/6] Running MemoryBank inference ..."
python run_inference.py --top_k 3
echo ""

# ── Step 5: Convert JSON to CSV for evaluation ───────────────
echo "[Step 5/6] Converting inference_results.json → CSV ..."
python json_to_csv.py
echo ""

# ── Step 6: Evaluate ─────────────────────────────────────────
echo "[Step 6/6] Evaluating results (BLEU / ROUGE / BERTScore) ..."
python evaluation.py
echo ""

echo "============================================================"
echo "  Pipeline complete!"
echo "  Results: $BASE_DIR/output/inference_results.json"
echo "  CSV Data: $BASE_DIR/output/inference_results.csv"
echo "  Metrics: $BASE_DIR/output/evaluation.csv"
echo "============================================================"
} 2>&1 | tee -a "$LOG_FILE"
