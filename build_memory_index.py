"""
Step 3: Build BERT + FAISS Memory Index for all personas
=========================================================
Mirrors MemoryBank-SiliconFriend/memory_bank/build_memory_index.py but uses
BERT-base-uncased + FAISS (via memory_retrieval.py) instead of LlamaIndex.

Run this AFTER:
  1. convert_to_memorybank_format.py   → memory.json exists
  2. summarize_memory.py               → overall_history / overall_personality filled

For each persona in memory.json:
  - Converts history + summaries into memory document chunks
  - Embeds all chunks using BERT-base-uncased
  - Builds a FAISS IndexFlatIP (cosine similarity)
  - Saves index to memory_bank/faiss_index/<persona_id>/

Usage:
  python build_memory_index.py                     # all personas
  python build_memory_index.py --persona_id P_001  # single persona
"""

import json
import os
import argparse

from memory_retrieval import BERTMemoryRetrieval, build_memory_docs

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MEMORY_FILE = "/DATA/rohan_kirti/niladri2/baselines/MemoryBank-Baseline/memory_bank/memory.json"
INDEX_DIR   = "/DATA/rohan_kirti/niladri2/baselines/MemoryBank-Baseline/memory_bank/faiss_index"
EMBEDDING_MODEL = "bert-base-uncased"
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Build BERT+FAISS memory index")
    parser.add_argument("--persona_id", type=str, default=None,
                        help="Build index only for this persona. Default: all.")
    parser.add_argument("--memory_file", type=str, default=MEMORY_FILE)
    parser.add_argument("--index_dir",   type=str, default=INDEX_DIR)
    args = parser.parse_args()

    # ── Load memory ──────────────────────────────────────────────────
    print(f"Loading {args.memory_file} ...")
    with open(args.memory_file, "r", encoding="utf-8") as f:
        memory_dict = json.load(f)

    personas = [args.persona_id] if args.persona_id else list(memory_dict.keys())
    print(f"Building FAISS index for personas: {personas}")

    # ── Initialize retriever (loads BERT model once) ────────────────
    retriever = BERTMemoryRetrieval(model_name=EMBEDDING_MODEL)

    # ── Build index per persona ──────────────────────────────────────
    for pid in personas:
        if pid not in memory_dict:
            print(f"  [SKIP] {pid} not found in memory.json")
            continue

        persona_mem = memory_dict[pid]
        history = persona_mem.get("history", {})
        if not history:
            print(f"  [SKIP] {pid} has no history entries.")
            continue

        print(f"\n{'='*60}")
        print(f"  Building index for: {pid}  ({len(history)} dates in history)")
        print(f"{'='*60}")

        # Convert to memory document chunks
        docs = build_memory_docs(persona_mem, pid)
        print(f"  Created {len(docs)} memory document chunks.")

        if not docs:
            print(f"  [SKIP] No documents to index for {pid}")
            continue

        # Build and save FAISS index
        save_dir = retriever.build_and_save_index(pid, docs, index_dir=args.index_dir)
        print(f"  ✅ Index for {pid} saved at: {save_dir}")

    print(f"\n{'='*60}")
    print(f"  Done! FAISS indices saved under: {args.index_dir}")
    print(f"  Next step: run run_inference.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
