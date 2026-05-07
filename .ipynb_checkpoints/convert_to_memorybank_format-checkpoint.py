"""
Step 1: Convert sorted_conversations.json → MemoryBank memory.json format
==========================================================================
- Groups conversations by persona_id
- Sorts by timestamp (chronological order)
- Uses 100% of conversations for BOTH HISTORY (memory building) and QUERY (inference)
  (The inference script will dynamically filter memories to ensure only past dates are used)
- For SMALL SUBSET MODE: only processes the first N_PERSONAS personas

MemoryBank memory.json format per persona:
{
  "P_001": {
    "history": {
      "2015-01-11": [{"query": "...", "response": "..."}, ...],  # User/Agent turns
      "2016-09-28": [...],
      ...
    },
    "summary": {
      "2015-01-11": {"content": "..."}  # pre-seeded from year_summary
    },
    "personality": {
      "2015-01-11": "..."  # pre-seeded from personality + big_five_scores
    }
  }
}

Query split (for inference) is saved separately to query_set.json:
{
  "P_001": [
    {
      "conversation_id": N,
      "date": "2024-xx-xx",
      "turns": [...],
      "year_summary": "..."
    }
  ]
}
"""

import json
import os
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "sorted_conversations.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "memory_bank")
MEMORY_FILE = os.path.join(OUTPUT_DIR, "memory.json")
QUERY_FILE = os.path.join(OUTPUT_DIR, "query_set.json")

# For small subset: only use first N personas. Set to None to use ALL.
N_PERSONAS = None     # Process all personas
# ─────────────────────────────────────────────


def extract_date(timestamp_str):
    """Extract YYYY-MM-DD from timestamp like '2015-01-11T07:19:59'."""
    return timestamp_str.split("T")[0]


def turns_to_qa_pairs(turns):
    """
    Convert a list of turns [{"speaker": "User"/"Agent", "utterance": "..."}, ...]
    into MemoryBank format: [{"query": "...", "response": "..."}, ...]
    Pairs each consecutive User → Agent turn.
    """
    pairs = []
    i = 0
    while i < len(turns) - 1:
        user_turn = turns[i]
        agent_turn = turns[i + 1]
        if user_turn["speaker"] == "User" and agent_turn["speaker"] == "Agent":
            pairs.append({
                "query": user_turn["utterance"],
                "response": agent_turn["utterance"]
            })
            i += 2
        else:
            i += 1
    return pairs


def personality_to_text(conv):
    """Create a readable personality description from personality fields."""
    personality = conv.get("personality", "")
    income = conv.get("income_range", "")
    trait = conv.get("dominant_trait", "")
    big5 = conv.get("big_five_scores", {})
    big5_str = ", ".join([f"{k}={v:.1f}" for k, v in big5.items()]) if big5 else ""

    desc = (
        f"User personality: {personality}. "
        f"Income range: {income}. "
        f"Dominant trait: {trait}. "
        f"Big Five scores: {big5_str}."
    )
    return desc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────
    print(f"Loading {INPUT_FILE} ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_convs = json.load(f)
    print(f"  Loaded {len(all_convs)} conversations total.")

    # ── Group by persona_id ────────────────────────────────
    persona_convs = defaultdict(list)
    for conv in all_convs:
        persona_convs[conv["persona_id"]].append(conv)

    # Sort personas and optionally take subset
    all_personas = sorted(persona_convs.keys())
    if N_PERSONAS is not None:
        all_personas = all_personas[:N_PERSONAS]
        print(f"  SUBSET MODE: using only {N_PERSONAS} personas: {all_personas}")
    else:
        print(f"  Using all {len(all_personas)} personas.")

    # ── Build memory.json and query_set.json ───────────────
    memory_dict = {}
    query_dict = {}

    for persona_id in all_personas:
        convs = persona_convs[persona_id]

        # Sort chronologically by timestamp
        convs_sorted = sorted(convs, key=lambda c: c["timestamp"])
        
        # Group by year
        year_to_convs = defaultdict(list)
        for c in convs_sorted:
            year_to_convs[c.get("year")].append(c)
        
        sorted_years = sorted(year_to_convs.keys())
        n_years = len(sorted_years)
        
        # 80% (floor) for memory, rest for query
        n_memory_years = int(n_years * 0.8)
        
        memory_years = sorted_years[:n_memory_years]
        query_years = sorted_years[n_memory_years:]
        
        history_convs = [c for y in memory_years for c in year_to_convs[y]]
        query_convs   = [c for y in query_years for c in year_to_convs[y]]

        print(f"\n  Persona {persona_id}: {n_years} years total ({len(convs_sorted)} convs)")
        print(f"    Memory: {len(memory_years)} years ({len(history_convs)} convs) | Query: {len(query_years)} years ({len(query_convs)} convs)")

        # ──── Build HISTORY (memory.json format) ────
        persona_memory = {
            "history":     {},
            "summary":     {},
            "personality": {}
        }

        for conv in history_convs:
            date = extract_date(conv["timestamp"])
            qa_pairs = turns_to_qa_pairs(conv["turns"])
            if not qa_pairs:
                continue

            # history: raw dialogue pairs
            if date not in persona_memory["history"]:
                persona_memory["history"][date] = []
            persona_memory["history"][date].extend(qa_pairs)

            # summary: pre-seed from year_summary (skip LLM summarization later if present)
            year_summary = conv.get("year_summary", "").strip()
            if year_summary:
                persona_memory["summary"][date] = {"content": year_summary}

            # personality: pre-seed from personality fields
            persona_memory["personality"][date] = personality_to_text(conv)

        memory_dict[persona_id] = persona_memory

        # ──── Build QUERY set ────
        query_items = []
        for conv in query_convs:
            date = extract_date(conv["timestamp"])
            query_items.append({
                "conversation_id": conv["conversation_id"],
                "date": date,
                "year": conv.get("year"),
                "turns": conv["turns"],
                "year_summary": conv.get("year_summary", ""),
                "outcome": conv.get("outcome", ""),
                "personality": personality_to_text(conv)
            })
        query_dict[persona_id] = query_items

    # ── Save outputs ───────────────────────────────────────
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_dict, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved memory.json → {MEMORY_FILE}")
    print(f"   Personas in memory: {list(memory_dict.keys())}")
    for pid, pmem in memory_dict.items():
        print(f"   {pid}: {len(pmem['history'])} history dates, "
              f"{len(pmem['summary'])} pre-seeded summaries")

    with open(QUERY_FILE, "w", encoding="utf-8") as f:
        json.dump(query_dict, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved query_set.json → {QUERY_FILE}")
    for pid, qitems in query_dict.items():
        print(f"   {pid}: {len(qitems)} query conversations")

    print("\nDone! Next step: run summarize_memory.py")


if __name__ == "__main__":
    main()
