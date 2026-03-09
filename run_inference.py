"""
Step 5: Run MemoryBank Inference using Qwen 2.5 3B Instruct
============================================================
Mirrors MemoryBank-SiliconFriend/SiliconFriend-ChatGPT/cli_llamaindex.py but with:
  - BERT-base-uncased + FAISS for memory retrieval  (instead of LlamaIndex)
  - Qwen 2.5 3B Instruct for response generation    (instead of OpenAI GPT)

For each persona's QUERY conversations (from query_set.json):
  - Iterates through ALL User?Agent turn pairs in each conversation
  - For EACH User turn:
    1. Retrieves top-K related memories from FAISS index strictly BEFORE the conversation date
    2. Builds a system prompt with:
         - overall_history     (past summaries strictly before the conversation date)
         - overall_personality (user traits + recommended agent strategy)
         - retrieved_memories  (specific relevant past interactions)
    3. Generates the next agent response using Qwen 2.5 3B
  - Saves results to output/inference_results.json

CLI:
  python run_inference.py                       # all personas
  python run_inference.py --persona_id P_001    # single persona
  python run_inference.py --top_k 5             # retrieve top 5 memories

Output format (inference_results.json):
{
  "P_001": [
    {
      "conversation_id": 10,
      "date": "2024-xx-xx",
      "turns": [
        {
          "turn_index": 0,
          "user_query": "User utterance...",
          "ground_truth_response": "Actual agent response...",
          "retrieved_memories": [{"text": ..., "date": ..., "score": ...}],
          "generated_response": "Qwen-generated agent response..."
        },
        ...
      ]
    }
  ]
}
"""

import json
import os
import torch
import argparse
import math
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_retrieval import BERTMemoryRetrieval

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE    = os.path.join(SCRIPT_DIR, "memory_bank", "memory.json")
QUERY_FILE     = os.path.join(SCRIPT_DIR, "memory_bank", "query_set.json")
INDEX_DIR      = os.path.join(SCRIPT_DIR, "memory_bank", "faiss_index")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "output", "session_split_v1")
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "inference_results.json")
OUTPUT_CSV     = os.path.join(OUTPUT_DIR, "inference_results.csv")

QWEN_MODEL     = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL= "bert-base-uncased"
TOP_K          = 3
MAX_NEW_TOKENS = 400
# ---------------------------------------------


# -- System Prompt Template ----------------------------------------------------
# Mirrors MemoryBank-SiliconFriend/utils/prompt_utils.py's meta_prompt

SYSTEM_PROMPT_WITH_MEMORY = """\
You are a professional insurance sales agent. You assist users in finding the \
best motor insurance policies based on their needs and budget.

You have access to this user's past interaction history:

[USER PROFILE]
{overall_personality}

[PAST INTERACTIONS SUMMARY]
{overall_history}

[MOST RELEVANT PAST MEMORIES]
The following are the most relevant past conversations recalled for this query:
{retrieved_memories}

Based on the user's profile, history, and retrieved memories, provide a helpful, \
personalized, and empathetic insurance recommendation. Refer to past context where relevant.\
"""

SYSTEM_PROMPT_NO_MEMORY = """\
You are a professional insurance sales agent. You assist users in finding the \
best motor insurance policies based on their needs and budget.
Provide a helpful, personalized, and empathetic insurance recommendation.\
"""


def load_qwen_model(model_name=QWEN_MODEL):
    print(f"Loading Qwen model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    print("  Model loaded.\n")
    return model, tokenizer


def qwen_generate(model, tokenizer, system_msg, user_msg, max_new_tokens=MAX_NEW_TOKENS):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None
        )
    new_tokens = output_ids[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def format_retrieved_memories(retrieved):
    """Format retrieved memory list into a readable string for the prompt."""
    if not retrieved:
        return "No specific past memories retrieved."
    lines = []
    for i, r in enumerate(retrieved, 1):
        lines.append(f"  [{i}] (Date: {r['date']}, Relevance: {r['score']:.3f})")
        lines.append(f"      {r['text'][:400].strip()}")
    return "\n".join(lines)


def get_all_qa_pairs(turns):
    """
    Extract ALL (User utterance, Agent response) pairs from a multi-turn conversation.

    Given turns like:
      [User, Agent, User, Agent, User, Agent, ...]
    Returns:
      [(user_q1, agent_r1), (user_q2, agent_r2), ...]

    If the conversation ends with a User turn (no trailing Agent), the response
    is set to empty string.
    """
    pairs = []
    i = 0
    while i < len(turns):
        if turns[i].get("speaker") == "User":
            user_utt = turns[i].get("utterance", "").strip()
            # Look for the next Agent turn immediately following
            if i + 1 < len(turns) and turns[i + 1].get("speaker") == "Agent":
                agent_utt = turns[i + 1].get("utterance", "").strip()
                i += 2
            else:
                agent_utt = ""  # No agent follow-up
                i += 1
            if user_utt:
                pairs.append((user_utt, agent_utt))
        else:
            i += 1
    return pairs


def run_inference_for_persona(pid, persona_mem, query_convs, retriever, model, tokenizer, top_k=TOP_K, query_start_date="9999-99-99"):
    """
    Run the full MemoryBank baseline inference for one persona.
    Processes ALL User?Agent turn pairs in each query conversation.

    Returns list of result dicts (one per query conversation, each with all turns).
    """
    # Load persona's FAISS index
    try:
        faiss_index, texts, dates = retriever.load_index(pid)
        has_index = True
    except FileNotFoundError as e:
        print(f"  [WARNING] {e}")
        print(f"  Falling back to no-memory generation for {pid}.")
        has_index = False

    overall_personality = persona_mem.get("overall_personality", "No personality profile available.")

    results = []
    for conv in query_convs:
        conv_id = conv["conversation_id"]
        date    = conv["date"]
        turns   = conv["turns"]

        # -- Dynamically build overall history strictly up to 'query_start_date' --
        past_histories = []
        for d, sum_data in sorted(persona_mem.get("summary", {}).items()):
            if d < query_start_date and sum_data.get("content"):
                past_histories.append(f"[{d}] {sum_data['content']}")
        
        if past_histories:
            overall_history = "\n".join(past_histories)
        else:
            overall_history = "No prior conversation history available."

        # -- Extract ALL User?Agent pairs from this conversation ------
        qa_pairs = get_all_qa_pairs(turns)
        if not qa_pairs:
            print(f"  [SKIP] Conv {conv_id}: no User turns found.")
            continue

        print(f"\n  Conv {conv_id} [{date}]  |  {len(qa_pairs)} user turns total")

        turn_results = []
        for turn_idx, (user_query, ground_truth) in enumerate(qa_pairs):
            print(f"    Turn {turn_idx+1}/{len(qa_pairs)}: {user_query[:80]}...")

            # -- Retrieve relevant memories for THIS user turn --------
            # Restrict retrieval to dates strictly before query_start_date
            if has_index:
                retrieved = retriever.search(user_query, faiss_index, texts, dates, top_k=top_k, max_date=query_start_date)
            else:
                retrieved = []

            retrieved_str = format_retrieved_memories(retrieved)

            # -- Build system prompt ----------------------------------
            if has_index and retrieved:
                system_prompt = SYSTEM_PROMPT_WITH_MEMORY.format(
                    overall_personality=overall_personality,
                    overall_history=overall_history,
                    retrieved_memories=retrieved_str
                )
            else:
                system_prompt = SYSTEM_PROMPT_NO_MEMORY

            # -- Generate response with Qwen --------------------------
            print(f"    Generating ...", end=" ", flush=True)
            generated = qwen_generate(model, tokenizer, system_prompt, user_query)
            print(f"done. [{generated[:80]}...]")

            turn_results.append({
                "turn_index"            : turn_idx,
                "user_query"            : user_query,
                "ground_truth_response" : ground_truth,
                "retrieved_memories"    : retrieved,
                "generated_response"    : generated
            })

        results.append({
            "conversation_id" : conv_id,
            "date"            : date,
            "num_turns"       : len(qa_pairs),
            "turns"           : turn_results
        })
        print(f"  [OK] Conv {conv_id}: {len(turn_results)} turns processed.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run MemoryBank inference with Qwen 2.5 3B")
    parser.add_argument("--persona_id", type=str, default=None,
                        help="Run only for this persona. Default: all.")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--memory_file", type=str, default=MEMORY_FILE)
    parser.add_argument("--query_file",  type=str, default=QUERY_FILE)
    parser.add_argument("--index_dir",   type=str, default=INDEX_DIR)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Load memory + query data -------------------------------------
    print(f"Loading memory: {args.memory_file}")
    with open(args.memory_file, "r", encoding="utf-8") as f:
        memory_dict = json.load(f)

    print(f"Loading query set: {args.query_file}")
    with open(args.query_file, "r", encoding="utf-8") as f:
        query_dict = json.load(f)

    personas = [args.persona_id] if args.persona_id else list(memory_dict.keys())
    print(f"\nRunning inference for personas: {personas}")

    # -- Load models -------------------------------------------------
    retriever = BERTMemoryRetrieval(model_name=EMBEDDING_MODEL)
    model, tokenizer = load_qwen_model()

    # -- Run per-persona inference ------------------------------------
    all_results = {}
    for pid in personas:
        if pid not in memory_dict:
            print(f"[SKIP] {pid} not in memory.json")
            continue
        if pid not in query_dict:
            print(f"[SKIP] {pid} not in query_set.json")
            continue

        print(f"\n{'='*60}")
        print(f"  Persona: {pid}  |  {len(query_dict[pid])} query conversations")
        print(f"{'='*60}")

        # Identify the 80/20 session-level split to find query_start_date
        persona_mem = memory_dict[pid]
        distinct_sessions = sorted(persona_mem.get("history", {}).keys())
        num_sessions = len(distinct_sessions)
        num_query = math.ceil(0.2 * num_sessions)
        num_history = num_sessions - num_query
        
        if num_history > 0 and num_history < num_sessions:
            query_start_date = distinct_sessions[num_history]
            print(f"  Split: {num_history} history sessions, {num_query} query sessions. Query Start Date: {query_start_date}")
        else:
            query_start_date = "9999-99-99"
            print(f"  Split: Using all as history.")

        persona_results = run_inference_for_persona(
            pid,
            persona_mem,
            query_dict[pid],
            retriever,
            model,
            tokenizer,
            top_k=args.top_k,
            query_start_date=query_start_date
        )
        all_results[pid] = persona_results
        print(f"\n  [OK] {pid}: {len(persona_results)} conversations processed.")

        # -- Save results on the go for this persona ------------------
        if os.path.exists(args.output_file):
            print(f"    Appending to existing results in {args.output_file}...")
            try:
                with open(args.output_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
            except Exception:
                existing_results = {}
            if pid not in existing_results:
                existing_results[pid] = persona_results
            else:
                existing_results[pid].extend(persona_results)
            final_results = existing_results
        else:
            final_results = all_results.copy()

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    # -- Final Save: Flatten to CSV -------------------------
    print(f"\nFinalizing outputs ...")
    csv_rows = []
    for pid, persona_convs in all_results.items():
        for conv in persona_convs:
            conv_id = conv["conversation_id"]
            for turn in conv["turns"]:
                csv_rows.append({
                    "Persona Id": pid,
                    "conversation id": conv_id,
                    "turn index": turn["turn_index"],
                    "ground response": turn["ground_truth_response"],
                    "generated response": turn["generated_response"]
                })
    
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"   CSV results saved ? {OUTPUT_CSV}")

    print(f"\n{'='*60}")
    print(f"[OK] Inference complete!")
    print(f"   Results saved ? {args.output_file}")
    print(f"   Next step: run evaluate.py to compute BLEU / BERTScore")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
