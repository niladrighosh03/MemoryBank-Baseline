"""
Step 2: Summarize Memory using a configurable chat model
========================================================
Mirrors MemoryBank-SiliconFriend/memory_bank/summarize_memory.py but supports
either a local HuggingFace model or an OpenAI-compatible chat endpoint.

For each persona, for each date in memory.history:
  - If summary is missing → generate event/topic summary
  - If personality is missing → generate personality analysis

Then generates:
  - overall_history   : condensed summary across ALL dates
  - overall_personality: holistic personality profile

NOTE: If year_summary was pre-seeded (from convert_to_memorybank_format.py),
this script will SKIP that date's summarization (already done).
Set FORCE_RESUMMARY = True to overwrite even pre-seeded entries.
"""

import json
import os
import argparse

from llm_backend import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_TEMPERATURE,
    DEFAULT_QWEN_MODEL,
    load_chat_backend,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(SCRIPT_DIR, "memory_bank", "memory.json")
MODEL_NAME = DEFAULT_QWEN_MODEL
MAX_NEW_TOKENS_SUMMARY = 300
MAX_NEW_TOKENS_OVERALL = 400
FORCE_RESUMMARY = False  # Set True to overwrite pre-seeded summaries
# ─────────────────────────────────────────────


# ── Prompt builders (from MemoryBank-SiliconFriend prompt logic) ──────────────

SYSTEM_SUMMARIZER = (
    "You are an AI assistant that summarizes conversation content "
    "concisely and extracts key information."
)

def build_event_summary_prompt(date, qa_pairs, user_name="User", agent_name="Agent"):
    """Prompt to summarize what happened in a conversation on a given date."""
    prompt = (
        f"Please summarize the following conversation on {date} as concisely "
        f"as possible. Extract the main topics discussed and key outcomes. "
        f"If there are multiple key events, summarize them separately.\n\n"
        f"Conversation on {date}:\n"
    )
    for pair in qa_pairs:
        prompt += f"{user_name}: {pair['query'].strip()}\n"
        prompt += f"{agent_name}: {pair['response'].strip()}\n"
    prompt += "\nSummary:"
    return prompt


def build_personality_prompt(date, qa_pairs, user_name="User", agent_name="Agent"):
    """Prompt to infer user personality from a conversation."""
    prompt = (
        f"Based on the following conversation on {date}, "
        f"please summarize the user's personality traits, preferences, "
        f"and the most appropriate response strategy for the agent.\n\n"
        f"Conversation:\n"
    )
    for pair in qa_pairs:
        prompt += f"{user_name}: {pair['query'].strip()}\n"
        prompt += f"{agent_name}: {pair['response'].strip()}\n"
    prompt += f"\n{user_name}'s personality traits, preferences and agent response strategy:"
    return prompt


def build_overall_history_prompt(summaries):
    """Prompt to create an overall history from per-date summaries."""
    prompt = (
        "Please provide a highly concise summary of the following events across "
        "multiple conversations. Capture essential key information:\n\n"
    )
    for date, summary_dict in summaries:
        content = summary_dict["content"]
        prompt += f"[{date}]: {content.strip()}\n"
    prompt += "\nOverall summary:"
    return prompt


def build_overall_personality_prompt(personalities):
    """Prompt to create an overall personality profile from per-date entries."""
    prompt = (
        "The following are the user's personality traits and preferences "
        "across multiple conversations:\n\n"
    )
    for date, personality_text in personalities:
        prompt += f"[{date}]: {personality_text.strip()}\n"
    prompt += (
        "\nPlease provide a highly concise overall profile of this user's "
        "personality and the most appropriate agent response strategy. Summary:"
    )
    return prompt


# ── Main summarization logic ─────────────────────────────────────────────────

def summarize_memory(memory_dict, llm, persona_id=None):
    """
    For each persona (or a specific one if persona_id is given),
    fill in missing summary/personality fields using the configured backend.
    """
    for pid, persona_mem in memory_dict.items():
        if persona_id and pid != persona_id:
            continue

        print(f"\n{'='*60}")
        print(f"  Processing persona: {pid}")
        print(f"{'='*60}")

        history = persona_mem.get("history", {})
        if not history:
            print(f"  [SKIP] No history for {pid}")
            continue

        if "summary" not in persona_mem:
            persona_mem["summary"] = {}
        if "personality" not in persona_mem:
            persona_mem["personality"] = {}

        for date, qa_pairs in history.items():
            if not qa_pairs:
                continue

            # ── Event summary ──────────────────────────────────
            already_summarized = (
                date in persona_mem["summary"] and
                persona_mem["summary"][date].get("content", "").strip()
            )
            if already_summarized and not FORCE_RESUMMARY:
                print(f"  [{date}] Summary: already pre-seeded, skipping.")
            else:
                print(f"  [{date}] Generating event summary ...", end=" ", flush=True)
                prompt = build_event_summary_prompt(date, qa_pairs)
                summary_text = llm.generate(
                    SYSTEM_SUMMARIZER,
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS_SUMMARY,
                )
                persona_mem["summary"][date] = {"content": summary_text}
                print(f"done. [{summary_text[:80]}...]")

            # ── Personality analysis ─────────────────────────────
            already_personality = (
                date in persona_mem["personality"] and
                persona_mem["personality"][date].strip()
            )
            if already_personality and not FORCE_RESUMMARY:
                print(f"  [{date}] Personality: already pre-seeded, skipping.")
            else:
                print(f"  [{date}] Generating personality analysis ...", end=" ", flush=True)
                prompt = build_personality_prompt(date, qa_pairs)
                personality_text = llm.generate(
                    SYSTEM_SUMMARIZER,
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS_SUMMARY,
                )
                persona_mem["personality"][date] = personality_text
                print(f"done. [{personality_text[:80]}...]")

        # ── Overall history ──────────────────────────────────────
        print(f"\n  [{pid}] Generating overall_history ...", end=" ", flush=True)
        summary_items = list(persona_mem["summary"].items())
        overall_history_prompt = build_overall_history_prompt(summary_items)
        persona_mem["overall_history"] = llm.generate(
            SYSTEM_SUMMARIZER,
            overall_history_prompt,
            max_new_tokens=MAX_NEW_TOKENS_OVERALL,
        )
        print(f"done.")

        # ── Overall personality ──────────────────────────────────
        print(f"  [{pid}] Generating overall_personality ...", end=" ", flush=True)
        personality_items = list(persona_mem["personality"].items())
        overall_personality_prompt = build_overall_personality_prompt(personality_items)
        persona_mem["overall_personality"] = llm.generate(
            SYSTEM_SUMMARIZER,
            overall_personality_prompt,
            max_new_tokens=MAX_NEW_TOKENS_OVERALL,
        )
        print(f"done.")

        print(f"\n  ✅ {pid}: overall_history and overall_personality generated.")

    return memory_dict


def main():
    parser = argparse.ArgumentParser(description="Summarize MemoryBank with a configurable chat backend")
    parser.add_argument("--persona_id", type=str, default=None,
                        help="Process only this persona (e.g., P_001). Default: all.")
    parser.add_argument("--memory_file", type=str, default=MEMORY_FILE,
                        help="Path to memory.json")
    parser.add_argument("--backend", type=str, choices=["qwen", "openai"], default="qwen",
                        help="Generation backend to use for summarization.")
    parser.add_argument("--model_name", type=str, default=None,
                        help=f"Model name to use. Defaults: qwen={DEFAULT_QWEN_MODEL}, openai={DEFAULT_OPENAI_MODEL}")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for the OpenAI-compatible backend. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_OPENAI_BASE_URL,
                        help="Base URL for the OpenAI-compatible backend.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_OPENAI_TEMPERATURE,
                        help="Sampling temperature for the OpenAI-compatible backend.")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable provider-side thinking mode when supported.")
    args = parser.parse_args()

    # Load memory
    print(f"Loading {args.memory_file} ...")
    with open(args.memory_file, "r", encoding="utf-8") as f:
        memory_dict = json.load(f)
    print(f"  Loaded {len(memory_dict)} personas: {list(memory_dict.keys())}")

    # Load model
    llm = load_chat_backend(
        backend=args.backend,
        model_name=args.model_name if args.model_name else None,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
    )

    # Summarize
    memory_dict = summarize_memory(memory_dict, llm, args.persona_id)

    # Save back
    with open(args.memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_dict, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved updated memory → {args.memory_file}")
    print("Next step: run build_memory_index.py")


if __name__ == "__main__":
    main()
