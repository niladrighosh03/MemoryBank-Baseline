"""
LLM-as-Judge evaluation for MemoryBank-Baseline.

Reads evaluation.csv (which already has automated metrics), calls the OpenAI
judge for every row, and writes the resulting g_score back as a new column.

Usage:
    python llm_as_judge_evaluation.py
"""

import csv
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent.resolve()
EVAL_CSV      = SCRIPT_DIR / "output" / "evaluation.csv"
OUTPUT_CSV    = SCRIPT_DIR / "output" / "evaluation_judge.csv"
INFER_CSV     = SCRIPT_DIR / "output" / "inference_results.csv"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL       = "gpt-4o-mini"
MAX_RETRIES = 3
RETRY_DELAY = 5   # seconds between retries

# ── Load API key ──────────────────────────────────────────────────────────────
load_dotenv(SCRIPT_DIR / ".env")
api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip("'\"")
if not api_key:
    raise SystemExit("No OPENAI_API_KEY found. Set it in .env or as an env var.")

client = OpenAI(api_key=api_key)

# ── Judge system prompt ───────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an impartial judge and your task is to evaluate whether a given agent response uses persuasion strategies. You are evaluating a conversation between a User and an Agent. The Agent's goal is to persuade the User to purchase a product.
Your task is to rate how effectively the Agent employs recognized persuasion strategies in its responses, on a scale from 1 to 5.
The persuasion strategies you are looking for are:

Logical Appeal: Uses facts, specifications, and rational arguments such as features, performance metrics, or ratings to convince the user logically.
Emotional Appeal: Attempts to influence the user by attending to their emotions or feelings, such as excitement, happiness, or sentimental value related to the product.
Credibility Appeal: Persuasion based on trust, brand reputation, or authority, emphasizing reliability or proven quality (e.g., highlighting that a product is from a well-known brand).
Persona-based Appeal: Persuasion tailored to the user's personality, preferences, or profile.
Personal Appeal: Focuses on general positive opinions or personal recommendations, often highlighting popularity, positive reviews, or overall satisfaction with the product.

SCORING SCALE (1–5)
5 – Strong Strategic Persuasion
The agent actively and clearly employs multiple persuasion strategies (two or more) in a well-integrated manner. The strategies are distinct, intentional, and well-executed. For example, the agent combines logical facts with emotional framing, or tailors the pitch to the user's persona while citing credibility signals. The persuasion feels natural and layered.
4 – Clear Strategic Persuasion
The agent clearly uses at least one persuasion strategy and may show traces of a second. The strategy is deliberate and effectively applied. For example, the agent provides concrete specs and ratings (Logical Appeal) or explicitly references brand reputation (Credibility Appeal). The usage is clear but may lack the depth or combination seen at level 5.
3 – Moderate Strategic Persuasion
The agent shows some use of persuasion strategies, but the application is surface-level or generic. For example, the agent mentions a feature without elaborating on why it matters, or makes a vague emotional statement without connecting it to the user's situation. The strategy is present but not strongly executed.
2 – Minimal Strategic Persuasion
The agent's response contains only weak or incidental traces of persuasion strategy. Any persuasive element feels unintentional or formulaic rather than strategic. For example, a passing mention of a product being "great" without supporting evidence or emotional depth. No clear effort to employ a recognized strategy.
1 – No Strategic Persuasion
The agent's response shows no identifiable use of any persuasion strategy. The response is purely informational, transactional, or off-topic. There is no attempt to appeal to logic, emotion, credibility, the user's persona, or personal recommendation.
Important: *** Must Return ONLY a single integer (1,2,3,4,5) ***. No explanation. No text, no formatting. Only the number.

"""
# ── Helper: build conversation history up to (but not including) current turn ─
def build_history(conv_rows: list[dict], current_idx: int) -> list[dict]:
    """Return OpenAI-style message list for all turns before current_idx."""
    history = []
    for i in range(current_idx):
        row = conv_rows[i]
        history.append({"role": "user",      "content": row["user query"]})
        history.append({"role": "assistant", "content": row["ground response"]})
    return history


def format_history_text(history: list[dict]) -> str:
    """Turn the message list into readable text for the eval prompt."""
    if not history:
        return "(No prior turns — this is the first message in the conversation.)"
    lines = []
    turn = 1
    for i in range(0, len(history), 2):
        lines.append(f"[Turn {turn}]")
        lines.append(f"User:  {history[i]['content']}")
        if i + 1 < len(history):
            lines.append(f"Agent: {history[i+1]['content']}")
        lines.append("")
        turn += 1
    return "\n".join(lines)


# ── Judge call ────────────────────────────────────────────────────────────────
def judge_response(history: list[dict], user_message: str, agent_response: str) -> int | None:
    """Call the LLM judge and return an integer score (1–5), or None on failure."""
    history_text = format_history_text(history)

    eval_prompt = f"""## Conversation History (ground-truth prior turns)
{history_text}
## Current User Message
{user_message}

## Agent Response to Evaluate
{agent_response}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": eval_prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            score = int(content)
            if score < 1 or score > 5:
                raise ValueError(f"Score {score} out of range 1–5")
            return score
        except (ValueError, TypeError) as e:
            print(f"    Parse error (attempt {attempt + 1}): {e!r}  raw={content!r}")
        except Exception as e:
            print(f"    API error  (attempt {attempt + 1}): {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation for MemoryBank outputs")
    parser.add_argument("--eval_csv", type=str, default=str(EVAL_CSV),
                        help="Path to evaluation.csv")
    parser.add_argument("--output_csv", type=str, default=str(OUTPUT_CSV),
                        help="Path to evaluation_judge.csv")
    args = parser.parse_args()

    eval_csv = Path(args.eval_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load evaluation.csv ─────────────────────────────────────────────
    print(f"Loading  {eval_csv} …")
    print(f"Output → {output_csv}")
    with open(eval_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        eval_fieldnames = reader.fieldnames or []
        eval_rows = list(reader)

    print(f"  {len(eval_rows)} rows loaded.")

    # Add g_score column if not already present
    if "g_score" not in eval_fieldnames:
        eval_fieldnames = list(eval_fieldnames) + ["g_score"]

    # ── 2. Group rows by (Persona Id, conversation id) for history building ──
    # Use the evaluation rows directly; ground response = ground truth for history
    conversations: dict[tuple, list[dict]] = {}
    for row in eval_rows:
        key = (row["Persona Id"], row["conversation id"])
        conversations.setdefault(key, []).append(row)

    # Sort each conversation by turn index
    for key in conversations:
        conversations[key].sort(key=lambda r: int(r["turn index"]))

    # ── 3. Score each row ──────────────────────────────────────────────────
    total = len(eval_rows)
    scored = 0
    failed = 0

    for conv_key, conv_rows in conversations.items():
        persona_id, conv_id = conv_key
        print(f"\nConversation  persona={persona_id}  conv_id={conv_id}  ({len(conv_rows)} turns)")

        for turn_idx, row in enumerate(conv_rows):
            turn_no = row["turn index"]
            print(f"  Turn {turn_no} … ", end="", flush=True)

            # Skip if already scored (resume support)
            existing = row.get("g_score", "").strip()
            if existing and existing.lower() not in ("", "none", "null"):
                print(f"already scored → {existing}")
                scored += 1
                continue

            history = build_history(conv_rows, turn_idx)
            score   = judge_response(history, row["user query"], row["generated response"])

            row["g_score"] = score if score is not None else "None"

            if score is not None:
                print(f"g_score = {score}")
                scored += 1
            else:
                print("FAILED (None)")
                failed += 1

    # ── 4. Write results to evaluation_judge.csv (original is never touched) ──
    print(f"\nWriting scores to {output_csv} …")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eval_fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)

    # ── 5. Summary ─────────────────────────────────────────────────────────
    all_scores = [
        int(r["g_score"])
        for r in eval_rows
        if str(r.get("g_score", "")).strip() not in ("", "None", "null")
    ]
    print(f"\n{'='*60}")
    print(f"Evaluation complete.")
    print(f"  Total rows   : {total}")
    print(f"  Scored       : {scored}")
    print(f"  Failed       : {failed}")
    if all_scores:
        print(f"  Average g_score : {sum(all_scores) / len(all_scores):.3f}")
        print(f"  Score distribution:")
        for s in range(1, 6):
            count = all_scores.count(s)
            pct   = count / len(all_scores) * 100
            print(f"    {s}: {count:4d}  ({pct:.1f}%)")
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()
