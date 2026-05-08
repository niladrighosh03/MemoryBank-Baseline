# %%
import argparse
import os
import pandas as pd
import torch
import math
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV_PATH = os.path.join(SCRIPT_DIR, "output", "inference_results.csv")
OUTPUT_CSV_PATH = os.path.join(SCRIPT_DIR, "output", "evaluation.csv")

# --- Metric Definitions ---

def calculate_perplexity(text: str, model, tokenizer) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
    
    return math.exp(loss.item())

def compute_bleu2(candidate_sentence: str, reference_sentence: str) -> float:
    candidate_tokens = candidate_sentence.strip().split()
    reference_tokens = reference_sentence.strip().split()
    weights = (0.5, 0.5)
    smoothing = SmoothingFunction().method1
    
    return sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=weights,
        smoothing_function=smoothing
    )

def compute_bert_score_f1(candidates, references, lang="en", model_type="bert-base-uncased", verbose=False):
    P, R, F1 = score(candidates, references, lang=lang, model_type=model_type, verbose=verbose)
    return float(F1.tolist()[0])

def distinct_2(text):
    words = text.strip().split()
    if len(words) < 2:
        return 0.0

    bigrams = [(words[i], words[i+1]) for i in range(len(words) - 1)]
    return len(set(bigrams)) / len(bigrams)

def calculate_rouge1(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure

# --- METEOR (UNCHANGED LOGIC) ---

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def preprocess_sentence(sentence):
    if not isinstance(sentence, str):
        return []
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    return sentence.split()

def compute_meteor(reference, hypothesis, alpha=0.5):
    ref_tokens = preprocess_sentence(reference)
    hyp_tokens = preprocess_sentence(hypothesis)

    precision = len(set(ref_tokens) & set(hyp_tokens)) / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall = len(set(ref_tokens) & set(hyp_tokens)) / len(ref_tokens) if len(ref_tokens) > 0 else 0

    if precision == 0 and recall == 0:
        return 0.0

    penalty = alpha * (len(ref_tokens) / (len(ref_tokens) + len(hyp_tokens)))
    return precision * recall / (precision + (1 - alpha) * recall + penalty)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate MemoryBank generations")
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV_PATH,
                        help="Path to inference_results.csv")
    parser.add_argument("--output_csv", type=str, default=OUTPUT_CSV_PATH,
                        help="Path to evaluation.csv")
    parser.add_argument("--ppl_model_name", type=str, default=MODEL_NAME,
                        help="Local HuggingFace model used for perplexity.")
    parser.add_argument("--skip_ppl", action="store_true", default=False,
                        help="Skip perplexity computation. Useful for remote-model runs.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    model = None
    tokenizer = None
    if not args.skip_ppl:
        model = AutoModelForCausalLM.from_pretrained(
            args.ppl_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.ppl_model_name)

    df = pd.read_csv(args.input_csv)

    df['PPL'] = 0.0
    df['BLEU-2'] = 0.0
    df['BERTScore-F1'] = 0.0
    df['Distinct-2'] = 0.0
    df['ROUGE-1'] = 0.0
    df['METEOR'] = 0.0

    for i in range(len(df)):
        agent_reply = str(df.loc[i, 'ground response'])
        model_reply = str(df.loc[i, 'generated response'])

        if not model_reply.strip():
            continue

        if not args.skip_ppl:
            df.loc[i, 'PPL'] = calculate_perplexity(model_reply, model, tokenizer)
        else:
            df.loc[i, 'PPL'] = np.nan
        df.loc[i, 'BLEU-2'] = compute_bleu2(model_reply, agent_reply)
        df.loc[i, 'BERTScore-F1'] = compute_bert_score_f1([model_reply], [agent_reply])
        df.loc[i, 'Distinct-2'] = distinct_2(model_reply)
        df.loc[i, 'ROUGE-1'] = calculate_rouge1(agent_reply, model_reply)

        if not (is_number(model_reply) and is_number(agent_reply)):
            df.loc[i, 'METEOR'] = compute_meteor(agent_reply, model_reply)
        else:
            df.loc[i, 'METEOR'] = np.nan

        # Save progressively on the go
        df.to_csv(args.output_csv, index=False)
        
        # Optional: Print progress every 100 turns
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(df)} turns. Saved to {args.output_csv}")

    # ---- FINAL AVERAGES (ONLY PRINT) ----
    print("\nAverage Metrics")
    print("-" * 40)
    print(f"PPL          : {df['PPL'].mean():.4f}")
    print(f"BLEU-2       : {df['BLEU-2'].mean():.4f}")
    print(f"BERTScore-F1 : {df['BERTScore-F1'].mean():.4f}")
    print(f"Distinct-2   : {df['Distinct-2'].mean():.4f}")
    print(f"ROUGE-1      : {df['ROUGE-1'].mean():.4f}")
    print(f"METEOR       : {df['METEOR'].mean():.4f}")

if __name__ == "__main__":
    main()


# %%


