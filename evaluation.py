# %%
import pandas as pd
import torch
import math
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from datetime import datetime
from bert_score import score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
INPUT_CSV_PATH = os.path.join(SCRIPT_DIR, "output", "session_split_v1", "inference_results.csv")
OUTPUT_CSV_PATH = os.path.join(SCRIPT_DIR, "output", "session_split_v1", "evaluation.csv")

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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_csv(INPUT_CSV_PATH)

    # Initialize columns with NaN
    metrics = ['PPL', 'BLEU-2', 'BERTScore-F1', 'Distinct-2', 'ROUGE-1', 'METEOR']
    for m in metrics:
        df[m] = np.nan

    # Resume from existing evaluation.csv if it exists
    import os
    if os.path.exists(OUTPUT_CSV_PATH):
        try:
            old_df = pd.read_csv(OUTPUT_CSV_PATH)
            # Use 'Persona Id', 'conversation id', 'turn index' to merge
            if all(col in old_df.columns for col in ['Persona Id', 'conversation id', 'turn index']):
                old_df_indexed = old_df.set_index(['Persona Id', 'conversation id', 'turn index'])
                df_indexed = df.set_index(['Persona Id', 'conversation id', 'turn index'])
                df_indexed.update(old_df_indexed)
                df = df_indexed.reset_index()
        except Exception as e:
            print(f"Could not load existing {OUTPUT_CSV_PATH}: {e}")

    start_time = datetime.now()

    for i in range(len(df)):
        # Skip if already evaluated
        if not pd.isna(df.loc[i, 'PPL']) and df.loc[i, 'PPL'] > 0:
            continue
            
        agent_reply = str(df.loc[i, 'ground response'])
        model_reply = str(df.loc[i, 'generated response'])

        if not model_reply.strip():
            continue

        df.loc[i, 'PPL'] = calculate_perplexity(model_reply, model, tokenizer)
        df.loc[i, 'BLEU-2'] = compute_bleu2(model_reply, agent_reply)
        df.loc[i, 'BERTScore-F1'] = compute_bert_score_f1([model_reply], [agent_reply])
        df.loc[i, 'Distinct-2'] = distinct_2(model_reply)
        df.loc[i, 'ROUGE-1'] = calculate_rouge1(agent_reply, model_reply)

        if not (is_number(model_reply) and is_number(agent_reply)):
            df.loc[i, 'METEOR'] = compute_meteor(agent_reply, model_reply)
        else:
            df.loc[i, 'METEOR'] = np.nan

        # Save progressively on the go
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        
        # Optional: Print progress every 100 turns
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(df)} turns. Saved to {OUTPUT_CSV_PATH}")

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



