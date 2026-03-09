"""
Memory Retrieval using BERT-base-uncased + FAISS
=================================================
Mirrors MemoryBank-SiliconFriend/memory_bank/memory_retrieval/local_doc_qa.py
but uses raw HuggingFace transformers (AutoModel + AutoTokenizer) for
BERT-base-uncased embeddings + FAISS (no sentence-transformers needed).

Class: BERTMemoryRetrieval
  - build_and_save_index(persona_id, memory_docs, index_dir)
  - load_index(persona_id, index_dir) ? (faiss_index, texts, dates)
  - search(query, faiss_index, texts, dates, top_k=3) ? List[(text, date)]
"""

import os
import json
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL = "bert-base-uncased"
DEFAULT_INDEX_DIR = os.path.join(SCRIPT_DIR, "memory_bank", "faiss_index")
TOP_K = 3
BATCH_SIZE = 16
MAX_SEQ_LEN = 512
# ---------------------------------------------


def mean_pool(token_embeddings, attention_mask):
    """Mean pooling over token embeddings, respecting padding mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class BERTMemoryRetrieval:
    """
    BERT-base-uncased + FAISS memory retrieval using raw HuggingFace transformers.
    Avoids sentence-transformers to skip Keras / TF import conflicts.
    """

    def __init__(self, model_name=EMBEDDING_MODEL, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Loading BERT embedding model: {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print("  Embedding model loaded.")

    def _embed(self, texts):
        """
        Embed a list of strings using BERT mean-pooling.
        Returns np.ndarray of shape (N, hidden_dim), L2-normalized.
        """
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            embeddings = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            # L2 normalize so inner product ? cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def build_and_save_index(self, persona_id, memory_docs, index_dir=DEFAULT_INDEX_DIR):
        """
        Build a FAISS index from memory_docs and save to disk.

        Args:
            persona_id (str): e.g., "P_001"
            memory_docs (list of dict): [{"text": "...", "date": "2015-01-11"}, ...]
            index_dir (str): root directory to save per-persona index
        """
        save_dir = os.path.join(index_dir, persona_id)
        os.makedirs(save_dir, exist_ok=True)

        texts = [d["text"] for d in memory_docs]
        dates = [d["date"] for d in memory_docs]

        print(f"  [{persona_id}] Embedding {len(texts)} memory chunks ...")
        embeddings = self._embed(texts)
        dim = embeddings.shape[1]

        # Build FAISS IndexFlatIP (Inner Product = cosine similarity after L2 norm)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save FAISS index
        index_path = os.path.join(save_dir, "index.faiss")
        faiss.write_index(index, index_path)

        # Save texts + dates (needed for retrieval lookup)
        meta_path = os.path.join(save_dir, "texts.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"texts": texts, "dates": dates}, f, indent=2, ensure_ascii=False)

        print(f"  [{persona_id}] [OK] FAISS index saved ? {save_dir}")
        return save_dir

    def load_index(self, persona_id, index_dir=DEFAULT_INDEX_DIR):
        """
        Load a saved FAISS index for a given persona.

        Returns:
            (faiss_index, texts, dates)
        """
        save_dir = os.path.join(index_dir, persona_id)
        index_path = os.path.join(save_dir, "index.faiss")
        meta_path = os.path.join(save_dir, "texts.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No FAISS index found for {persona_id} at {index_path}. "
                f"Run build_memory_index.py first."
            )

        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        texts = meta["texts"]
        dates = meta["dates"]
        print(f"  [{persona_id}] Loaded FAISS index: {index.ntotal} vectors, "
              f"{len(texts)} text chunks.")
        return index, texts, dates

    def search(self, query, faiss_index, texts, dates, top_k=TOP_K, max_date=None):
        """
        Retrieve top-K memory chunks relevant to the query.

        Args:
            query (str): The user's query string
            faiss_index: loaded FAISS index
            texts (list[str]): corresponding text chunks
            dates (list[str]): corresponding dates
            top_k (int): number of results to return
            max_date (str): If provided, ONLY retrieve memories strictly BEFORE this date

        Returns:
            List of dicts: [{"text": "...", "date": "...", "score": 0.92}, ...]
        """
        query_emb = self._embed([query])  # shape (1, dim)
        
        # If filtering by date, retrieve ALL from FAISS (since ntotal per persona is small, ~11)
        # to ensure we find top_k valid ones after filtering.
        k_search = faiss_index.ntotal if max_date else min(top_k, faiss_index.ntotal)
        scores, indices = faiss_index.search(query_emb, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
                
            item_date = dates[idx]
            
            # Filter strictly past dates to prevent data leakage from the future
            if max_date and item_date >= max_date:
                continue
                
            results.append({
                "text":  texts[idx],
                "date":  item_date,
                "score": float(score)
            })
            
            if len(results) == top_k:
                break
                
        return results


# ---------------------------------------------
# Helper: build memory_docs from memory.json
# ---------------------------------------------

def build_memory_docs(persona_memory, persona_id="User"):
    """
    Convert a single persona's memory dict into a list of memory document chunks.
    Each chunk = one date's raw dialogue + its summary (if available).

    Mirrors MemoryBank-SiliconFriend/memory_bank/build_memory_index.py
    generate_memory_docs()
    """
    docs = []
    history = persona_memory.get("history", {})
    summary = persona_memory.get("summary", {})

    for date, qa_pairs in history.items():
        # Build raw dialogue string
        text = f"Conversation on {date}:\n"
        for pair in qa_pairs:
            text += f"[User]: {pair['query'].strip()}\n"
            text += f"[Agent]: {pair['response'].strip()}\n"

        # Append date summary if available
        if date in summary and summary[date].get("content", ""):
            text += f"\nSummary of {date}: {summary[date]['content'].strip()}"

        docs.append({"text": text, "date": date})

    return docs


# ---------------------------------------------
# Quick test
# ---------------------------------------------
if __name__ == "__main__":
    MEMORY_FILE = os.path.join(SCRIPT_DIR, "memory_bank", "memory.json")

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory_dict = json.load(f)

    retriever = BERTMemoryRetrieval()

    pid = list(memory_dict.keys())[0]
    docs = build_memory_docs(memory_dict[pid], pid)
    print(f"\nBuilt {len(docs)} memory docs for {pid}")

    retriever.build_and_save_index(pid, docs)
    faiss_index, texts, dates = retriever.load_index(pid)

    query = "What insurance policy did the user discuss before?"
    results = retriever.search(query, faiss_index, texts, dates, top_k=2)
    print(f"\nQuery: '{query}'")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Date: {r['date']} | Score: {r['score']:.4f}")
        print(f"     Text: {r['text'][:200]}...")
