import faiss
import numpy as np

index = faiss.IndexFlatIP(384)
q_emb = np.random.rand(1, 384).astype(np.float32)
try:
    index.search(q_emb, 0)
    print("k=0 works")
except Exception as e:
    import traceback
    traceback.print_exc()
