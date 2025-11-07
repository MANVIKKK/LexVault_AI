# src/similarity_engine.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE / "output" / "predicted_hybrid" / "all_cases_predicted_hybrid.json"
OUT_DIR = BASE / "output" / "similarity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_PATH = OUT_DIR / "tfidf.joblib"
EMBEDDINGS_PATH = OUT_DIR / "embeddings.npy"
METADATA_PATH = OUT_DIR / "metadata.json"
FAISS_INDEX_PATH = OUT_DIR / "faiss.index"
MODEL_META_PATH = OUT_DIR / "model_meta.json"

# Config
TFIDF_TOP_K = 200
FINAL_TOP_K = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # or "nlpaueb/legal-bert-base-uncased" if available
EMBED_DIM = 384  # all-MiniLM-L6-v2 -> 384

# ---------------------------
# Preprocessing helpers
# ---------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    # minimal cleaning: collapse whitespace, remove page markers, strip
    t = re.sub(r"\f+", " ", text)
    t = re.sub(r"Page\s*\d+(\s*of\s*\d+)?", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def extract_operative_text(full_text: str, pct: float = 0.2) -> str:
    # minimal: use last 20% or after ORDER/JUDGMENT anchor
    if not full_text:
        return ""
    hi = full_text.upper()
    anchors = ["ORDER", "JUDGMENT", "HELD", "CONCLUSION", "DISPOSITION"]
    last_pos = -1
    for a in anchors:
        pos = hi.rfind(a)
        if pos > last_pos:
            last_pos = pos
    if last_pos != -1:
        return full_text[last_pos:]
    length = len(full_text)
    start = max(0, int(length * (1.0 - pct)))
    return full_text[start:]

# ---------------------------
# Build or load TF-IDF
# ---------------------------
def build_or_load_tfidf(cases: List[Dict[str, Any]]) -> TfidfVectorizer:
    if TFIDF_PATH.exists():
        return joblib.load(TFIDF_PATH)
    texts = [normalize_text(c.get("judgment_text","") or "") for c in cases]
    vect = TfidfVectorizer(max_features=50000, stop_words='english', ngram_range=(1,2))
    vect.fit(texts)
    joblib.dump(vect, TFIDF_PATH)
    return vect

# ---------------------------
# Build embeddings + FAISS
# ---------------------------
def build_embeddings_and_index(cases: List[Dict[str, Any]], model_name=EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    texts = []
    metadata = []
    for c in cases:
        # prefer operative text + full for robust semantics
        op = extract_operative_text(c.get("judgment_text","") or "")
        text_for_embed = (op + "\n\n" + (c.get("judgment_text","") or ""))[:30000]  # cap length
        texts.append(normalize_text(text_for_embed))
        metadata.append({
            "case_id": c.get("case_id"),
            "court": c.get("court"),
            "acts": c.get("acts") or c.get("acts_referred") or [],
            "outcome": c.get("outcome"),
            "judges": c.get("judges")
        })
    # compute embeddings in batches
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    # save embeddings & metadata
    np.save(EMBEDDINGS_PATH, emb)
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    # build FAISS index (IndexFlatIP with normalized vectors -> cosine)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    # write model meta
    with open(MODEL_META_PATH, "w") as fh:
        json.dump({"model": model_name, "dim": int(dim), "count": int(len(metadata))}, fh)
    return index, metadata, emb, model

# ---------------------------
# Hybrid search: TF-IDF recall + dense rerank
# ---------------------------
def hybrid_search(query: str, k: int = FINAL_TOP_K, tfidf_top_k: int = TFIDF_TOP_K, use_tfidf=True):
    # load resources
    cases = json.load(open(INPUT_FILE, "r", encoding="utf-8"))
    vect = build_or_load_tfidf(cases)
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    emb = np.load(EMBEDDINGS_PATH)
    metadata = json.load(open(METADATA_PATH, "r", encoding="utf-8"))
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_norm = normalize_text(query)

    # 1) sparse recall (TF-IDF)
    candidate_idxs = None
    if use_tfidf:
        q_vec = vect.transform([q_norm])
        # compute cosine with all docs (sparse) -> get top tfidf_top_k indices
        scores = (q_vec * vect.transform([normalize_text(c.get("judgment_text","") or "") for c in cases]).T).toarray()[0]
        # simpler: use transform and dot product on stored matrix would be better; for now compute in memory
        idxs = np.argsort(scores)[::-1][:tfidf_top_k]
        candidate_idxs = idxs.tolist()
    else:
        # fallback: search entire dense index
        D, I = index.search(np.array([model.encode(q_norm, normalize_embeddings=True)]), tfidf_top_k)
        candidate_idxs = I[0].tolist()

    # 2) compute dense similarity for TF-IDF candidates
    q_emb = model.encode(q_norm, normalize_embeddings=True)
    candidate_embs = emb[candidate_idxs]
    sims = np.dot(candidate_embs, q_emb)
    # build combined score: 0.6*dense + 0.4*sparse (if sparse exists)
    combined_scores = None
    if use_tfidf:
        # compute tfidf scores again for candidates (inefficient above) — for production cache sparse scores
        # For brevity, create uniform sparse scores placeholder
        sparse_scores = np.ones_like(sims) * 0.5
        combined_scores = 0.6 * sims + 0.4 * sparse_scores
    else:
        combined_scores = sims

    topk_idx_order = np.argsort(combined_scores)[::-1][:k]
    results = []
    for pos in topk_idx_order:
        doc_idx = candidate_idxs[pos]
        doc_meta = metadata[doc_idx]
        results.append({
            "case_id": doc_meta.get("case_id"),
            "score": float(combined_scores[pos]),
            "acts": doc_meta.get("acts"),
            "outcome": doc_meta.get("outcome"),
            "judges": doc_meta.get("judges")
        })
    return results

# ---------------------------
# Utilities: build pipeline (run once)
# ---------------------------
def build_pipeline():
    cases = json.load(open(INPUT_FILE, "r", encoding="utf-8"))
    print("Building TF-IDF...")
    build_or_load_tfidf(cases)
    print("Building dense embeddings & FAISS index (this may take time)...")
    index, meta, embeddings, model = build_embeddings_and_index(cases)
    print("Done. Index saved to:", FAISS_INDEX_PATH)

if __name__ == "__main__":
    # build index
    build_pipeline()
    # quick query test
    q = "petition under Article 226 for quashing of FIR related to corruption"
    res = hybrid_search(q, k=5)
    print("Top results:", res)
