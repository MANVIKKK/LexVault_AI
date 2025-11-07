#!/usr/bin/env python3
"""
Patched hybrid outcome predictor for legal judgments.

Features added:
- Sentence-level ensemble fallback
- Embedding-based KNN fallback (sentence-transformers)
- LLM fallback helper (OpenAI-compatible api_call_fn placeholder)
- Improved combine_rule_and_ml that uses: rule -> ml -> sentence-ensemble -> knn -> llm -> moderate rule -> undetermined
- Utilities to build/load label embeddings
- Logging of low-confidence cases for active learning

Drop-in replacement for src/outcome_predictor.py. Adjust paths or model names as required.
"""
from pathlib import Path
import json
import re
import math
from typing import Tuple, Optional, Dict, Any
import joblib
from collections import defaultdict
import logging
import os

# ML imports (optional at runtime; ensure scikit-learn & joblib installed)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    TfidfVectorizer = None
    LogisticRegression = None
    Pipeline = None

# Sentence tokenization
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass
from nltk.tokenize import sent_tokenize

# Optional embedding model
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

# OPTIONAL: OpenAI client (only used if you want llm fallback)
try:
    import openai
except Exception:
    openai = None

# -------------------------
# Configuration & Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "output" / "extracted_v4_1"
OUTPUT_DIR = BASE_DIR / "output" / "predicted_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "outcome_pipeline.joblib"
EMBED_PATH = OUTPUT_DIR / "label_embeddings.joblib"
LOG_PATH = OUTPUT_DIR / "prediction_log.jsonl"

# Embedding model name (change to legal-specific if available)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("outcome_predictor")

# -------------------------
# Rule-based lexicon & utils
# -------------------------

DECISION_LEXICON = {
    "allowed": ["allowed", "allow", "petition allowed", "petition is allowed", "petition succeeds", "relief granted", "grant the petition", "petitioner is successful"],
    "partly_allowed": ["partly allowed", "allowed in part", "partly allowed and partly dismissed", "partly allowed/discharged"],
    "dismissed": ["dismissed", "petition dismissed", "petition is hereby dismissed", "petition is dismissed as", "petition fails", "dismissal", "stands dismissed", "rejected", "disposed of"],
    "quashed": ["quashed", "set aside", "set aside the order", "quash the order"],
    "undetermined": []
}

TOKEN_TO_LABEL = {}
for label, phrases in DECISION_LEXICON.items():
    for ph in phrases:
        TOKEN_TO_LABEL[ph.lower()] = label

SYNONYM_MAP = {
    "quash": "allowed",
    "set aside": "allowed",
    "disposed of": "dismissed",
    "dismissal": "dismissed",
    "fails": "dismissed",
    "granted": "allowed"
}

NEGATION_WORDS = {"not", "no", "never", "without", "cannot", "can't", "didn't", "doesn't", "nor"}
NEGATION_WINDOW_TOKENS = 3


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # collapse multiple spaces and newlines
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    # normalize unicode ligatures if present
    try:
        import unicodedata
        t = unicodedata.normalize("NFKC", t)
    except Exception:
        pass
    return t.strip()


def extract_operative_text(full_text: str, pct: float = 0.20) -> str:
    if not full_text:
        return ""
    text = normalize_text(full_text)
    anchors = ["\nORDER", "\nJUDGMENT", "\nHELD", "\nCONCLUSION", "\nDISPOSITION", "\nORDER:", "\nJUDGMENT:"]
    idx = -1
    for a in anchors:
        pos = text.upper().rfind(a)
        if pos > idx:
            idx = pos
    if idx != -1:
        return text[idx + 1:].strip()
    length = len(text)
    start = max(0, int(length * (1.0 - pct)))
    return text[start:].strip()


def check_negation_around(match_span: Tuple[int, int], text: str) -> bool:
    start_idx = match_span[0]
    pre = text[max(0, start_idx - 200): start_idx]
    toks = re.findall(r"\w+|[^\s\w]", pre)
    toks = [t.lower() for t in toks if t.strip()]
    if not toks:
        return False
    last_tokens = toks[-(NEGATION_WINDOW_TOKENS + 3):]
    for neg in NEGATION_WORDS:
        if neg in last_tokens:
            return True
    return False


def rule_based_detect(full_text: str) -> Dict[str, Any]:
    if not full_text:
        return {"outcome": "undetermined", "confidence": 0.0, "source": "rule-based", "reasoning": "Empty text"}

    operative = extract_operative_text(full_text, pct=0.25)
    text_to_search = operative.lower()

    matches = []
    reasoning_parts = []

    for phrase, label in TOKEN_TO_LABEL.items():
        if phrase in text_to_search:
            m = re.search(re.escape(phrase), text_to_search)
            if m:
                negated = check_negation_around((m.start(), m.end()), text_to_search)
                if negated:
                    reasoning_parts.append(f"Found phrase '{phrase}' but negated near it")
                    continue
                matches.append((label, phrase, m.start(), m.end()))

    for syn, mapped in SYNONYM_MAP.items():
        if syn in text_to_search:
            m = re.search(re.escape(syn), text_to_search)
            if m:
                negated = check_negation_around((m.start(), m.end()), text_to_search)
                if not negated:
                    matches.append((mapped, syn, m.start(), m.end()))
                else:
                    reasoning_parts.append(f"Found synonym '{syn}' but negated")

    strong_patterns = [
        (r"petition\s+(is\s+)?allowed", "allowed"),
        (r"petition\s+(is\s+)?partly\s+allowed|allowed\s+in\s+part", "partly_allowed"),
        (r"petition\s+(is\s+)?dismissed", "dismissed"),
        (r"\bquash(ed)?\b", "allowed"),
        (r"\bset aside\b", "allowed"),
        (r"\bdisposed of\b", "dismissed"),
        (r"\bno merit\b", "dismissed"),
        (r"\bstand(s)?\s+dismissed\b", "dismissed"),
        (r"petition\s+(is\s+)?dismissed\s+with\s+costs", "dismissed"),
    ]
    for pat, label in strong_patterns:
        for m in re.finditer(pat, text_to_search, flags=re.IGNORECASE):
            if not check_negation_around((m.start(), m.end()), text_to_search):
                matches.append((label, m.group(0), m.start(), m.end()))

    if not matches:
        return {"outcome": "undetermined", "confidence": 0.0, "source": "rule-based", "reasoning": "No rule-based match in operative section."}

    label_scores = defaultdict(float)
    label_reasons = defaultdict(list)
    text_len = len(text_to_search)
    for label, phrase, start, end in matches:
        label_scores[label] += 1.0
        proximity = 1.0 - (start / max(1, text_len))
        label_scores[label] += 0.5 * proximity
        label_reasons[label].append(phrase)

    sorted_labels = sorted(label_scores.items(), key=lambda x: -x[1])
    best_label, best_score = sorted_labels[0]
    total = sum(label_scores.values()) or 1.0
    confidence = min(0.999, max(0.05, best_score / total))
    if best_score >= 2.5:
        confidence = max(confidence, 0.85)
    elif best_score >= 1.5:
        confidence = max(confidence, 0.65)
    elif best_score >= 1.0:
        confidence = max(confidence, 0.45)

    reasoning = f"Rule matches: {dict(label_reasons)}; selected '{best_label}' with score {best_score:.2f}."
    if reasoning_parts:
        reasoning += " " + " ; ".join(reasoning_parts)

    if best_label == "quashed":
        canonical = "allowed"
    elif best_label == "partly_allowed":
        canonical = "partly_allowed"
    else:
        canonical = best_label

    return {"outcome": canonical, "confidence": float(confidence), "source": "rule-based", "reasoning": reasoning}

# -------------------------
# ML Fallback utilities
# -------------------------

def train_ml_pipeline(labeled_records: list, save_path: Path = MODEL_PATH):
    if Pipeline is None:
        raise RuntimeError("scikit-learn not available in environment")

    texts = [r.get("judgment_text", "") for r in labeled_records if r.get("judgment_text")]
    labels = [r.get("outcome") for r in labeled_records if r.get("outcome")]
    if not texts or not labels or len(texts) != len(labels):
        raise ValueError("Insufficient/invalid training data")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,3), min_df=3, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipe.fit(texts, labels)
    joblib.dump(pipe, save_path)
    return pipe


def load_ml_pipeline(path: Path = MODEL_PATH):
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def predict_with_ml(pipe, text: str) -> Tuple[str, float]:
    if not pipe or not text:
        return "undetermined", 0.0
    try:
        probs = pipe.predict_proba([text])[0]
        classes = pipe.classes_
        idx = probs.argmax()
        return classes[idx], float(probs[idx])
    except Exception:
        try:
            pred = pipe.predict([text])[0]
            return pred, 0.5
        except Exception:
            return "undetermined", 0.0

# -------------------------
# Sentence-level ensemble
# -------------------------

def get_candidate_sentences(full_text: str, max_sentences: int = 30) -> list:
    text = normalize_text(full_text)
    oper = extract_operative_text(text, pct=0.25)
    sents = sent_tokenize(oper)
    if len(sents) < 3:
        sents = sent_tokenize(text)
    return sents[-max_sentences:]


def ensemble_on_sentences(full_text: str, rule_res, ml_pipe):
    sents = get_candidate_sentences(full_text, max_sentences=40)
    candidates = []
    for s in sents:
        r = rule_based_detect(s)
        ml_label, ml_conf = ("undetermined", 0.0)
        if ml_pipe:
            ml_label, ml_conf = predict_with_ml(ml_pipe, s)
        if r["confidence"] >= 0.8 and r["outcome"] != "undetermined":
            candidates.append(("rule", r["outcome"], r["confidence"], s))
        elif ml_conf >= 0.6 and ml_label:
            candidates.append(("ml", ml_label, ml_conf, s))
    if not candidates:
        return None
    score = defaultdict(float)
    reasons = defaultdict(list)
    for src, lab, conf, sent in candidates:
        score[lab] += conf
        reasons[lab].append(f"{src}:{conf:.2f}:'{sent[:80]}...'")
    best = max(score.items(), key=lambda x: x[1])
    label = best[0]
    conf = min(0.99, float(best[1]) / (len(candidates) + 0.001))
    reasoning = f"Sentence-ensemble voted {dict(reasons)}"
    return {"outcome": label, "confidence": conf, "source": "sentence-ensemble", "reasoning": reasoning}

# -------------------------
# Embedding-based KNN fallback
# -------------------------

def build_label_embeddings(labeled_records: list, model_name: str = EMBED_MODEL_NAME):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    model = SentenceTransformer(model_name)
    texts = [normalize_text(r["judgment_text"])[-2000:] for r in labeled_records]
    labels = [r["outcome"] for r in labeled_records]
    emb = model.encode(texts, show_progress_bar=True)
    joblib.dump({"model_name": model_name, "emb": emb, "labels": labels, "texts": texts}, EMBED_PATH)
    logger.info(f"Saved label embeddings to {EMBED_PATH}")
    return True


def load_label_embeddings():
    if not EMBED_PATH.exists() or SentenceTransformer is None:
        return None
    data = joblib.load(EMBED_PATH)
    model = SentenceTransformer(data["model_name"])
    emb = np.array(data["emb"])
    labels = data["labels"]
    texts = data["texts"]
    return {"model": model, "emb": emb, "labels": labels, "texts": texts}


def knn_fallback(full_text: str, knn_store, top_k: int = 5):
    if not knn_store:
        return None
    model = knn_store["model"]
    emb_store = knn_store["emb"]
    labels = knn_store["labels"]
    texts = knn_store["texts"]
    q = normalize_text(full_text)[-2000:]
    qemb = model.encode([q])[0]
    sims = (emb_store @ qemb) / (np.linalg.norm(emb_store, axis=1) * (np.linalg.norm(qemb) + 1e-9))
    top_idx = sims.argsort()[::-1][:top_k]
    vote = defaultdict(float)
    details = []
    for i in top_idx:
        vote[labels[i]] += float(sims[i])
        details.append((labels[i], float(sims[i]), texts[i][:120]))
    if not vote:
        return None
    best = max(vote.items(), key=lambda x: x[1])
    conf = min(0.99, best[1] / (sum(vote.values()) + 1e-9))
    reasoning = f"KNN neighbors: {details}"
    return {"outcome": best[0], "confidence": float(conf), "source": "knn-emb", "reasoning": reasoning}

# -------------------------
# LLM fallback helper (OpenAI example)
# -------------------------

LLM_FALLBACK_PROMPT = """
You are an expert Legal Document Analyst. Input: an operative excerpt from a court judgment.
Task: determine the final outcome of the petition/appeal. Output ONLY a JSON object with keys:
  - "outcome": one of ["allowed","dismissed","partly_allowed","quashed","undetermined"]
  - "confidence": float 0.0-1.0 (estimate)
  - "reasoning": a one-sentence justification citing the exact clause or phrase you used from the excerpt.

Rules:
1. Do not invent facts beyond the excerpt.
2. If multiple outcomes for different petitioners exist and the excerpt does not disambiguate, return "undetermined".
3. If the excerpt contains explicit text like "petition is hereby dismissed" return "dismissed" with high confidence.
4. Use conservative confidence estimates.

Return example:
{"outcome":"dismissed","confidence":0.95,"reasoning":"Phrase 'petition is hereby dismissed' appears in final paragraph."}

Now analyze the DOCUMENT below and return JSON only.
"""


def default_openai_api_call(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 150) -> str:
    """
    Minimal OpenAI call wrapper. Requires OPENAI_API_KEY env var. Returns model's text.
    Replace with your internal LLM caller if needed.
    """
    if openai is None:
        raise RuntimeError("openai package not installed")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in env")
    openai.api_key = key
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        text = resp["choices"][0]["message"]["content"]
        return text
    except Exception as e:
        logger.exception("OpenAI call failed")
        raise


def llm_fallback_predict(full_text: str, api_call_fn, max_chars: int = 4000) -> dict:
    snippet = normalize_text(extract_operative_text(full_text, pct=0.25))
    if not snippet:
        snippet = normalize_text(full_text)[-max_chars:]
    prompt = LLM_FALLBACK_PROMPT + "\n\nDOCUMENT:\n" + snippet
    model_resp = api_call_fn(prompt)
    try:
        j = json.loads(model_resp)
        return {
            "outcome": j.get("outcome"),
            "confidence": float(j.get("confidence", 0.0)),
            "source": "llm",
            "reasoning": j.get("reasoning", "")
        }
    except Exception:
        return {"outcome": "undetermined", "confidence": 0.0, "source": "llm", "reasoning": model_resp[:400]}

# -------------------------
# New combine function with full fallback chain
# -------------------------

def combine_rule_and_ml(full_text: str, rule_res: Dict[str, Any], ml_pipe, knn_store=None, api_call_fn=None) -> Dict[str, Any]:
    RULE_CONF_THRESHOLD = 0.80
    ML_CONF_THRESHOLD = 0.60

    if rule_res["confidence"] >= RULE_CONF_THRESHOLD and rule_res["outcome"] != "undetermined":
        return rule_res

    if ml_pipe:
        ml_label, ml_conf = predict_with_ml(ml_pipe, full_text)
        if ml_conf >= ML_CONF_THRESHOLD and ml_label and ml_label != "undetermined":
            return {"outcome": ml_label, "confidence": float(ml_conf), "source": "ml", "reasoning": f"ML prob={ml_conf:.3f}"}

    ens = ensemble_on_sentences(full_text, rule_res, ml_pipe)
    if ens and ens["confidence"] >= 0.55:
        return ens

    if knn_store is not None:
        try:
            knn_res = knn_fallback(full_text, knn_store)
            if knn_res and knn_res["confidence"] >= 0.55:
                return knn_res
        except Exception:
            logger.exception("KNN fallback failed")

    if api_call_fn is not None:
        try:
            llm_res = llm_fallback_predict(full_text, api_call_fn)
            if llm_res and llm_res.get("confidence", 0.0) >= 0.5 and llm_res.get("outcome"):
                return llm_res
        except Exception:
            logger.exception("LLM fallback failed")

    if rule_res["confidence"] >= 0.45 and rule_res["outcome"] != "undetermined":
        rule_res["source"] = "rule-based"
        rule_res["reasoning"] = "Using moderate-confidence rule result. " + rule_res.get("reasoning", "")
        return rule_res

    return {"outcome": "undetermined", "confidence": 0.0, "source": "hybrid", "reasoning": "All fallbacks failed or not confident."}

# -------------------------
# Processing & CLI
# -------------------------

def process_all(outcome_mode: str = "hybrid", use_llm: bool = False):
    files = sorted(INPUT_DIR.glob("*.json"))
    logger.info(f"Found {len(files)} extracted-case files for outcome prediction.")
    ml_pipe = None
    if outcome_mode in ("ml", "hybrid"):
        ml_pipe = load_ml_pipeline()
        if not ml_pipe:
            logger.warning("ML pipeline not found; ML fallback disabled")

    knn_store = load_label_embeddings()
    if knn_store is None:
        logger.info("KNN embeddings not found or sentence-transformers missing; KNN fallback disabled")

    api_call_fn = None
    if use_llm:
        # wire default openai call if available
        if openai is not None:
            api_call_fn = default_openai_api_call
        else:
            logger.warning("LLM fallback requested but openai package not available; skipping LLM fallback")

    results = []
    low_conf_queue = []

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            logger.warning(f"Skipping combined file: {f.name}")
            continue
        case_id = data.get("case_id", f.stem)
        full_text = data.get("judgment_text") or ""
        if not full_text:
            full_text = " ".join([str(data.get(k, "")) for k in ("acts", "judges", "petitioners", "respondents")])

        rule_res = rule_based_detect(full_text)
        if outcome_mode == "rule":
            final_res = rule_res
        elif outcome_mode == "ml":
            if ml_pipe:
                ml_label, ml_conf = predict_with_ml(ml_pipe, full_text)
                final_res = {"outcome": ml_label, "confidence": float(ml_conf), "source": "ml", "reasoning": f"ML predicted '{ml_label}' with prob {ml_conf:.3f}"}
            else:
                final_res = {"outcome": "undetermined", "confidence": 0.0, "source": "ml", "reasoning": "ML model not available"}
        else:
            final_res = combine_rule_and_ml(full_text, rule_res, ml_pipe, knn_store=knn_store, api_call_fn=api_call_fn)

        out_record = dict(data)
        out_record["outcome"] = final_res.get("outcome")
        out_record["confidence"] = round(float(final_res.get("confidence", 0.0)), 3)
        out_record["source"] = final_res.get("source")
        out_record["reasoning"] = final_res.get("reasoning")

        out_path = OUTPUT_DIR / f.name
        with open(out_path, "w", encoding="utf-8") as oh:
            json.dump(out_record, oh, ensure_ascii=False, indent=2)
        results.append(out_record)
        logger.info(f"{f.name} -> {out_record['outcome']} (conf={out_record['confidence']}) via {out_record['source']}")

        if out_record["confidence"] < 0.5:
            low_conf_queue.append({"case_id": case_id, "file": str(f), "outcome": out_record["outcome"], "confidence": out_record["confidence"], "source": out_record["source"]})

        # append to incremental log
        with open(LOG_PATH, "a", encoding="utf-8") as logf:
            logf.write(json.dumps({"case_id": case_id, "outcome": out_record["outcome"], "confidence": out_record["confidence"], "source": out_record["source"], "reasoning": out_record["reasoning"]}, ensure_ascii=False) + "\n")

    combined = OUTPUT_DIR / "all_cases_predicted_hybrid.json"
    with open(combined, "w", encoding="utf-8") as ch:
        json.dump(results, ch, ensure_ascii=False, indent=2)
    logger.info(f"Completed predictions. Combined saved to: {combined}")

    if low_conf_queue:
        logger.warning(f"{len(low_conf_queue)} low-confidence cases (conf<0.5) written to: {OUTPUT_DIR / 'low_conf_cases.json'}")
        with open(OUTPUT_DIR / 'low_conf_cases.json', 'w', encoding='utf-8') as lcf:
            json.dump(low_conf_queue, lcf, ensure_ascii=False, indent=2)

# -------------------------
# Training & embedding builder CLI
# -------------------------

def train_from_file(labeled_json_path: Path):
    if Pipeline is None:
        raise RuntimeError("scikit-learn not installed in environment")
    if not labeled_json_path.exists():
        raise FileNotFoundError(labeled_json_path)
    with open(labeled_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Training file must be a JSON list of labeled records")
    pipe = train_ml_pipeline(data, save_path=MODEL_PATH)
    logger.info("Trained and saved ML pipeline.")
    return pipe


def build_embeddings_cli(labeled_json_path: Path, model_name: str = EMBED_MODEL_NAME):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed in environment")
    if not labeled_json_path.exists():
        raise FileNotFoundError(labeled_json_path)
    with open(labeled_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Embeddings training file must be a JSON list of labeled records")
    build_label_embeddings(data, model_name=model_name)
    logger.info("Built and saved embeddings store.")

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced hybrid outcome predictor for legal judgments")
    parser.add_argument("--mode", choices=["rule", "ml", "hybrid"], default="hybrid", help="Prediction mode")
    parser.add_argument("--train", type=str, default=None, help="Path to labeled JSON list to train ML pipeline")
    parser.add_argument("--build-emb", type=str, default=None, help="Path to labeled JSON list to build embedding store")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM fallback (requires OPENAI_API_KEY and openai package)")
    args = parser.parse_args()

    if args.train:
        train_from_file(Path(args.train))
    if args.build_emb:
        build_embeddings_cli(Path(args.build_emb))
    process_all(outcome_mode=args.mode, use_llm=args.use_llm)
