# src/summarizer.py
"""
LexVault-AI Summarizer
----------------------
Reads processed or predicted case data and generates a concise
summary for each judgment using TF-IDF sentence ranking.

Input:
    output/predicted_hybrid/all_cases_predicted_hybrid.json
Output:
    output/judgments_with_summaries.json
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# ----------------------------------------------------------
# Text Cleaning and Sentence Ranking
# ----------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize text and remove repetitive artifacts."""
    if not text:
        return ''
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page\s*\d+(\s*of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'IN THE HIGH COURT.*?(?=JUDGMENT|ORDER|CORAM|PETITIONER)', '', text, flags=re.IGNORECASE)
    text = text.strip()
    return text


def summarize_text(text: str, max_sentences: int = 4) -> str:
    """
    Extractive summarization using TF-IDF sentence ranking.
    Selects the top N sentences that best represent the document.
    """
    if not text:
        return ""
    text = clean_text(text)
    sents = sent_tokenize(text)

    # If the text is already short
    if len(sents) <= max_sentences:
        return ' '.join(sents)

    try:
        # TF-IDF vectorization per sentence
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
        M = vec.fit_transform(sents)
        doc_vec = M.mean(axis=0)
        scores = (M @ doc_vec.T).A1
        top_idx = np.argsort(scores)[-max_sentences:][::-1]
        summary = ' '.join([sents[i] for i in sorted(top_idx)])
        return summary.strip()
    except Exception as e:
        print(f"⚠️ Summarization failed for one case: {e}")
        return ' '.join(sents[:max_sentences])


# ----------------------------------------------------------
# Summarization Runner
# ----------------------------------------------------------
def main():
    BASE = Path(__file__).resolve().parent.parent
    INPUT_PATH = BASE / "output" / "predicted_hybrid" / "all_cases_predicted_hybrid.json"
    OUTPUT_PATH = BASE / "output" / "judgments_with_summaries.json"

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"❌ Input file not found: {INPUT_PATH}")

    print(f"📘 Loading cases from: {INPUT_PATH}")
    data = json.load(open(INPUT_PATH, "r", encoding="utf-8"))

    summarized_cases = []
    print(f"🧠 Generating summaries for {len(data)} cases...")

    for case in tqdm(data):
        text = case.get("judgment_text", "")
        summary = summarize_text(text, max_sentences=4)
        case["summary"] = summary
        summarized_cases.append(case)

    # Save summarized output
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summarized_cases, open(OUTPUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n✅ Summaries generated successfully!")
    print(f"📄 Output file saved at: {OUTPUT_PATH}")
    print(f"Example summary:\n{summarized_cases[0].get('summary', '')[:300]}...")


if __name__ == "__main__":
    main()
