#!/usr/bin/env python3
"""
LexVault — Legal Judgment Intelligence Dashboard
================================================
Streamlit dashboard for exploring:
- Judgments
- Predicted outcomes
- Semantic similarity
- Summaries
- Explainability (basic)
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt

# ---------------------------
# Config / Paths
# ---------------------------
ROOT = Path(__file__).resolve().parent
OUTPUT_JSON = ROOT / "output" / "judgments.json"
MODEL_PKL = ROOT / "output" / "predicted_hybrid" / "outcome_pipeline.joblib"
VECT_PKL = ROOT / "output" / "predicted_hybrid" / "tfidf_vectorizer.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------
# Utility Functions
# ---------------------------
@st.cache_data(show_spinner=False)
def load_json(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def build_faiss_index(summaries):
    model = load_embedding_model()
    embeddings = model.encode(summaries, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def safe_load_model_vectorizer():
    model = None
    vec = None
    try:
        if MODEL_PKL.exists():
            model = joblib.load(MODEL_PKL)
        if VECT_PKL.exists():
            vec = joblib.load(VECT_PKL)
    except Exception as e:
        st.error(f"⚠️ Failed to load model/vectorizer: {e}")
    return model, vec

def search_similar(index, embeddings, query_embedding, top_k=5):
    if index.ntotal == 0:
        return [], []
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)
    return distances[0], indices[0]

def get_dataframe(data):
    df = pd.json_normalize(data)
    for col in ["case_id", "court", "judgment_text", "summary", "outcome", "judges", "acts", "date"]:
        if col not in df.columns:
            df[col] = ""
    return df

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="LexVault — Legal AI Dashboard", layout="wide")
st.title("⚖️ LexVault — Legal Judgment Intelligence Dashboard")
st.markdown("Explore predictions, summaries, and semantic similarity across high court judgments.")

# ---------------------------
# Load Data
# ---------------------------
data = load_json(OUTPUT_JSON)
if not data:
    st.error("❌ No data found at `output/judgments.json`. Run the main pipeline first.")
    st.stop()

df = get_dataframe(data)

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("🔍 Filters & Tools")

query = st.sidebar.text_input("Search Query (optional)")
court_filter = st.sidebar.multiselect("Court", sorted(df["court"].unique()), default=sorted(df["court"].unique()))
outcome_filter = st.sidebar.multiselect("Outcome", sorted(df["outcome"].unique()), default=sorted(df["outcome"].unique()))
judge_filter = st.sidebar.text_input("Filter by Judge name")

st.sidebar.markdown("---")
st.sidebar.header("🧠 Model & Embeddings")
model, vectorizer = safe_load_model_vectorizer()
if model is None or vectorizer is None:
    st.sidebar.warning("No trained model/vectorizer found.")
else:
    st.sidebar.success("Outcome model loaded successfully.")

summaries = df["summary"].fillna("").tolist()
case_ids = df["case_id"].tolist()
index, embeddings = build_faiss_index(summaries)

# ---------------------------
# Search Section
# ---------------------------
col1, col2 = st.columns([2.2, 1])

with col1:
    st.subheader("🧭 Semantic Search")
    st.info("Search using natural language or select an existing case to find similar ones.")

    input_query = st.text_area("Enter query or select a case:", value=query, height=80)
    selected_case = st.selectbox("Or choose a case ID", ["-- none --"] + case_ids)

    if st.button("🔎 Search Similar Cases"):
        if selected_case != "-- none --":
            idx = case_ids.index(selected_case)
            input_text = summaries[idx]
        else:
            input_text = input_query.strip()

        if not input_text:
            st.warning("Please enter a query or choose a case.")
        else:
            model_emb = load_embedding_model()
            q_emb = model_emb.encode([input_text], convert_to_numpy=True)
            distances, indices = search_similar(index, embeddings, q_emb, top_k=5)

            st.write("### Top Similar Cases:")
            for rank, idx in enumerate(indices, start=1):
                row = df.iloc[idx]
                st.markdown(f"**{rank}. {row['case_id']} — {row['court']}**  \n"
                            f"**Outcome:** {row['outcome']}  \n"
                            f"**Summary:** {row['summary'][:350]}...")

with col2:
    st.subheader("📊 Quick Stats")
    st.metric("Total Cases", len(df))
    st.metric("Courts", df["court"].nunique())
    st.metric("Summaries Available", int((df['summary'].str.len() > 0).sum()))

    st.markdown("### Outcome Distribution")
    counts = df["outcome"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_ylabel("Count")
    ax.set_xlabel("Outcome")
    plt.xticks(rotation=20)
    st.pyplot(fig)

# ---------------------------
# Case Browser
# ---------------------------
st.markdown("---")
st.subheader("📚 Browse Cases")

filtered = df[
    (df["court"].isin(court_filter)) &
    (df["outcome"].isin(outcome_filter))
]
if judge_filter:
    filtered = filtered[filtered["judges"].astype(str).str.contains(judge_filter, case=False, na=False)]
if query:
    mask = filtered["judgment_text"].str.contains(query, case=False, na=False) | filtered["summary"].str.contains(query, case=False, na=False)
    filtered = filtered[mask]

st.write(f"Showing {len(filtered)} cases after filters.")
if st.checkbox("Show Table", value=True):
    st.dataframe(filtered[["case_id", "court", "outcome", "date"]].reset_index(drop=True), height=300)

selected = st.selectbox("Select a case for details", ["-- none --"] + filtered["case_id"].tolist())
if selected != "-- none --":
    row = df[df["case_id"] == selected].iloc[0]
    st.markdown(f"### {row['case_id']} — {row['court']}")
    st.markdown(f"**Outcome:** {row['outcome']}")
    st.markdown(f"**Judges:** {row['judges']}")
    st.markdown(f"**Summary:** {row['summary']}")
    st.markdown("**Full Judgment (truncated):**")
    st.text(row["judgment_text"][:3000] + ("..." if len(row["judgment_text"]) > 3000 else ""))

# ---------------------------
# Download Section
# ---------------------------
st.markdown("---")
st.subheader("💾 Export Data")
if OUTPUT_JSON.exists():
    with open(OUTPUT_JSON, "rb") as f:
        st.download_button("📥 Download judgments.json", data=f, file_name="judgments.json", mime="application/json")
else:
    st.warning("No output file found to download.")

st.success("✅ Dashboard ready! Use filters and search to explore LexVault outputs.")
