# ⚖️ LexVault — Legal Judgment Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

LexVault is an **AI-driven legal analytics platform** designed to process and analyze high court judgments using advanced **Natural Language Processing (NLP)** and **Machine Learning (ML)**.  
It extracts entities, predicts case outcomes, finds similar past judgments, and generates summaries — all visualized interactively via a Streamlit dashboard.

---

## 🚀 Features

✅ **Automated Legal Text Processing**
- Extracts judges, petitioners, respondents, and acts from raw judgments.  
- Cleans and normalizes unstructured text.  

✅ **Outcome Prediction (Hybrid Model)**
- Combines rule-based logic and ML (TF-IDF + Logistic Regression).  
- Detects outcomes like `allowed`, `dismissed`, or `partly allowed`.  

✅ **Semantic Similarity Engine**
- Uses **Sentence Transformers** and **FAISS** for efficient similarity search.  
- Finds and clusters related judgments for legal research.  

✅ **Summarization & Explainability**
- Generates concise summaries of judgments.  
- Computes metrics like ROUGE, BLEU, and BERTScore for summary quality.  

✅ **Interactive Dashboard (Streamlit)**
- Explore, search, and filter cases.  
- Semantic case similarity and feature explainability.  
- Visual charts for outcome distributions.

---

## 🧠 Architecture Overview

