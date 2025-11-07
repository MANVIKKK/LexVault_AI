#!/usr/bin/env python3
"""
Final Aggregator for LexVault AI
--------------------------------
Combines all processed data (entities, outcomes, similarity, summaries)
into one unified file: output/judgments.json

Handles variations in summarizer output automatically.
"""

import json
from pathlib import Path
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "output"

# Input files from previous stages
ENTITIES_PATH = OUTPUT / "extracted_v4_1" / "all_cases_extracted_v4_1.json"
OUTCOME_PATH = OUTPUT / "predicted_hybrid" / "all_cases_predicted_hybrid.json"
SUMMARY_PATH = OUTPUT / "judgments_with_summaries.json"
SIM_PATH = OUTPUT / "similarity" / "metadata.json"
FINAL_PATH = OUTPUT / "judgments.json"

print("📘 Loading data from previous stages...")

def safe_load(path):
    if not path.exists():
        print(f"⚠️ Missing file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load {path}: {e}")
            return []

entity_data = safe_load(ENTITIES_PATH)
outcome_data = safe_load(OUTCOME_PATH)
summary_data = safe_load(SUMMARY_PATH)
sim_data = safe_load(SIM_PATH)

# Convert lists to dict by case_id for fast lookup
def to_dict(data):
    mapping = {}
    for item in data:
        if isinstance(item, dict) and "case_id" in item:
            mapping[item["case_id"]] = item
    return mapping

ent_map = to_dict(entity_data)
out_map = to_dict(outcome_data)
sim_map = to_dict(sim_data)

# 🧩 Fix summarizer output (some are list[str], some are list[dict])
sum_map = {}
if isinstance(summary_data, list):
    for item in summary_data:
        if isinstance(item, dict) and "case_id" in item:
            sum_map[item["case_id"]] = item.get("summary", "")
        elif isinstance(item, str):
            # fallback when summaries are only strings
            sum_map[str(len(sum_map) + 1)] = item
else:
    print("⚠️ Unexpected summary data format:", type(summary_data))

print(f"🔄 Combining {len(out_map)} cases into final judgments.json...")

final_cases = []

for case in tqdm(out_map.values()):
    cid = case.get("case_id")

    entity = ent_map.get(cid, {})
    summary = sum_map.get(cid, "")
    sim = sim_map.get(cid, {})

    final_case = {
        "case_id": cid,
        "court": case.get("court") or entity.get("court", ""),
        "date": case.get("date") or entity.get("date", ""),
        "judges": case.get("judges") or entity.get("judges", []),
        "petitioners": case.get("petitioners") or entity.get("petitioners", []),
        "respondents": case.get("respondents") or entity.get("respondents", []),
        "acts": case.get("acts") or entity.get("acts", []),
        "judgment_text": case.get("judgment_text") or "",
        "outcome": case.get("outcome") or "undetermined",
        "confidence": case.get("confidence", 0.0),
        "summary": summary or "Summary not available.",
        "similar_cases": sim.get("similar_cases", []) if isinstance(sim, dict) else [],
    }
    final_cases.append(final_case)

# Save final combined judgments
FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(FINAL_PATH, "w", encoding="utf-8") as f:
    json.dump(final_cases, f, ensure_ascii=False, indent=2)

print(f"\n✅ Aggregation complete. Final file saved to: {FINAL_PATH}")
print(f"📦 Total cases aggregated: {len(final_cases)}")
