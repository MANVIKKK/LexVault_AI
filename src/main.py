#!/usr/bin/env python3
"""
LexVault-AI Main Pipeline Orchestrator (Fixed & Robust)
-------------------------------------------------------
Executes the full LexVault AI pipeline stage by stage:

1️⃣ Fetch raw PDFs → extract text
2️⃣ Preprocess data
3️⃣ Extract entities (judges, parties, acts)
4️⃣ Predict outcomes (dismissed / allowed)
5️⃣ Compute semantic similarity between cases
6️⃣ Summarize judgments
7️⃣ Aggregate everything into final judgments.json

Usage:
    python src/main.py
"""

import subprocess
import sys
import time
import shutil
from pathlib import Path

# ----------------------------
# Global paths
# ----------------------------
BASE = Path(__file__).resolve().parent          # src/
PROJECT_ROOT = BASE.parent                      # LexVault/
OUTPUT = PROJECT_ROOT / "output"                # output/
PYTHON = sys.executable                         # Current Python interpreter

# ----------------------------
# Helper to run each stage safely
# ----------------------------
def run_stage(name: str, script: str, expected_output: Path = None):
    """
    Run a stage script as a subprocess.
    Stops pipeline if any stage fails.
    """
    print(f"\n🔹 Running Stage: {name}")
    print("---------------------------------------------------")
    start_time = time.time()

    script_path = BASE / script
    if not script_path.exists():
        print(f"❌ ERROR: Missing script file {script_path}")
        sys.exit(1)

    try:
        # Run script in same environment
        result = subprocess.run(
            [PYTHON, str(script_path)],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,   # ensure consistent working directory
        )

        # Print live outputs
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print("⚠️", result.stderr.strip())

        elapsed = time.time() - start_time
        print(f"✅ {name} completed successfully in {elapsed:.2f}s")

        # Verify expected output (if specified)
        if expected_output:
            if not expected_output.exists():
                print(f"⚠️ WARNING: Expected output not found at {expected_output}")
            else:
                print(f"📄 Output verified: {expected_output}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ {name} FAILED.")
        print(f"--- STDERR ---\n{e.stderr}")
        print(f"--- STDOUT ---\n{e.stdout}")
        sys.exit(1)


# ----------------------------
# Main Pipeline Definition
# ----------------------------
def main():
    print("🚀 Starting LexVault AI Full Pipeline...")
    print("==========================================")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    PIPELINE = [
        ("Fetch Data", "fetch_data.py", OUTPUT / "raw_cases.json"),
        ("Preprocess Data", "preprocess.py", OUTPUT / "processed_cases.json"),
        ("Entity Extraction", "entity_extractor.py", OUTPUT / "extracted_v4_1" / "all_cases_extracted_v4_1.json"),
        ("Outcome Prediction", "outcome_predictor.py", OUTPUT / "predicted_hybrid" / "all_cases_predicted_hybrid.json"),
        ("Semantic Similarity", "similarity_engine.py", OUTPUT / "similarity" / "metadata.json"),
        ("Summarization", "summarizer.py", OUTPUT / "judgments_with_summaries.json"),
        ("Final Aggregation", "final_aggregator.py", OUTPUT / "judgments.json"),
    ]

    for stage_name, script, check_file in PIPELINE:
        run_stage(stage_name, script, check_file)

    print("\n🎯 LexVault AI Pipeline Completed Successfully!")
    print("==========================================")
    print(f"📁 Final Outputs:")
    print(f"   ✅ {OUTPUT / 'judgments_with_summaries.json'}")
    print(f"   ✅ {OUTPUT / 'judgments.json'}")
    print("==========================================")
    print("🏁 Pipeline execution finished.\n")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
