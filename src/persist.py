# src/persist.py
import json, os
from pathlib import Path

def save_judgments(judgments, path='output/judgments.json'):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(judgments, fh, indent=2, ensure_ascii=False)
    print(f'[INFO] Saved {len(judgments)} judgments to {path}')

def load_judgments(path='output/judgments.json'):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception as e:
        print(f'[WARN] could not read {path}: {e}')
        return []
