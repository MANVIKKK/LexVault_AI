# src/entity_extractor.py
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter
from dateutil import parser as dateparser

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "cleaned"
OUTPUT_DIR = BASE_DIR / "output" / "extracted_v4_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------
# Utilities / Preprocessing
# ---------------------

LIGATURES = {"\uFB01": "fi", "\uFB02": "fl"}  # ﬁ, ﬂ

def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = text.replace("\u00A0", " ")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text

def collapse_whitespace_and_merge_sentences(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\f+", "\n", text)
    text = re.sub(r"\bPage\s*\d+(\s*of\s*\d+)?\b", " ", text, flags=re.IGNORECASE)

    lines = [ln.rstrip() for ln in text.split("\n")]
    cnt = Counter([ln.strip() for ln in lines if ln.strip()])
    repeats = {ln for ln, c in cnt.items() if c > 2 and len(ln) > 10}

    filtered = []
    for ln in lines:
        if ln.strip() in repeats:
            continue
        filtered.append(ln)

    text = "\n".join(filtered)
    paragraphs = re.split(r"\n{2,}", text)
    cleaned = []
    for p in paragraphs:
        p_comp = re.sub(r"\n+", " ", p)
        p_comp = re.sub(r"\s+", " ", p_comp).strip()
        cleaned.append(p_comp)
    return "\n\n".join([p for p in cleaned if p])

def preprocess_text_for_extraction(raw: str) -> str:
    txt = normalize_unicode(raw)
    txt = collapse_whitespace_and_merge_sentences(txt)
    txt = re.sub(r"(Mr\.|Ms\.|Mrs\.|Hon'?ble)\s*\n\s*(Justice)", r"\1 \2", txt, flags=re.IGNORECASE)
    txt = re.sub(r"(Justice)\s*\n\s*([A-Z])", r"\1 \2", txt)
    return txt

# ---------------------
# Extraction Functions
# ---------------------

def extract_court(lines_top):
    for ln in lines_top[:12]:
        if not ln:
            continue
        if "HIGH COURT" in ln.upper() or "SUPREME COURT" in ln.upper():
            return ln.strip()
    for ln in lines_top[:12]:
        if re.search(r"high court", ln, re.IGNORECASE) or re.search(r"supreme court", ln, re.IGNORECASE):
            return ln.strip()
    return None

def extract_case_id(lines_top):
    join_top = " ".join([ln for ln in lines_top[:20] if ln])
    patterns = [
        r"\b[A-Z]{1,5}\.?[A-Z0-9\-._]*\s*(?:No\.?|No|n[oO]\.)\s*\d+\/\d{2,4}\b",
        r"\b(?:CIVIL|CRIMINAL|WRIT|APPEAL|CRL|W\.P\.|WPC|CRM|FA|TP)\b.*?\d{1,6}\/\d{2,4}",
        r"\b[A-Z]{2,6}\d{6,}\b",
        r"\b[A-Z]{1,5}\d{6,}[_\-\dA-Za-z]*\b",
    ]
    for pat in patterns:
        for m in re.findall(pat, join_top, flags=re.IGNORECASE):
            if len(m) > 3:
                return m.strip()
    return None

def extract_date_from_text_or_filename(text, filename):
    for pat in [r"\bDated[:\s]*([A-Za-z0-9,\s/-]{6,30})", r"\bDated this\s*[:\s]*([A-Za-z0-9,\s/-]{6,30})"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return str(dateparser.parse(m.group(1), fuzzy=True).date())
            except Exception:
                pass
    date_patterns = [
        r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[-_]\d{2}[-_]\d{2}\b",
    ]
    for pat in date_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return str(dateparser.parse(m.group(0), fuzzy=True).date())
            except Exception:
                continue
    m = re.search(r"(\d{4}[-_]\d{2}[-_]\d{2})", filename)
    if m:
        return m.group(1)
    return None

def extract_judges(full_text):
    judges = []
    coram_match = re.search(r"(CORAM|BEFORE|PRESENT)\s*[:\-]?\s*(.*?)\n\n", full_text[:3000], flags=re.IGNORECASE | re.DOTALL)
    if coram_match:
        block = coram_match.group(2)
        parts = re.split(r",| and | AND |;|\n", block)
        for p in parts:
            p_clean = p.strip()
            if re.search(r"Justice|Hon'?ble|Mr\.|Ms\.|Mrs\.", p_clean, re.IGNORECASE):
                judges.append(re.sub(r'\s+', ' ', p_clean))
    if not judges:
        found = re.findall(r"(Hon'ble\s+)?(?:Mr\.|Ms\.|Mrs\.)?\s*Justice\s+[A-Z][A-Za-z.\s-]{3,60}", full_text[:4000], flags=re.IGNORECASE)
        judges.extend([re.sub(r'\s+', ' ', f.strip()) for f in found])
    if not judges:
        found2 = re.findall(r"\bJustice\s+[A-Z][A-Za-z.\s-]{2,60}", full_text, flags=re.IGNORECASE)
        judges.extend([re.sub(r'\s+', ' ', f.strip()) for f in found2])
    unique = []
    for j in judges:
        if j and j not in unique:
            unique.append(j)
    return unique or ["Unknown"]

def extract_parties(full_text, filename):
    petitioners, respondents = [], []
    head = full_text[:2000]
    m = re.search(r"(PETITIONER[S]?|APPELLANT[S]?)\s*[:\-]?\s*(.+?)(?:\n[A-Z]{2,}|VERSUS|VS\.|v\.)", head, flags=re.IGNORECASE | re.DOTALL)
    if m:
        names = m.group(2).strip()
        candidates = re.split(r";|\band\b|,|\n", names)
        petitioners.extend([c.strip() for c in candidates if len(c.strip()) > 2])
    m2 = re.search(r"(RESPONDENT[S]?|DEFENDANT[S]?)\s*[:\-]?\s*(.+?)(?:\n[A-Z]{2,}|JUDGMENT|ORDER|$)", head, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        names = m2.group(2).strip()
        candidates = re.split(r";|\band\b|,|\n", names)
        respondents.extend([c.strip() for c in candidates if len(c.strip()) > 2])
    vs_match = re.search(r"^(.{2,120}?)\s+(?:v\.|vs\.|versus|Vs\.)\s+(.{2,120}?)$", head.split("\n")[0], flags=re.IGNORECASE)
    if vs_match:
        petitioners.append(vs_match.group(1).strip())
        respondents.append(vs_match.group(2).strip())
    bet = re.search(r"Between\s*[:\-]?\s*(.+?)\s*(?:And|Vs\.?|v\.)\s*(.+?)\n", head, flags=re.IGNORECASE | re.DOTALL)
    if bet:
        petitioners.append(re.sub(r'\s+', ' ', bet.group(1).strip()))
        respondents.append(re.sub(r'\s+', ' ', bet.group(2).strip()))
    if not petitioners and "vs" in filename.lower():
        parts = re.split(r"_vs_|-vs-|_v_|-v-", filename, flags=re.IGNORECASE)
        if len(parts) >= 2:
            petitioners.append(parts[0].replace('_', ' ').strip())
            respondents.append(parts[1].replace('_', ' ').strip())
    petitioners = [re.sub(r'\s+', ' ', p).strip() for p in petitioners if p]
    respondents = [re.sub(r'\s+', ' ', r).strip() for r in respondents if r]
    petitioners = list(dict.fromkeys(petitioners)) or ["Unknown"]
    respondents = list(dict.fromkeys(respondents)) or ["Unknown"]
    return petitioners, respondents

def extract_acts(full_text):
    acts = []
    patterns = [
        r"\bSection\s+\d+[A-Za-z0-9\-/]*\s*(?:of\s+the\s+)?[A-Za-z.\s()]*?(?:IPC|CrPC|Code|Act|Constitution|Evidence Act)\b",
        r"\bu/s\s*\d+[A-Za-z0-9\-/]*\s*(?:IPC|CrPC)?\b",
        r"\bArticle\s+\d+\s+(?:of\s+the\s+Constitution)?\b",
        r"\b(?:IPC|CrPC|Evidence Act|Indian Penal Code|Constitution of India)\b"
    ]
    for pat in patterns:
        for m in re.findall(pat, full_text, flags=re.IGNORECASE):
            acts.append(re.sub(r'\s+', ' ', m.strip()))
    normalized = []
    for a in acts:
        a_norm = re.sub(r"\bu/s\b", "Section", a, flags=re.IGNORECASE)
        normalized.append(a_norm.strip())
    normalized = list(dict.fromkeys(normalized)) or ["Unknown"]
    return normalized

# ---------------------
# Main per-file
# ---------------------

def process_file(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        j = json.load(fh)
    raw_text = j.get("cleaned_text") or j.get("judgment_text") or ""
    processed_text = preprocess_text_for_extraction(raw_text)
    top_lines = [ln.strip() for ln in processed_text.split("\n") if ln.strip()][:40]

    court = extract_court(top_lines) or j.get("court")
    case_id = extract_case_id(top_lines) or j.get("case_id")
    date = extract_date_from_text_or_filename(processed_text, path.name)

    judges = extract_judges(processed_text)
    petitioners, respondents = extract_parties(processed_text, path.name)
    acts = extract_acts(processed_text)

    judgment_text = processed_text.strip() or None

    return {
        "case_id": case_id or None,
        "court": court or None,
        "date": date or None,
        "judges": judges or ["Unknown"],
        "petitioners": petitioners or ["Unknown"],
        "respondents": respondents or ["Unknown"],
        "acts": acts or ["Unknown"],
        "judgment_text": judgment_text or None
    }

def extract_all():
    files = sorted(INPUT_DIR.glob("*.json"))
    print(f"📂 Found {len(files)} files for entity extraction.")
    all_cases = []
    for f in files:
        result = process_file(f)
        all_cases.append(result)
        out_file = OUTPUT_DIR / f.name
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"✅ Extracted {f.name}")
    combined = OUTPUT_DIR / "all_cases_extracted_v4_1.json"
    with open(combined, "w", encoding="utf-8") as fh:
        json.dump(all_cases, fh, ensure_ascii=False, indent=2)
    print(f"\n✨ Extraction complete. Combined file saved to {combined}")

if __name__ == "__main__":
    extract_all()
