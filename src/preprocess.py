# src/preprocess.py
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_judgment_text(text: str) -> str:
    """
    Cleans judgment text by removing unwanted symbols, repeated headers,
    line numbers, and multiple blank lines.
    """
    if not text:
        return ""

    # Normalize line breaks and remove multiple spaces
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove repeating "IN THE HIGH COURT..." lines (appears multiple times)
    text = re.sub(
        r"(IN THE HIGH COURT OF JUDICATURE AT (MADRAS|DELHI)[\s\S]{0,80}?)\1+",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    # Remove multiple occurrences of court heading
    text = re.sub(
        r"(IN THE HIGH COURT OF JUDICATURE AT (MADRAS|DELHI))",
        lambda m: "\n" + m.group(1).upper(),
        text,
        flags=re.IGNORECASE,
    )

    # Remove leading line numbers or page numbers
    text = re.sub(r"^\s*\d+\s+", "", text, flags=re.MULTILINE)

    # Remove sections with only numbers (like "1\n1\n1")
    text = re.sub(r"(\n\d+\s*)+", "\n", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove any random control characters
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u0B80-\u0BFF]+", " ", text)

    # Strip spaces at start/end
    text = text.strip()

    return text


def preprocess_all():
    files = list(INPUT_DIR.glob("*.json"))
    print(f"📂 Found {len(files)} JSON files to preprocess.")

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = data.get("judgment_text", "")
        cleaned = clean_judgment_text(text)

        # Add cleaned version to data
        data["cleaned_text"] = cleaned

        # Save new version to output
        output_file = OUTPUT_DIR / file_path.name
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Cleaned and saved: {output_file.name}")

    print("\n✨ All judgments cleaned and saved in 'data/cleaned/' folder.")


if __name__ == "__main__":
    preprocess_all()
