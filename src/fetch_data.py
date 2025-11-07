# src/fetch_data.py
import fitz  # PyMuPDF
import json
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"

# Create output folder if not exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts full text from a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path.name}: {e}")
    return text.strip()


def process_folder(folder_path: Path, court_name: str):
    """Processes all PDFs in a folder and saves JSON files."""
    pdf_files = list(folder_path.glob("*.pdf"))
    print(f"📂 Found {len(pdf_files)} PDFs in {folder_path.name}")

    for pdf_path in pdf_files:
        case_id = pdf_path.stem  # file name without extension
        text = extract_text_from_pdf(pdf_path)

        if not text:
            print(f"⚠️ Skipping {case_id} (empty text)")
            continue

        data = {
            "case_id": case_id,
            "court": court_name,
            "judgment_text": text
        }

        out_file = OUTPUT_DIR / f"{case_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {out_file.name}")


def fetch_all_data():
    """Main runner function — processes both courts."""
    madras_path = DATA_DIR / "madras_cases"
    delhi_path = DATA_DIR / "delhi_cases"

    if madras_path.exists():
        process_folder(madras_path, "Madras High Court")
    if delhi_path.exists():
        process_folder(delhi_path, "Delhi High Court")

    print("\n✅ All PDF judgments processed and saved to 'data/processed/' folder.")


if __name__ == "__main__":
    fetch_all_data()
