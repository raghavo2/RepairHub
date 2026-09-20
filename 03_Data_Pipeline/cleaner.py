import json
import logging
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from config import (
    EXTRACTED_FOLDER,
    CLEANED_FOLDER,
    LOG_FOLDER,
    CLEANER_LOG,
)

# ==========================================================
# Logging
# ==========================================================

LOG_FOLDER.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=CLEANER_LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# Load JSON
# ==========================================================

def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)

# ==========================================================
# Clean Text
# ==========================================================

def clean_text(text: str):
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x0c", "")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ==========================================================
# Advanced Sliding-Window Boilerplate Removal
# ==========================================================

def remove_boilerplate(pages_text: list[str]) -> list[str]:
    """
    Detects and mathematically masks out any sequence of 60+ characters 
    that repeats across >= 30% of pages, ignoring newlines completely.
    """
    num_pages = len(pages_text)
    if num_pages <= 2:
        return pages_text

    threshold = max(3, int(num_pages * 0.30))
    chunk_size = 60
    chunk_counts = Counter()

    # 1. Build frequency map of all 60-char sequences
    for text in pages_text:
        page_chunks = set()
        for i in range(len(text) - chunk_size + 1):
            page_chunks.add(text[i:i+chunk_size])
        
        for c in page_chunks:
            chunk_counts[c] += 1

    # 2. Identify sequences that meet the threshold
    bad_chunks = {chunk for chunk, count in chunk_counts.items() if count >= threshold}

    if not bad_chunks:
        return pages_text

    # 3. Use a boolean mask to safely delete overlapping bad chunks
    cleaned_pages = []
    for text in pages_text:
        if not text:
            cleaned_pages.append(text)
            continue
            
        keep_chars = [True] * len(text)
        
        # Apply the deletion mask
        for i in range(len(text) - chunk_size + 1):
            if text[i:i+chunk_size] in bad_chunks:
                for j in range(i, i + chunk_size):
                    keep_chars[j] = False
                    
        # Reconstruct string from surviving characters
        cleaned_text = "".join([text[i] for i in range(len(text)) if keep_chars[i]])
        
        # Clean up residual artifacts
        cleaned_text = re.sub(r'[ \t]{2,}', ' ', cleaned_text)
        cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)
        cleaned_pages.append(cleaned_text.strip())
        
    return cleaned_pages

# ==========================================================
# Clean Document
# ==========================================================

def clean_document(document):
    for page in document["pages"]:
        page["raw_text"] = page.pop("text", "")
        page["clean_text"] = clean_text(page["raw_text"])

    pages_texts = [page["clean_text"] for page in document.get("pages", [])]
    cleaned_texts = remove_boilerplate(pages_texts)

    for i, page in enumerate(document["pages"]):
        page["clean_text"] = clean_text(cleaned_texts[i])

    return document

# ==========================================================
# Save JSON
# ==========================================================

def save_json(document):
    CLEANED_FOLDER.mkdir(parents=True, exist_ok=True)
    filename = Path(document["document_name"]).stem + "_clean.json"
    output = CLEANED_FOLDER / filename

    with open(output, "w", encoding="utf-8") as file:
        json.dump(document, file, indent=4, ensure_ascii=False)

    logger.info(f"Saved {filename}")

# ==========================================================
# Process Directory
# ==========================================================

def process_directory():
    json_files = sorted(EXTRACTED_FOLDER.glob("*.json"))
    successful = 0

    for json_file in tqdm(json_files, desc="Cleaning"):
        document = load_json(json_file)
        document = clean_document(document)
        save_json(document)
        successful += 1

    print("\n" + "=" * 50)
    print("Cleaning Completed")
    print("=" * 50)
    print(f"Documents Cleaned : {successful}")

if __name__ == "__main__":
    process_directory()