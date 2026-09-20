import json
import logging
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

from config import (
    KNOWLEDGE_BASE,
    EXTRACTED_FOLDER,
    PARSER_LOG,
    LOG_FOLDER,
    SUPPORTED_EXTENSIONS,
)


# ==========================================================
# Logging
# ==========================================================

LOG_FOLDER.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=PARSER_LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================================
# Find PDFs
# ==========================================================

def find_pdf_files(root_folder: Path):
    """
    Recursively find all PDF files.
    """

    pdf_files = []

    for extension in SUPPORTED_EXTENSIONS:
        pdf_files.extend(root_folder.rglob(f"*{extension}"))

    pdf_files.sort()

    logger.info(f"Found {len(pdf_files)} PDF(s).")

    return pdf_files


# ==========================================================
# Open PDF
# ==========================================================

def open_pdf(pdf_path: Path):
    """
    Safely open a PDF.
    """

    try:

        document = fitz.open(pdf_path)

        logger.info(f"Opened: {pdf_path.name}")

        return document

    except Exception as e:

        logger.error(f"Cannot open {pdf_path.name}: {e}")

        return None


# ==========================================================
# Extract One Page
# ==========================================================

def extract_page(page, page_number):

    text = page.get_text("text")

    return {

        "page_number": page_number,

        "text": text,

        "character_count": len(text),

        "word_count": len(text.split())

    }


# ==========================================================
# Extract Whole Document
# ==========================================================

def extract_document(document, pdf_path: Path):

    pages = []

    for page_index in range(len(document)):

        page = document.load_page(page_index)

        page_data = extract_page(page, page_index + 1)

        pages.append(page_data)

    extracted_document = {

        "document_name": pdf_path.name,

        "source_path": str(pdf_path.relative_to(KNOWLEDGE_BASE)),

        "total_pages": len(document),

        "pages": pages

    }

    logger.info(
        f"Extracted {pdf_path.name} ({len(document)} pages)"
    )

    return extracted_document


# ==========================================================
# Save JSON
# ==========================================================

def save_json(document_dictionary):

    EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)

    output_name = (
        Path(document_dictionary["document_name"]).stem + ".json"
    )

    output_path = EXTRACTED_FOLDER / output_name

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            document_dictionary,
            file,
            indent=4,
            ensure_ascii=False
        )

    logger.info(f"Saved: {output_name}")


# ==========================================================
# Main Pipeline
# ==========================================================

def process_directory():

    pdf_files = find_pdf_files(KNOWLEDGE_BASE)

    successful = 0
    failed = 0

    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):

        document = open_pdf(pdf_path)

        if document is None:

            failed += 1
            continue

        extracted_document = extract_document(
            document,
            pdf_path
        )

        save_json(extracted_document)

        document.close()

        successful += 1

    print()

    print("=" * 50)
    print("Parsing Completed")
    print("=" * 50)

    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")
    print(f"Total PDFs : {len(pdf_files)}")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    process_directory()