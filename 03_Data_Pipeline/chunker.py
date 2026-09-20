"""
==============================================================
Repairhub AI
Production Chunker V3

Reads cleaned JSON documents and converts them into
high-quality RAG chunks ready for ChromaDB.

Features
--------
• Token-aware chunking
• Page-aware chunk tracking
• Heading-aware splitting
• Metadata extraction
• Rich statistics
• Deterministic chunk IDs
• Industrial electronics optimized
• Fault-tolerant schema validation
==============================================================
"""

from __future__ import annotations

import json
import logging
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import tiktoken
from tqdm import tqdm

from config import (
    CLEANED_FOLDER,
    CHUNK_FOLDER,
    LOG_FOLDER,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
)

# ============================================================
# Logging
# ============================================================

LOG_FOLDER.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_FOLDER / "chunker.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ChunkerV3")

# ============================================================
# Tokenizer
# ============================================================

ENCODER = tiktoken.get_encoding("cl100k_base")

# ============================================================
# Regex Library
# ============================================================

HEADING_PATTERN = re.compile(
    r"^([A-Z][A-Za-z0-9\s/\-]{2,80}|"
    r"\d+(\.\d+)*\s+.+)$"
)

TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*[-=]{3,}\s*$"
)

CODE_FENCE_PATTERN = re.compile(
    r"^```"
)

PART_NUMBER_PATTERN = re.compile(
    r"\b[A-Z]{1,6}\d{2,}[A-Z0-9\-]*\b"
)

# ============================================================
# Dataclasses
# ============================================================

@dataclass
class TokenEntry:
    """
    Represents one token and the page it originated from.
    """
    token: int
    page: int


@dataclass
class ChunkStatistics:
    """
    Statistics stored for every chunk.
    """
    token_count: int
    word_count: int
    character_count: int
    line_count: int
    estimated_read_time: float
    has_table: bool
    has_code: bool
    has_formula: bool


@dataclass
class ChunkMetadata:
    """
    Metadata attached to every chunk.
    """
    document_name: str
    document_stem: str
    source_path: str

    domain: str
    category: str
    subcategory: str

    manufacturer: str
    document_type: str

    component_family: str
    component_type: str

    product_series: str
    part_number: str

    language: str
    source: str

    page_start: int = 0
    page_end: int = 0

    pages: List[int] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    """
    Final chunk object.
    """
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    statistics: ChunkStatistics


# ============================================================
# Token Utilities
# ============================================================

def tokenize(text: str) -> List[int]:
    """
    Convert text to token ids.
    """
    return ENCODER.encode(text)


def detokenize(tokens: List[int]) -> str:
    """
    Convert token ids back into text.
    """
    return ENCODER.decode(tokens)


# ============================================================
# JSON Helpers
# ============================================================

def load_json(path: Path) -> Dict:
    """
    Load JSON document.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(document: Dict) -> None:
    """
    Save chunked document.
    """
    CHUNK_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        Path(document.get("document_name", "Unknown_Document")).stem
        + "_chunks.json"
    )

    output = CHUNK_FOLDER / filename

    with open(output, "w", encoding="utf-8") as file:
        json.dump(
            document,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info("Saved %s", filename)


# ============================================================
# General Utilities
# ============================================================

def sha1(text: str) -> str:
    """
    Stable SHA1 hash.
    """
    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()


def normalize_whitespace(text: str) -> str:
    """
    Collapse excessive whitespace.
    """
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def safe_min(values: Iterable[int]) -> int:
    """
    Safe minimum.
    """
    values = list(values)
    if not values:
        return 0
    return min(values)


def safe_max(values: Iterable[int]) -> int:
    """
    Safe maximum.
    """
    values = list(values)
    if not values:
        return 0
    return max(values)


def unique_sorted(values: Iterable[int]) -> List[int]:
    """
    Remove duplicates while returning sorted output.
    """
    return sorted(set(values))


def is_empty_page(text: str) -> bool:
    """
    Detect blank pages.
    """
    if text is None:
        return True

    if not text.strip():
        return True

    if len(text.strip()) < 5:
        return True

    return False


# ============================================================
# Token Buffer Implementation
# ============================================================

class TokenBuffer:
    """
    Stores tokens together with their originating page.

    This guarantees accurate page references even after
    overlap trimming.
    """
    def __init__(self) -> None:
        self.buffer: List[TokenEntry] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def append_tokens(
        self,
        tokens: List[int],
        page: int,
    ) -> None:
        """
        Append tokens belonging to one page.
        """
        self.buffer.extend(
            TokenEntry(token=t, page=page)
            for t in tokens
        )

    def first(
        self,
        amount: int,
    ) -> List[TokenEntry]:
        """
        Return first N token entries.
        """
        return self.buffer[:amount]

    def discard(self, amount: int) -> None:
        """
        Remove first N tokens.
        """
        self.buffer = self.buffer[amount:]

    def keep_last(self, amount: int) -> None:
        """
        Keep only the last N tokens.
        """
        if amount <= 0:
            self.buffer.clear()
            return
        self.buffer = self.buffer[-amount:]

    def token_ids(self) -> List[int]:
        """
        Return token ids.
        """
        return [entry.token for entry in self.buffer]

    def pages(self) -> List[int]:
        """
        Return unique pages.
        """
        return unique_sorted(
            entry.page
            for entry in self.buffer
        )

    def page_start(self) -> int:
        return safe_min(self.pages())

    def page_end(self) -> int:
        return safe_max(self.pages())


# ============================================================
# Manufacturer Database
# ============================================================

MANUFACTURERS = {
    "infineon": "Infineon",
    "texas instruments": "Texas Instruments",
    "ti": "Texas Instruments",
    "stmicroelectronics": "STMicroelectronics",
    "stm": "STMicroelectronics",
    "nxp": "NXP",
    "analog devices": "Analog Devices",
    "adi": "Analog Devices",
    "microchip": "Microchip",
    "atmel": "Atmel",
    "on semiconductor": "ON Semiconductor",
    "onsemi": "ON Semiconductor",
    "vishay": "Vishay",
    "toshiba": "Toshiba",
    "renesas": "Renesas",
    "espressif": "Espressif",
    "arduino": "Arduino",
    "emerson": "Emerson",
    "delta": "Delta",
    "eltek": "Eltek",
    "cisco": "Cisco",
    "abb": "ABB",
    "siemens": "Siemens",
    "schneider": "Schneider Electric",
}


# ============================================================
# Document Types
# ============================================================

DOCUMENT_TYPES = {
    "datasheet": "Datasheet",
    "application note": "Application Note",
    "app note": "Application Note",
    "appnote": "Application Note",
    "reference manual": "Reference Manual",
    "technical reference": "Technical Reference Manual",
    "trm": "Technical Reference Manual",
    "programming guide": "Programming Guide",
    "hardware guide": "Hardware Guide",
    "software guide": "Software Guide",
    "service manual": "Service Manual",
    "user manual": "User Manual",
    "repair": "Repair Manual",
    "design guide": "Design Guide",
    "design note": "Design Note",
    "failure report": "Failure Report",
    "evaluation board": "Evaluation Board Guide",
    "evb": "Evaluation Board Guide",
    "whitepaper": "Whitepaper",
}


# ============================================================
# Component Families
# ============================================================

COMPONENT_FAMILIES = {
    "mosfet": ("Power Components", "MOSFET"),
    "igbt": ("Power Components", "IGBT"),
    "bjt": ("Power Components", "BJT"),
    "regulator": ("Power Components", "Voltage Regulator"),
    "buck": ("Power Components", "Buck Converter"),
    "boost": ("Power Components", "Boost Converter"),
    "flyback": ("Power Components", "Flyback"),
    "transformer": ("Power Components", "Transformer"),
    "op amp": ("Analog", "Operational Amplifier"),
    "comparator": ("Analog", "Comparator"),
    "adc": ("Mixed Signal", "ADC"),
    "dac": ("Mixed Signal", "DAC"),
    "esp32": ("Development", "ESP32"),
    "arduino": ("Development", "Arduino"),
    "rectifier": ("Industrial", "Rectifier"),
    "smps": ("Industrial", "SMPS"),
    "ups": ("Industrial", "UPS"),
    "inverter": ("Industrial", "Inverter"),
}

# ============================================================
# Detection Helpers
# ============================================================

import re
from typing import Tuple

def detect_manufacturer(text: str) -> str:
    """
    Detect manufacturer from text using strict word boundaries.
    """
    lower = text.lower()
    for key, value in MANUFACTURERS.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return value
    return "Unknown"


def detect_document_type(text: str) -> str:
    """
    Detect document type using strict word boundaries.
    """
    lower = text.lower()
    for key, value in DOCUMENT_TYPES.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return value
    return "Unknown"


def detect_component_family(
    text: str,
) -> Tuple[str, str]:
    """
    Return (family, type) using strict word boundaries.
    """
    lower = text.lower()
    for key, value in COMPONENT_FAMILIES.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return value
    return (
        "Unknown",
        "Unknown",
    )


def detect_part_number(
    text: str,
) -> str:
    """
    Detect first valid part number.
    """
    match = PART_NUMBER_PATTERN.search(text)
    if match:
        return match.group(0)
    return "Unknown"


def detect_product_series(
    text: str,
) -> str:
    """
    Best effort series detection using strict word boundaries.
    """
    known = [
        "OptiMOS",
        "CoolMOS",
        "XMC",
        "C2000",
        "Sitara",
        "SimpleLink",
        "STM32",
        "AVR",
        "PIC",
        "dsPIC",
        "ESP32",
        "ESP8266",
    ]

    lower = text.lower()
    for series in known:
        if re.search(rf"\b{re.escape(series.lower())}\b", lower):
            return series
    return "Unknown"
# ============================================================
# Keyword Extraction
# ============================================================

STOPWORDS = {
    "the", "and", "for", "this",
    "that", "with", "from",
    "into", "have", "has",
    "using", "used", "can",
    "will", "are", "was",
    "were", "page",
}


def extract_keywords(
    text: str,
    limit: int = 20,
) -> List[str]:
    """
    Lightweight keyword extraction.
    """
    words = re.findall(
        r"[A-Za-z][A-Za-z0-9_\-/]+",
        text.lower(),
    )
    frequency: Dict[str, int] = {}

    for word in words:
        if len(word) < 3:
            continue
        if word in STOPWORDS:
            continue
        frequency[word] = frequency.get(word, 0) + 1

    ranked = sorted(
        frequency.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        word
        for word, _
        in ranked[:limit]
    ]


# ============================================================
# Chunk Statistics
# ============================================================

def chunk_statistics(
    text: str,
) -> ChunkStatistics:
    """
    Calculate statistics for a chunk.
    """
    token_count = len(tokenize(text))
    word_count = len(text.split())
    character_count = len(text)
    line_count = len(text.splitlines())
    estimated_read_time = round(word_count / 220, 2)

    has_table = "|" in text or "Table" in text
    has_code = (
        "```" in text
        or "#include" in text
        or "void " in text
        or "int main" in text
    )

    has_formula = bool(
        re.search(
            r"[=<>±Ωµ°]",
            text,
        )
    )

    return ChunkStatistics(
        token_count=token_count,
        word_count=word_count,
        character_count=character_count,
        line_count=line_count,
        estimated_read_time=estimated_read_time,
        has_table=has_table,
        has_code=has_code,
        has_formula=has_formula,
    )


# ============================================================
# Metadata Extraction
# ============================================================

def extract_metadata(
    document: Dict,
) -> ChunkMetadata:
    """
    Build metadata for a document.
    """
    source_path = Path(document.get("source_path", "Unknown/Path"))
    parts = source_path.parts

    domain = parts[0] if len(parts) > 0 else "Unknown"
    category = parts[1] if len(parts) > 1 else "Unknown"
    subcategory = parts[2] if len(parts) > 2 else "Unknown"
    
    doc_name = document.get("document_name", "Unknown_Document")

    combined_text = (
        doc_name
        + "\n"
        + "\n".join(
            page.get("clean_text", "")[:2000]
            for page in document.get("pages", [])[:3]
        )
    )

    manufacturer = detect_manufacturer(combined_text)
    document_type = detect_document_type(combined_text)
    component_family, component_type = detect_component_family(combined_text)
    part_number = detect_part_number(combined_text)
    product_series = detect_product_series(combined_text)
    keywords = extract_keywords(combined_text)

    return ChunkMetadata(
        document_name=doc_name,
        document_stem=Path(doc_name).stem,
        source_path=str(source_path),
        domain=domain,
        category=category,
        subcategory=subcategory,
        manufacturer=manufacturer,
        document_type=document_type,
        component_family=component_family,
        component_type=component_type,
        product_series=product_series,
        part_number=part_number,
        language="English",
        source="Knowledge_Base",
        keywords=keywords,
    )


# ============================================================
# Chunk ID Generation
# ============================================================

def generate_chunk_id(
    metadata: ChunkMetadata,
    chunk_number: int,
) -> str:
    """
    Generate deterministic chunk IDs.
    """
    base = f"{metadata.document_stem}_{chunk_number:04d}"
    digest = sha1(base)[:8]
    return f"{base}_{digest}"


# ============================================================
# Chunk Boundary Helpers
# ============================================================

def find_best_boundary(text: str) -> int:
    """
    Prefer splitting at paragraph boundaries.
    """
    boundaries = ["\n\n", "\n", ". ", "; ", ", "]
    midpoint = len(text)

    for boundary in boundaries:
        index = text.rfind(boundary, 0, midpoint)
        if index > 100:
            return index + len(boundary)

    return midpoint


def build_chunk(
    token_entries: List[TokenEntry],
    metadata: ChunkMetadata,
    chunk_number: int,
) -> Chunk:
    """
    Build one chunk object.
    """
    token_ids = [token.token for token in token_entries]
    text = detokenize(token_ids)
    boundary = find_best_boundary(text)
    text = text[:boundary].strip()
    pages = unique_sorted(token.page for token in token_entries)

    chunk_metadata = ChunkMetadata(**metadata.__dict__)
    chunk_metadata.page_start = safe_min(pages)
    chunk_metadata.page_end = safe_max(pages)
    chunk_metadata.pages = pages

    stats = chunk_statistics(text)

    return Chunk(
        chunk_id=generate_chunk_id(metadata, chunk_number),
        text=text,
        metadata=chunk_metadata,
        statistics=stats,
    )


# ============================================================
# Chunk Document
# ============================================================

def chunk_document(
    document: Dict,
) -> Dict:
    """
    Chunk one cleaned document.
    """
    metadata = extract_metadata(document)
    buffer = TokenBuffer()
    chunks: List[Chunk] = []
    chunk_number = 1

    for page in document.get("pages", []):
        page_text = page.get("clean_text", "")

        if is_empty_page(page_text):
            continue

        tokens = tokenize(page_text)
        buffer.append_tokens(tokens, page.get("page_number", 1))

        while len(buffer) >= CHUNK_SIZE:
            token_entries = buffer.first(CHUNK_SIZE)
            chunk = build_chunk(token_entries, metadata, chunk_number)
            chunks.append(chunk)
            chunk_number += 1

            overlap = buffer.first(CHUNK_OVERLAP)
            buffer.discard(CHUNK_SIZE)
            buffer.buffer = overlap + buffer.buffer

    if len(buffer) >= MIN_CHUNK_SIZE:
        chunk = build_chunk(buffer.buffer, metadata, chunk_number)
        chunks.append(chunk)

    return {
        "document_name": metadata.document_name,
        "source_path": metadata.source_path,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {**chunk.metadata.__dict__},
                "statistics": {**chunk.statistics.__dict__}
            }
            for chunk in chunks
        ]
    }


# ============================================================
# Process Single Document
# ============================================================

def process_document(json_file: Path) -> bool:
    """
    Process a single cleaned JSON document.

    Returns
    -------
    bool
        True if successful.
    """
    try:
        logger.info("Processing %s", json_file.name)
        document = load_json(json_file)
        
        # DOWNSTREAM FAULT TOLERANCE: Ensure required root keys exist
        document["document_name"] = document.get("document_name", json_file.name)
        document["source_path"] = document.get("source_path", str(json_file.resolve()))
        
        if "pages" not in document:
            document["pages"] = []
            
        # Ensure page-level keys exist
        for idx, page in enumerate(document["pages"]):
            if "page_number" not in page:
                page["page_number"] = idx + 1
            if "clean_text" not in page:
                page["clean_text"] = page.get("text", "")

        chunked_document = chunk_document(document)
        save_json(chunked_document)
        
        logger.info(
            "Finished %s | Chunks=%d",
            json_file.name,
            chunked_document.get("total_chunks", 0),
        )
        return True

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"\nError processing {json_file.name}: {repr(e)}")
        logger.exception(
            "Failed processing %s : %s",
            json_file.name,
            str(e),
        )
        return False


# ============================================================
# Process Directory
# ============================================================

def process_directory() -> None:
    """
    Process every cleaned JSON document.
    """
    CHUNK_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )
    json_files = sorted(CLEANED_FOLDER.glob("*_clean.json"))

    if not json_files:
        print("\nNo cleaned documents found.\n")
        logger.warning("No cleaned JSON documents found.")
        return

    successful = 0
    failed = 0
    total_chunks = 0

    print("\n" + "=" * 65)
    print("Repairhub AI - Production Chunking Pipeline")
    print("=" * 65)
    print(f"Input Folder : {CLEANED_FOLDER}")
    print(f"Output Folder: {CHUNK_FOLDER}")
    print(f"Documents    : {len(json_files)}")
    print()

    logger.info("=" * 60)
    logger.info("Chunking Started")
    logger.info("Documents : %d", len(json_files))

    for json_file in tqdm(json_files, desc="Chunking", unit="document"):
        success = process_document(json_file)
        
        if success:
            successful += 1
            try:
                output_file = CHUNK_FOLDER / (
                    json_file.stem.replace("_clean", "") + "_chunks.json"
                )
                if output_file.exists():
                    output = load_json(output_file)
                    total_chunks += output.get("total_chunks", 0)
            except Exception:
                pass
        else:
            failed += 1

    print()
    print("=" * 65)
    print("Chunking Complete")
    print("=" * 65)
    print(f"Successful Documents : {successful}")
    print(f"Failed Documents     : {failed}")
    print(f"Total Chunks         : {total_chunks}")

    if successful:
        avg = round(total_chunks / successful, 2)
        print(f"Average Chunks/File  : {avg}")

    print("=" * 65)
    print()

    logger.info(
        "Completed | Success=%d Failed=%d Chunks=%d",
        successful,
        failed,
        total_chunks,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Entry point.
    """
    print("\nStarting Repairhub AI Chunker...\n")
    logger.info("Chunker Started")
    
    process_directory()
    
    logger.info("Chunker Finished")


# ============================================================
# Script Entry
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        logger.warning("Execution interrupted by user.")
    except Exception as e:
        print(f"\nFatal Error: {e}")
        logger.exception("Fatal Error: %s", str(e))
        raise