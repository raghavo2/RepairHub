from pathlib import Path
import json
import logging

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    CHUNK_FOLDER,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    MODEL_NAME,
    LOG_FOLDER,
)

# ==========================================================
# Logging
# ==========================================================

LOG_FILE = LOG_FOLDER / "embedding.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# Load Embedding Model
# ==========================================================

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model Loaded.")

# ==========================================================
# ChromaDB Initialization
# ==========================================================

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# ==========================================================
# JSON Helpers
# ==========================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

# ==========================================================
# Document Processing
# ==========================================================

def process_document(document, file_name: str):
    """
    Extracts chunks, sanitizes metadata, and batches ingestion into ChromaDB.
    """
    ids = []
    texts = []
    metadatas = []

    for chunk in document.get("chunks", []):
        ids.append(chunk["chunk_id"])
        texts.append(chunk["text"])

        # ChromaDB strictly rejects lists/dicts in metadata.
        # We must flatten keywords and pages into strings.
        sanitized_metadata = {}
        for key, value in chunk.get("metadata", {}).items():
            if isinstance(value, list):
                sanitized_metadata[key] = ", ".join(map(str, value))
            elif value is None:
                sanitized_metadata[key] = ""
            else:
                sanitized_metadata[key] = value
                
        metadatas.append(sanitized_metadata)

    # Batch processing to avoid memory spikes and Chroma payload limits
    BATCH_SIZE = 500
    
    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_metadatas = metadatas[i : i + BATCH_SIZE]

        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metadatas
        )

# ==========================================================
# Main Execution
# ==========================================================

def main():
    json_files = sorted(
        CHUNK_FOLDER.glob("*_chunks.json")
    )

    if not json_files:
        print("\nNo chunked documents found.\n")
        return

    processed = 0
    failed = 0

    for json_file in tqdm(json_files, desc="Embedding", unit="doc"):
        try:
            document = load_json(json_file)
            process_document(document, json_file.name)
            processed += 1
            logger.info(f"Successfully embedded {json_file.name}")
        except Exception as e:
            failed += 1
            logger.error(f"Failed to process {json_file.name}: {str(e)}")
            print(f"\nError processing {json_file.name}: {e}")

    print()
    print("=" * 60)
    print("Vector Database Ingestion Complete")
    print("=" * 60)
    print(f"Documents Successfully Embedded : {processed}")
    print(f"Documents Failed                : {failed}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")