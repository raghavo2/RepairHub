from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE = PROJECT_ROOT / "02_Knowledge_Base"

PIPELINE_ROOT = PROJECT_ROOT / "03_Data_Pipeline"

# ==========================================================
# Parser
# ==========================================================

EXTRACTED_FOLDER = PIPELINE_ROOT / "extracted"

# ==========================================================
# Cleaner
# ==========================================================

CLEANED_FOLDER = PIPELINE_ROOT / "cleaned"

# ==========================================================
# Logs
# ==========================================================

LOG_FOLDER = PIPELINE_ROOT / "logs"

PARSER_LOG = LOG_FOLDER / "parser.log"

CLEANER_LOG = LOG_FOLDER / "cleaner.log"

SUPPORTED_EXTENSIONS = [".pdf"]

# ==========================================================
# Chunker
# ==========================================================

CHUNK_FOLDER = PIPELINE_ROOT / "chunks"

# Chunk Size (tokens)
CHUNK_SIZE = 500

# Token overlap
CHUNK_OVERLAP = 75

# Ignore tiny chunks
MIN_CHUNK_SIZE = 100
# ==========================================================
# Embeddings
# ==========================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

VECTOR_DB_PATH = PROJECT_ROOT / "04_Vector_DB"

COLLECTION_NAME = "Repairhub_knowledge_base"