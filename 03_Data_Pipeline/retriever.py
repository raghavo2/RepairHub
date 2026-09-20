import logging
from typing import List, Dict, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    MODEL_NAME,
)

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ==========================================================
# Retriever Class
# ==========================================================

class DocumentRetriever:
    def __init__(self):
        """
        Initializes the embedding model and connects to the ChromaDB client.
        """
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded.")

        logger.info("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            logger.info(f"Successfully connected to collection: {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to load collection '{COLLECTION_NAME}'. Error: {e}")
            self.collection = None

    def _build_chroma_filter(self, metadata_filters: Optional[Dict]) -> Optional[Dict]:
        """
        Converts a standard Python dictionary into ChromaDB's required filter syntax.
        """
        if not metadata_filters:
            return None
            
        valid_filters = {k: v for k, v in metadata_filters.items() if v}
        
        if not valid_filters:
            return None
            
        if len(valid_filters) == 1:
            key, value = list(valid_filters.items())[0]
            return {key: {"$eq": value}}
            
        return {
            "$and": [
                {k: {"$eq": v}} for k, v in valid_filters.items()
            ]
        }

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Executes a hybrid search: Semantic search strictly bounded by metadata filters.
        """
        if not self.collection:
            logger.warning("Retrieval failed: Collection is not initialized.")
            return []

        # 1. Compile the structured lookup constraints
        chroma_filter = self._build_chroma_filter(metadata_filters)

        if chroma_filter:
            logger.info(f"Applying structured filters: {chroma_filter}")

        # 2. Encode the semantic search query WITH the BGE instruction prefix
        bge_query = f"Represent this sentence for searching relevant passages: {query}"
        
        query_embedding = self.model.encode(
            bge_query,
            normalize_embeddings=True,
        ).tolist()

        # 3. Execute hybrid query in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=chroma_filter
        )

        retrieved = []

        if not results["ids"] or not results["ids"][0]:
            return retrieved

        # 4. Parse outputs
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return retrieved

# ==========================================================
# Interactive Hybrid Testing
# ==========================================================

if __name__ == "__main__":
    
    retriever = DocumentRetriever()

    print("\n" + "=" * 60)
    print("Repairhub AI - Interactive Hybrid Retrieval")
    print("Type 'exit' or 'quit' at the query prompt to stop.")
    print("=" * 60)

    while True:
        try:
            user_query = input("\nEnter search query: ").strip()
            
            if user_query.lower() in ['exit', 'quit']:
                print("\nExiting search. Goodbye!")
                break
            if not user_query:
                continue

            print("\n--- Optional Metadata Filters (Press Enter to skip) ---")
            mfg_filter = input("Manufacturer (e.g., 'Texas Instruments'): ").strip()
            comp_filter = input("Component Family (e.g., 'MOSFET'): ").strip()
            
            agent_filters = {}
            if mfg_filter:
                agent_filters["manufacturer"] = mfg_filter
            if comp_filter:
                agent_filters["component_family"] = comp_filter

            print(f"\nSearching for: '{user_query}'...")
            
            results = retriever.retrieve(
                query=user_query, 
                top_k=3, 
                metadata_filters=agent_filters
            )

            if not results:
                print("\n[!] No matching documents found in the database for those specific constraints.")
                continue

            for i, result in enumerate(results, start=1):
                print("\n" + "=" * 80)
                print(f"Result #{i} | ID: {result['id']} | Distance: {result['distance']:.4f}")
                print("-" * 80)
                print(f"Manufacturer : {result['metadata'].get('manufacturer', 'N/A')}")
                print(f"Component    : {result['metadata'].get('component_family', 'N/A')} - {result['metadata'].get('component_type', 'N/A')}")
                print(f"Document     : {result['metadata'].get('document_stem', 'N/A')}")
                print("-" * 80)
                print(result["text"][:350].replace('\n', ' ') + "...\n")

        except KeyboardInterrupt:
            print("\n\nProcess interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred during search: {e}")