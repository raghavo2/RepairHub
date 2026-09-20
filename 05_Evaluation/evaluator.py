import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add 03_Data_Pipeline to python path to import Retriever
CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent / "03_Data_Pipeline"
sys.path.append(str(PIPELINE_DIR))

from retriever import DocumentRetriever

# Setup logging
LOG_FILE = CURRENT_DIR / "eval.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("RAGEvaluator")


class RAGEvaluator:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.retriever = DocumentRetriever()

    def load_dataset(self) -> List[Dict]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, top_k_values: List[int] = [1, 3, 5]) -> Dict:
        dataset = self.load_dataset()
        total_queries = len(dataset)

        if total_queries == 0:
            print("Dataset is empty.")
            return {}

        results_by_k = {}

        for k in top_k_values:
            hits = 0
            reciprocal_ranks = []
            precision_scores = []

            print(f"\n--- Evaluating @ k={k} ---")

            for item in dataset:
                query_id = item["query_id"]
                query = item["query"]
                expected_doc = item["expected_doc_stem"]
                filters = item.get("filters", None)

                # Execute hybrid retrieval
                retrieved_chunks = self.retriever.retrieve(
                    query=query, top_k=k, metadata_filters=filters
                )

                # Extract retrieved document stems
                retrieved_docs = [
                    chunk["metadata"].get("document_stem", "")
                    for chunk in retrieved_chunks
                ]

                # 1. Hit Rate / Recall check
                is_hit = expected_doc in retrieved_docs

                if is_hit:
                    hits += 1
                    rank = retrieved_docs.index(expected_doc) + 1
                    reciprocal_ranks.append(1.0 / rank)
                    precision_scores.append(1.0 / k)
                else:
                    reciprocal_ranks.append(0.0)
                    precision_scores.append(0.0)
                    
                    # Diagnostics for Missed Queries
                    print(f"  [MISS] Query ID: {query_id}")
                    print(f"         Expected: {expected_doc}")
                    print(f"         Got Docs: {list(dict.fromkeys(retrieved_docs))[:3]}") # Show top 3 unique docs

            hit_rate = hits / total_queries
            mrr = sum(reciprocal_ranks) / total_queries
            mean_precision = sum(precision_scores) / total_queries

            results_by_k[k] = {
                "Hit_Rate": round(hit_rate, 4),
                "MRR": round(mrr, 4),
                "Precision": round(mean_precision, 4),
            }

            print(f"\nRecall/Hit Rate @ {k} : {hit_rate * 100:.2f}%")
            print(f"MRR @ {k}             : {mrr:.4f}")
            print(f"Precision @ {k}       : {mean_precision:.4f}\n")

        return results_by_k


def main():
    dataset_path = CURRENT_DIR / "test_dataset.json"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("=" * 60)
    print("Repairhub AI - RAG Retrieval Evaluation Engine")
    print("=" * 60)

    evaluator = RAGEvaluator(dataset_path)
    evaluator.evaluate(top_k_values=[1, 3, 5])


if __name__ == "__main__":
    main()