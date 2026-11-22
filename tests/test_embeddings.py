"""Модульні та інтеграційні тести для векторних ембедингів і ChromaDB (ЛБ-3)."""

import json
from pathlib import Path
import unittest

from src.embeddings.build_index import (
    ROOT_DIR,
    format_chunk_for_embedding,
    load_config,
    resolve_chroma_path,
)
from src.embeddings.search import distance_to_similarity
from src.eval.eval_retrieval import EVAL_SET_PATH, compute_recall_at_k, load_eval_set


class TestLab3Embeddings(unittest.TestCase):
    def test_eval_set_schema_and_ground_truth_exist(self) -> None:
        eval_items = load_eval_set(EVAL_SET_PATH)
        self.assertGreaterEqual(len(eval_items), 15)
        self.assertLessEqual(len(eval_items), 25)

        chunks_path = ROOT_DIR / load_config()["chunking"]["output_path"]
        chunk_ids: set[str] = set()
        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunk_ids.add(json.loads(line)["chunk_id"])

        seen_qids: set[str] = set()
        for item in eval_items:
            self.assertNotIn(item["query_id"], seen_qids)
            seen_qids.add(item["query_id"])
            self.assertTrue(item["query"].strip())
            self.assertIn(item["relevant_chunk_id"], chunk_ids)

    def test_format_chunk_for_embedding_title_prefix(self) -> None:
        chunk = {"title": "Compiler", "text": "A parser builds an AST."}
        self.assertEqual(
            format_chunk_for_embedding(chunk, prepend_title=True),
            "Compiler\nA parser builds an AST.",
        )
        self.assertEqual(
            format_chunk_for_embedding(chunk, prepend_title=False),
            "A parser builds an AST.",
        )

    def test_distance_to_similarity_and_recall_metric(self) -> None:
        self.assertAlmostEqual(distance_to_similarity(0.25, "cosine"), 0.75)
        self.assertAlmostEqual(distance_to_similarity(0.0, "l2"), 1.0)
        self.assertEqual(compute_recall_at_k(["a", "b", "c"], "b", 2), 1.0)
        self.assertEqual(compute_recall_at_k(["a", "b", "c"], "c", 2), 0.0)

    def test_persisted_chroma_index_metadata_and_count(self) -> None:
        cfg = load_config()
        persist_dir = ROOT_DIR / cfg["vector_store"]["persist_directory"]
        if not persist_dir.exists():
            self.skipTest("Векторний індекс data/chroma_db ще не побудовано")

        import chromadb

        client = chromadb.PersistentClient(path=resolve_chroma_path(persist_dir))
        col = client.get_collection(name=cfg["vector_store"]["collection_name"])
        self.assertEqual(col.count(), 2225)

        sample = col.peek(limit=1)
        meta = sample["metadatas"][0]
        for key in ("doc_id", "chunk_index", "title", "url", "word_count", "char_count"):
            self.assertIn(key, meta)


if __name__ == "__main__":
    unittest.main()
