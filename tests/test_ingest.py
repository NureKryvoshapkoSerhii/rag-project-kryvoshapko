"""Базові модульні тести для пайплайну очищення та чанкування (ЛБ-2)."""

import unittest

from src.ingest.chunk import chunk_document
from src.ingest.clean import clean_text, is_cs_article, is_valid_document


class TestIngestPipeline(unittest.TestCase):
    def test_clean_text_removes_html_and_normalizes_whitespace(self) -> None:
        raw = "<p>Python is a <b>programming</b>&nbsp;language.</p>\r\n\r\n\r\n<script>bad()</script>  Simple."
        cleaned = clean_text(raw)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("bad()", cleaned)
        self.assertIn("Python is a", cleaned)
        self.assertIn("Simple.", cleaned)

    def test_is_valid_document_min_length(self) -> None:
        self.assertFalse(is_valid_document("Short text.", min_char_length=100))
        self.assertTrue(is_valid_document("a" * 100, min_char_length=100))

    def test_is_cs_article_filter(self) -> None:
        self.assertTrue(
            is_cs_article(
                "Python (programming language)",
                "Python is a high-level programming language created by Guido van Rossum.",
            )
        )
        self.assertFalse(
            is_cs_article(
                "Borscht",
                "Borscht is a sour soup common in Eastern Europe made with beetroots.",
            )
        )

    def test_chunk_document_fixed_size_and_overlap(self) -> None:
        words = [f"word{i}" for i in range(600)]
        doc = {
            "doc_id": "42",
            "title": "Test Article",
            "url": "https://simple.wikipedia.org/wiki/Test_Article",
            "text": " ".join(words),
        }
        chunks = chunk_document(doc, chunk_size_words=300, overlap_ratio=0.15)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["chunk_id"], "simplewiki_42_c000")
        self.assertEqual(chunks[1]["chunk_id"], "simplewiki_42_c001")
        self.assertEqual(chunks[0]["word_count"], 300)
        # Перекриття 15% від 300 слів = 45 слів: чанк 1 має починатися зі слова word255
        self.assertTrue(chunks[1]["text"].startswith("word255 "))


if __name__ == "__main__":
    unittest.main()
