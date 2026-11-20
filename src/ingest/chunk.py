"""Чанкування документів фіксованого розміру з перекриттям (Лабораторна робота № 2)."""

from __future__ import annotations

from typing import Any


def chunk_document(
    doc: dict[str, Any],
    chunk_size_words: int = 300,
    overlap_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    """Розбиває документ на чанки фіксованого розміру за словами з перекриттям.

    Кожен чанк містить:
      - chunk_id: стабільний глобальний ідентифікатор (simplewiki_{doc_id}_c{idx:03d});
      - doc_id: ідентифікатор документа-джерела;
      - chunk_index: порядковий номер чанка у документі (з 0);
      - title, url: метадані джерела для подальшого цитування;
      - word_count, char_count, text: вміст та розмір фрагмента.
    """
    text = doc.get("text", "").strip()
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    overlap_words = max(1, int(round(chunk_size_words * overlap_ratio)))
    step = max(1, chunk_size_words - overlap_words)

    doc_id = str(doc["doc_id"])
    title = str(doc.get("title", ""))
    url = str(doc.get("url", ""))

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(
            {
                "chunk_id": f"simplewiki_{doc_id}_c{chunk_index:03d}",
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "title": title,
                "url": url,
                "word_count": len(chunk_words),
                "char_count": len(chunk_text),
                "text": chunk_text,
            }
        )

        if end >= len(words):
            break

        # Якщо залишок після перекриття вже повністю входить у поточний чанк, зупиняємося
        next_start = start + step
        if next_start >= len(words):
            break
        start = next_start
        chunk_index += 1

    return chunks
