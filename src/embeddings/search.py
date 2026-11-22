"""CLI-скрипт семантичного пошуку у векторному індексі ChromaDB (Лабораторна робота № 3).

Приклади запуску із кореня репозиторію:
    python -m src.embeddings.search "How does TCP deliver packets reliably?"
    python -m src.embeddings.search "What is a hash table?" --top-k 5
    python -m src.embeddings.search --demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.embeddings.build_index import ROOT_DIR, load_config, resolve_chroma_path

DEMO_QUERIES: list[str] = [
    "How does TCP make sure network packets arrive in the right order?",
    "How does a hash table use a hash function for fast lookup?",
    "What is the difference between var, let, and const in JavaScript?",
    "What three guarantees are described by Eric Brewer's CAP theorem?",
    "How do public and private keys work in asymmetric RSA encryption?",
]


def distance_to_similarity(distance: float, metric: str = "cosine") -> float:
    """Перетворює відстань ChromaDB у зрозумілу оцінку подібності (вище = ближче)."""
    if metric in ("cosine", "ip"):
        return round(1.0 - float(distance), 4)
    # Для метрики L2 (квадрат евклідової відстані) використовуємо 1 / (1 + d)
    return round(1.0 / (1.0 + max(0.0, float(distance))), 4)


def search_dense(
    query: str,
    top_k: int = 5,
    *,
    cfg: dict[str, Any] | None = None,
    model: SentenceTransformer | None = None,
    collection: Any | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Виконує щільний векторний пошук у ChromaDB та повертає top-K чанків і час (мс)."""
    cfg = cfg or load_config()
    emb_cfg = cfg["embeddings"]
    vs_cfg = cfg["vector_store"]

    if model is None:
        model = SentenceTransformer(
            emb_cfg["model_name"],
            device=emb_cfg.get("device", "cpu"),
        )
    if collection is None:
        persist_dir = ROOT_DIR / vs_cfg["persist_directory"]
        client = chromadb.PersistentClient(path=resolve_chroma_path(persist_dir))
        collection = client.get_collection(name=vs_cfg["collection_name"])

    metric = (collection.metadata or {}).get(
        "hnsw:space", vs_cfg.get("distance_metric", "cosine")
    )
    use_norm = bool(emb_cfg.get("normalize_embeddings", True))

    t0 = time.perf_counter()
    query_vec = model.encode(
        [query],
        normalize_embeddings=use_norm,
        convert_to_numpy=True,
    )[0].tolist()

    raw = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]

    results: list[dict[str, Any]] = []
    for rank, (cid, doc_text, meta, dist) in enumerate(
        zip(ids, docs, metas, dists), start=1
    ):
        results.append(
            {
                "rank": rank,
                "chunk_id": cid,
                "doc_id": meta.get("doc_id", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "similarity": distance_to_similarity(dist, metric=metric),
                "distance": round(float(dist), 4),
                "text": doc_text,
            }
        )

    return results, round(elapsed_ms, 2)


def print_results(query: str, results: list[dict[str, Any]], elapsed_ms: float) -> None:
    """Друкує результати пошуку у зручному для читання консольному форматі."""
    print(f'\nЗапит: "{query}" ({elapsed_ms:.1f} мс)')
    print("-" * 78)
    for item in results:
        snippet = item["text"].replace("\n", " ").strip()
        if len(snippet) > 155:
            snippet = snippet[:152].rstrip() + "..."
        print(
            f"  #{item['rank']} | {item['chunk_id']} | score={item['similarity']:.4f} | "
            f"{item['title']} (chunk {item['chunk_index']})"
        )
        print(f"      {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Семантичний пошук у векторному індексі ChromaDB (ЛБ-3)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Пошуковий запит англійською мовою",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Кількість найближчих чанків у видачі (за замовчуванням 5)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Запустити демонстрацію пошуку на 5 тестових запитах",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вивести результати у форматі JSON",
    )
    args = parser.parse_args()

    if not args.demo and not args.query:
        parser.error("Вкажіть текст запиту або прапорець --demo.")

    cfg = load_config()
    emb_cfg = cfg["embeddings"]
    vs_cfg = cfg["vector_store"]

    model = SentenceTransformer(
        emb_cfg["model_name"],
        device=emb_cfg.get("device", "cpu"),
    )
    client = chromadb.PersistentClient(
        path=resolve_chroma_path(ROOT_DIR / vs_cfg["persist_directory"])
    )
    collection = client.get_collection(name=vs_cfg["collection_name"])

    queries = DEMO_QUERIES if args.demo else [args.query]
    all_outputs: list[dict[str, Any]] = []

    for q in queries:
        results, elapsed_ms = search_dense(
            q,
            top_k=args.top_k,
            cfg=cfg,
            model=model,
            collection=collection,
        )
        if args.json:
            all_outputs.append({"query": q, "elapsed_ms": elapsed_ms, "results": results})
        else:
            print_results(q, results, elapsed_ms)

    if args.json:
        print(json.dumps(all_outputs if args.demo else all_outputs[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
