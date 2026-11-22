"""Оцінювання якості векторного пошуку (Recall@5, Recall@10) та підбір параметрів індексу (ЛБ-3).

Запуск із кореня репозиторію:
    python -m src.eval.eval_retrieval                 # повний експеримент із 6 конфігураціями
    python -m src.eval.eval_retrieval --current-only  # швидка оцінка поточного індексу data/chroma_db
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.embeddings.build_index import (
    ROOT_DIR,
    build_hnsw_metadata,
    format_chunk_for_embedding,
    load_chunks,
    load_config,
    resolve_chroma_path,
)

EVAL_SET_PATH = ROOT_DIR / "tests" / "eval-set.jsonl"
EVAL_RESULTS_PATH = ROOT_DIR / "data" / "processed" / "lab3_eval_results.json"

# Сітка експериментальних конфігурацій для підбору параметрів пошуку (Завдання 5)
TUNING_CONFIGS: list[dict[str, Any]] = [
    {
        "id": "C1_raw_l2_low_hnsw",
        "label": "Raw text + L2 (M=4, ef_s=5)",
        "prepend_title": False,
        "normalize_embeddings": False,
        "metric": "l2",
        "hnsw_m": 4,
        "hnsw_construction_ef": 16,
        "hnsw_search_ef": 5,
    },
    {
        "id": "C2_raw_cosine_light",
        "label": "Raw text + Cosine (M=8, ef_s=10)",
        "prepend_title": False,
        "normalize_embeddings": True,
        "metric": "cosine",
        "hnsw_m": 8,
        "hnsw_construction_ef": 32,
        "hnsw_search_ef": 10,
    },
    {
        "id": "C3_raw_cosine_std",
        "label": "Raw text + Cosine (M=16, ef_s=50)",
        "prepend_title": False,
        "normalize_embeddings": True,
        "metric": "cosine",
        "hnsw_m": 16,
        "hnsw_construction_ef": 100,
        "hnsw_search_ef": 50,
    },
    {
        "id": "C4_title_l2_std",
        "label": "Title+text + L2 (M=16, ef_s=50)",
        "prepend_title": True,
        "normalize_embeddings": False,
        "metric": "l2",
        "hnsw_m": 16,
        "hnsw_construction_ef": 100,
        "hnsw_search_ef": 50,
    },
    {
        "id": "C5_title_ip_std",
        "label": "Title+text + IP (M=16, ef_s=50)",
        "prepend_title": True,
        "normalize_embeddings": True,
        "metric": "ip",
        "hnsw_m": 16,
        "hnsw_construction_ef": 100,
        "hnsw_search_ef": 50,
    },
    {
        "id": "C6_title_cosine_tuned",
        "label": "Title+text + Cosine (M=32, ef_s=100)",
        "prepend_title": True,
        "normalize_embeddings": True,
        "metric": "cosine",
        "hnsw_m": 32,
        "hnsw_construction_ef": 200,
        "hnsw_search_ef": 100,
    },
]


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict[str, Any]]:
    """Зчитує оцінювальний набір пар «запит – релевантний чанк» із JSONL."""
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(json.loads(line))
    return items


def compute_recall_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    """Повертає 1.0, якщо relevant_id входить у перші k результатів, інакше 0.0."""
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def evaluate_collection(
    collection: Any,
    eval_items: list[dict[str, Any]],
    query_embeddings: list[list[float]],
    top_k_max: int = 10,
) -> dict[str, Any]:
    """Обчислює Recall@5, Recall@10 (за чанком і за документом), MRR@10 та затримку."""
    hits_chunk_5 = 0
    hits_chunk_10 = 0
    hits_doc_5 = 0
    hits_doc_10 = 0
    mrr_sum = 0.0
    latencies_ms: list[float] = []
    per_query: list[dict[str, Any]] = []

    for item, q_emb in zip(eval_items, query_embeddings):
        t0 = time.perf_counter()
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k_max,
            include=["metadatas", "distances"],
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)

        retrieved_chunk_ids = res["ids"][0]
        retrieved_doc_ids = [str(m.get("doc_id", "")) for m in res["metadatas"][0]]

        target_chunk = item["relevant_chunk_id"]
        target_doc = str(item["relevant_doc_id"])

        rank_chunk = (
            retrieved_chunk_ids.index(target_chunk) + 1
            if target_chunk in retrieved_chunk_ids
            else None
        )
        rank_doc = (
            retrieved_doc_ids.index(target_doc) + 1
            if target_doc in retrieved_doc_ids
            else None
        )

        hit_c5 = compute_recall_at_k(retrieved_chunk_ids, target_chunk, 5)
        hit_c10 = compute_recall_at_k(retrieved_chunk_ids, target_chunk, 10)
        hit_d5 = compute_recall_at_k(retrieved_doc_ids, target_doc, 5)
        hit_d10 = compute_recall_at_k(retrieved_doc_ids, target_doc, 10)

        hits_chunk_5 += int(hit_c5)
        hits_chunk_10 += int(hit_c10)
        hits_doc_5 += int(hit_d5)
        hits_doc_10 += int(hit_d10)
        if rank_chunk is not None:
            mrr_sum += 1.0 / rank_chunk

        per_query.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "target_chunk_id": target_chunk,
                "target_title": item["title"],
                "chunk_position": item.get("chunk_position", ""),
                "rank_chunk": rank_chunk,
                "rank_doc": rank_doc,
                "top1_chunk_id": retrieved_chunk_ids[0] if retrieved_chunk_ids else None,
                "top1_title": res["metadatas"][0][0].get("title") if res["metadatas"][0] else None,
                "hit_at_5": bool(hit_c5),
                "hit_at_10": bool(hit_c10),
                "latency_ms": round(lat_ms, 2),
            }
        )

    n = max(1, len(eval_items))
    return {
        "num_queries": len(eval_items),
        "hits_at_5": hits_chunk_5,
        "hits_at_10": hits_chunk_10,
        "recall_at_5": round(hits_chunk_5 / n, 4),
        "recall_at_10": round(hits_chunk_10 / n, 4),
        "doc_recall_at_5": round(hits_doc_5 / n, 4),
        "doc_recall_at_10": round(hits_doc_10 / n, 4),
        "mrr_at_10": round(mrr_sum / n, 4),
        "avg_query_ms": round(sum(latencies_ms) / n, 2),
        "per_query": per_query,
    }


def run_current_index_eval() -> dict[str, Any]:
    """Оцінює вже побудований persistent-індекс у data/chroma_db."""
    cfg = load_config()
    emb_cfg = cfg["embeddings"]
    vs_cfg = cfg["vector_store"]

    eval_items = load_eval_set()
    model = SentenceTransformer(
        emb_cfg["model_name"], device=emb_cfg.get("device", "cpu")
    )
    use_norm = bool(emb_cfg.get("normalize_embeddings", True))
    q_embs = model.encode(
        [item["query"] for item in eval_items],
        normalize_embeddings=use_norm,
        convert_to_numpy=True,
    ).tolist()

    client = chromadb.PersistentClient(
        path=resolve_chroma_path(ROOT_DIR / vs_cfg["persist_directory"])
    )
    collection = client.get_collection(name=vs_cfg["collection_name"])
    metrics = evaluate_collection(collection, eval_items, q_embs, top_k_max=10)

    print("=== ОЦІНКА ПОТОЧНОГО ІНДЕКСУ CHROMADB ===")
    print(f"Колекція:     {vs_cfg['collection_name']} ({collection.count()} векторів)")
    print(f"Запитів:      {metrics['num_queries']}")
    print(f"Recall@5:     {metrics['recall_at_5']*100:.1f}% ({metrics['hits_at_5']}/{metrics['num_queries']})")
    print(f"Recall@10:    {metrics['recall_at_10']*100:.1f}% ({metrics['hits_at_10']}/{metrics['num_queries']})")
    print(f"Doc Recall@5: {metrics['doc_recall_at_5']*100:.1f}% | Doc Recall@10: {metrics['doc_recall_at_10']*100:.1f}%")
    print(f"MRR@10:       {metrics['mrr_at_10']:.4f}")
    print(f"Сер. запит:   {metrics['avg_query_ms']:.2f} мс")
    print("\nДеталізація за запитами:")
    for pq in metrics["per_query"]:
        rank_str = f"#{pq['rank_chunk']}" if pq["rank_chunk"] is not None else ">10"
        status = "OK" if pq["hit_at_5"] else ("TOP10" if pq["hit_at_10"] else "MISS")
        print(
            f"  {pq['query_id']} [{status:5s}] rank={rank_str:3s} | "
            f"target={pq['target_chunk_id']} ({pq['target_title']})"
        )
    return metrics


def run_tuning_experiment() -> dict[str, Any]:
    """Порівнює 6 конфігурацій параметрів векторного пошуку на tests/eval-set.jsonl."""
    cfg = load_config()
    emb_cfg = cfg["embeddings"]
    chunks = load_chunks(ROOT_DIR / cfg["chunking"]["output_path"])
    eval_items = load_eval_set()

    model_name = emb_cfg["model_name"]
    device = emb_cfg.get("device", "cpu")
    print(f"[INFO] Завантаження моделі {model_name} ({device})...")
    model = SentenceTransformer(model_name, device=device)

    # Кешуємо вектори для двох варіантів тексту (raw text та title + text),
    # щоб не кодувати корпус повторно без потреби
    import numpy as np

    chunk_emb_cache: dict[tuple[bool, bool], list[list[float]]] = {}
    query_emb_cache: dict[bool, list[list[float]]] = {}

    queries = [item["query"] for item in eval_items]
    q_raw = model.encode(queries, normalize_embeddings=False, convert_to_numpy=True)
    q_norm = q_raw / np.clip(np.linalg.norm(q_raw, axis=1, keepdims=True), 1e-12, None)
    query_emb_cache[False] = q_raw.tolist()
    query_emb_cache[True] = q_norm.tolist()

    for use_title in (False, True):
        texts = [format_chunk_for_embedding(ch, prepend_title=use_title) for ch in chunks]
        t0 = time.perf_counter()
        raw_embs = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        norm_embs = raw_embs / np.clip(
            np.linalg.norm(raw_embs, axis=1, keepdims=True), 1e-12, None
        )
        dt = time.perf_counter() - t0
        chunk_emb_cache[(use_title, False)] = raw_embs.tolist()
        chunk_emb_cache[(use_title, True)] = norm_embs.tolist()
        print(
            f"[INFO] Закодовано корпус (prepend_title={use_title}) за {dt:.2f} с",
            flush=True,
        )

    ids = [ch["chunk_id"] for ch in chunks]
    docs = [ch["text"] for ch in chunks]
    metas = [
        {
            "doc_id": str(ch["doc_id"]),
            "chunk_index": int(ch["chunk_index"]),
            "title": str(ch["title"]),
            "url": str(ch["url"]),
            "word_count": int(ch["word_count"]),
            "char_count": int(ch["char_count"]),
        }
        for ch in chunks
    ]

    experiment_runs: list[dict[str, Any]] = []

    client = chromadb.EphemeralClient()

    for exp in TUNING_CONFIGS:
        existing_names = [c.name for c in client.list_collections()]
        if exp["id"] in existing_names:
            client.delete_collection(name=exp["id"])
        hnsw_meta = build_hnsw_metadata(
            distance_metric=exp["metric"],
            hnsw_m=exp["hnsw_m"],
            hnsw_construction_ef=exp["hnsw_construction_ef"],
            hnsw_search_ef=exp["hnsw_search_ef"],
        )
        col = client.create_collection(name=exp["id"], metadata=hnsw_meta)
        ch_embs = chunk_emb_cache[(exp["prepend_title"], exp["normalize_embeddings"])]

        t_idx = time.perf_counter()
        for start in range(0, len(chunks), 250):
            end = min(start + 250, len(chunks))
            col.add(
                ids=ids[start:end],
                embeddings=ch_embs[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
            )
        idx_sec = round(time.perf_counter() - t_idx, 2)

        q_embs = query_emb_cache[exp["normalize_embeddings"]]
        res = evaluate_collection(col, eval_items, q_embs, top_k_max=10)
        run_record = {
            "config": exp,
            "index_build_seconds": idx_sec,
            **res,
        }
        experiment_runs.append(run_record)
        print(
            f"  [{exp['id']}] R@5={res['recall_at_5']*100:5.1f}% "
            f"({res['hits_at_5']}/20) | R@10={res['recall_at_10']*100:5.1f}% "
            f"({res['hits_at_10']}/20) | MRR@10={res['mrr_at_10']:.4f} | "
            f"lat={res['avg_query_ms']:.2f} мс"
        )

    best_run = max(
        experiment_runs,
        key=lambda r: (
            r["recall_at_5"],
            r["recall_at_10"],
            r["mrr_at_10"],
            1 if r["config"]["metric"] == "cosine" else 0,
            r["config"]["hnsw_search_ef"],
        ),
    )
    summary = {
        "model_name": model_name,
        "eval_set_size": len(eval_items),
        "corpus_chunks": len(chunks),
        "best_config_id": best_run["config"]["id"],
        "experiments": experiment_runs,
    }

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] Найкраща конфігурація: {best_run['config']['id']} ({best_run['config']['label']})")
    print(
        f"[OK] Recall@5={best_run['recall_at_5']*100:.1f}%, "
        f"Recall@10={best_run['recall_at_10']*100:.1f}%, "
        f"MRR@10={best_run['mrr_at_10']:.4f}"
    )
    print(f"[OK] Результати збережено у {EVAL_RESULTS_PATH}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оцінювання Recall@5 / Recall@10 та підбір параметрів індексу (ЛБ-3)"
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Оцінити лише поточний persistent-індекс у data/chroma_db",
    )
    args = parser.parse_args()

    if args.current_only:
        run_current_index_eval()
    else:
        run_tuning_experiment()


if __name__ == "__main__":
    main()
