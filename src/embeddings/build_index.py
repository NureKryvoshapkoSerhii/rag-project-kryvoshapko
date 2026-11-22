"""Побудова векторних представлень та індексу ChromaDB над корпусом (Лабораторна робота № 3).

Запуск із кореня репозиторію:
    python -m src.embeddings.build_index
    python -m src.embeddings.build_index --no-title --metric l2
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import time
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"
INDEX_STATS_PATH = ROOT_DIR / "data" / "processed" / "index_stats.json"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Зчитує YAML-конфігурацію проєкту."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Завантажує підготовлені чанки з файлу JSONL (ЛБ-2)."""
    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def format_chunk_for_embedding(chunk: dict[str, Any], prepend_title: bool = True) -> str:
    """Формує текст чанка для кодування моделлю ембедингів.

    Додавання заголовка статті перед текстом зберігає тематичний контекст
    для внутрішніх чанків (chunk_index > 0), де назва теми може не повторюватися.
    """
    text = chunk["text"].strip()
    title = str(chunk.get("title") or "").strip()
    if prepend_title and title:
        return f"{title}\n{text}"
    return text


def directory_size_bytes(path: Path) -> int:
    """Обчислює сумарний розмір усіх файлів у директорії (у байтах)."""
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def resolve_chroma_path(path: Path) -> str:
    """Повертає безпечний шлях для ChromaDB (обхід обмеження hnswlib на кирилицю у Windows)."""
    path.mkdir(parents=True, exist_ok=True)
    path_str = str(path.resolve())
    if os.name == "nt" and not path_str.isascii():
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 1024):
            return buf.value
    return path_str


def build_hnsw_metadata(
    distance_metric: str = "cosine",
    hnsw_m: int = 32,
    hnsw_construction_ef: int = 200,
    hnsw_search_ef: int = 100,
) -> dict[str, Any]:
    """Формує словник параметрів HNSW для колекції ChromaDB."""
    return {
        "hnsw:space": distance_metric,
        "hnsw:M": int(hnsw_m),
        "hnsw:construction_ef": int(hnsw_construction_ef),
        "hnsw:search_ef": int(hnsw_search_ef),
    }


def build_vector_index(
    cfg: dict[str, Any] | None = None,
    *,
    prepend_title: bool | None = None,
    distance_metric: str | None = None,
    normalize_embeddings: bool | None = None,
    hnsw_m: int | None = None,
    hnsw_construction_ef: int | None = None,
    hnsw_search_ef: int | None = None,
    batch_size: int = 64,
    save_stats: bool = True,
) -> dict[str, Any]:
    """Кодує всі чанки корпусу та зберігає persistent-індекс у ChromaDB."""
    cfg = cfg or load_config()
    emb_cfg = cfg["embeddings"]
    vs_cfg = cfg["vector_store"]

    model_name = emb_cfg["model_name"]
    device = emb_cfg.get("device", "cpu")
    use_title = (
        bool(emb_cfg.get("prepend_title", True))
        if prepend_title is None
        else prepend_title
    )
    use_norm = (
        bool(emb_cfg.get("normalize_embeddings", True))
        if normalize_embeddings is None
        else normalize_embeddings
    )

    metric = distance_metric or vs_cfg.get("distance_metric", "cosine")
    m_val = int(hnsw_m if hnsw_m is not None else vs_cfg.get("hnsw_m", 32))
    ef_c_val = int(
        hnsw_construction_ef
        if hnsw_construction_ef is not None
        else vs_cfg.get("hnsw_construction_ef", 200)
    )
    ef_s_val = int(
        hnsw_search_ef
        if hnsw_search_ef is not None
        else vs_cfg.get("hnsw_search_ef", 100)
    )

    chunks_path = ROOT_DIR / cfg["chunking"]["output_path"]
    persist_dir = ROOT_DIR / vs_cfg["persist_directory"]
    collection_name = vs_cfg["collection_name"]

    chunks = load_chunks(chunks_path)
    print(f"[INFO] Завантажено {len(chunks)} чанків із {chunks_path}")
    print(
        f"[INFO] Модель ембедингів: {model_name} (device={device}, "
        f"prepend_title={use_title}, normalize={use_norm})"
    )

    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    load_model_sec = time.perf_counter() - t0

    texts_to_encode = [format_chunk_for_embedding(ch, prepend_title=use_title) for ch in chunks]

    t_enc = time.perf_counter()
    embeddings = model.encode(
        texts_to_encode,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=use_norm,
        convert_to_numpy=True,
    )
    encode_sec = time.perf_counter() - t_enc
    dim = int(embeddings.shape[1])
    print(
        f"[INFO] Закодовано {len(chunks)} чанків у вектори розмірності {dim} "
        f"за {encode_sec:.2f} с ({len(chunks) / max(encode_sec, 1e-6):.1f} чанків/с)"
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=resolve_chroma_path(persist_dir))

    # Повна перебудова колекції для відтворюваності індексу
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(name=collection_name)

    hnsw_meta = build_hnsw_metadata(
        distance_metric=metric,
        hnsw_m=m_val,
        hnsw_construction_ef=ef_c_val,
        hnsw_search_ef=ef_s_val,
    )
    collection = client.create_collection(name=collection_name, metadata=hnsw_meta)

    t_idx = time.perf_counter()
    upsert_batch = 250
    for start in range(0, len(chunks), upsert_batch):
        end = min(start + upsert_batch, len(chunks))
        batch_chunks = chunks[start:end]
        batch_embs = embeddings[start:end].tolist()

        collection.add(
            ids=[ch["chunk_id"] for ch in batch_chunks],
            embeddings=batch_embs,
            documents=[ch["text"] for ch in batch_chunks],
            metadatas=[
                {
                    "doc_id": str(ch["doc_id"]),
                    "chunk_index": int(ch["chunk_index"]),
                    "title": str(ch["title"]),
                    "url": str(ch["url"]),
                    "word_count": int(ch["word_count"]),
                    "char_count": int(ch["char_count"]),
                }
                for ch in batch_chunks
            ],
        )
    index_sec = time.perf_counter() - t_idx
    total_sec = load_model_sec + encode_sec + index_sec

    vector_count = collection.count()
    size_bytes = directory_size_bytes(persist_dir)
    size_mb = round(size_bytes / (1024 * 1024), 2)

    stats = {
        "model_name": model_name,
        "device": device,
        "dimension": dim,
        "vector_count": vector_count,
        "collection_name": collection_name,
        "persist_directory": vs_cfg["persist_directory"],
        "distance_metric": metric,
        "prepend_title": use_title,
        "normalize_embeddings": use_norm,
        "hnsw_m": m_val,
        "hnsw_construction_ef": ef_c_val,
        "hnsw_search_ef": ef_s_val,
        "model_load_seconds": round(load_model_sec, 2),
        "encode_seconds": round(encode_sec, 2),
        "index_seconds": round(index_sec, 2),
        "total_seconds": round(total_sec, 2),
        "chunks_per_second": round(len(chunks) / max(encode_sec, 1e-6), 1),
        "disk_size_bytes": size_bytes,
        "disk_size_mb": size_mb,
    }

    if save_stats:
        INDEX_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        INDEX_STATS_PATH.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        f"[OK] Індекс ChromaDB '{collection_name}' побудовано: "
        f"{vector_count} векторів (dim={dim}), метрика={metric}, "
        f"HNSW(M={m_val}, ef_c={ef_c_val}, ef_s={ef_s_val})"
    )
    print(f"[OK] Розмір індексу на диску ({persist_dir}): {size_mb} МБ ({size_bytes} байт)")
    return stats


def print_index_stats() -> None:
    """Виводить збережену статистику побудованого векторного індексу."""
    if not INDEX_STATS_PATH.exists():
        print("[ERROR] Спочатку побудуйте індекс: python -m src.embeddings.build_index")
        return
    stats = json.loads(INDEX_STATS_PATH.read_text(encoding="utf-8"))
    print("=== СТАТИСТИКА ВЕКТОРНОГО ІНДЕКСУ CHROMADB (ЛБ-3) ===")
    print(f"Модель ембедингів:  {stats['model_name']} ({stats['device']})")
    print(f"Колекція:           {stats['collection_name']}")
    print(f"Кількість векторів: {stats['vector_count']}")
    print(f"Розмірність:        {stats['dimension']}")
    print(f"Метрика відстані:   {stats['distance_metric']}")
    print(
        f"Заголовок у тексті: {stats['prepend_title']} | "
        f"L2-нормалізація: {stats['normalize_embeddings']}"
    )
    print(
        f"Параметри HNSW:     M={stats['hnsw_m']}, "
        f"construction_ef={stats['hnsw_construction_ef']}, search_ef={stats['hnsw_search_ef']}"
    )
    print(
        f"Час кодування:      {stats['encode_seconds']} с "
        f"({stats['chunks_per_second']} чанків/с) | Індексація: {stats['index_seconds']} с"
    )
    print(f"Розмір на диску:    {stats['disk_size_mb']} МБ ({stats['disk_size_bytes']} байт)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Векторизація корпусу та побудова індексу ChromaDB (ЛБ-3)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Вивести зведену статистику вже побудованого індексу з data/processed/index_stats.json",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Кодувати лише сирий текст чанка без додавання заголовка статті",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "l2", "ip"],
        default=None,
        help="Метрика відстані у ChromaDB (за замовчуванням із config/settings.yaml)",
    )
    parser.add_argument("--hnsw-m", type=int, default=None, help="Параметр HNSW M")
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=None,
        help="Параметр HNSW construction_ef",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=None,
        help="Параметр HNSW search_ef",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Розмір батчу кодування")
    args = parser.parse_args()

    if args.stats:
        print_index_stats()
        return

    build_vector_index(
        prepend_title=False if args.no_title else None,
        distance_metric=args.metric,
        hnsw_m=args.hnsw_m,
        hnsw_construction_ef=args.hnsw_ef_construction,
        hnsw_search_ef=args.hnsw_ef_search,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
