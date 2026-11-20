"""Пайплайн збору, очищення, чанкування та аналізу корпусу (Лабораторна робота № 2).

Запуск із кореня репозиторію:
    python -m src.ingest.pipeline              # завантаження з Hugging Face Hub + повний пайплайн
    python -m src.ingest.pipeline --from-cache # повторний швидкий запуск із локального кешу data/raw/
    python -m src.ingest.pipeline --inspect    # виведення статистики, 10 випадкових документів і прикладів чанків
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import statistics
from typing import Any

import yaml

from src.ingest.chunk import chunk_document
from src.ingest.clean import clean_text, is_cs_article, is_valid_document

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"
RAW_CACHE_PATH = ROOT_DIR / "data" / "raw" / "cs_simplewiki_raw.jsonl"
STATS_PATH = ROOT_DIR / "data" / "processed" / "corpus_stats.json"
SAMPLE_10_PATH = ROOT_DIR / "data" / "processed" / "sample_10_docs.json"
DATA_CARD_PATH = ROOT_DIR / "reports" / "lab-2" / "data-card.md"

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#-]{1,}")

# Стандартні англійські службові слова для виділення змістовного топ-20 словника
STOPWORDS: set[str] = {
    "the", "of", "and", "to", "in", "is", "it", "that", "for", "as", "are",
    "on", "with", "be", "by", "this", "an", "or", "from", "at", "was", "not",
    "can", "they", "which", "have", "has", "also", "used", "one", "more",
    "such", "their", "its", "these", "other", "into", "but", "when", "were",
    "all", "some", "may", "many", "there", "been", "than", "then", "about",
    "most", "only", "like", "use", "using", "called", "made", "make", "between",
    "each", "different", "often", "two", "first", "will", "would", "could",
    "should", "them", "they", "he", "she", "his", "her", "we", "you", "your",
    "who", "what", "where", "how", "why", "while", "after", "before", "over",
    "under", "through", "because", "if", "so", "no", "up", "out", "do", "does",
    "did", "had", "very", "new", "same", "example", "well", "known", "part",
    "time", "way", "see", "both", "any", "into", "being", "much", "even",
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fetch_raw_subset(cfg: dict[str, Any], from_cache: bool = False) -> tuple[list[dict[str, Any]], int]:
    """Завантажує тематичну підмножину CS & IT із Simple English Wikipedia."""
    ds_cfg = cfg["dataset"]
    max_docs = int(ds_cfg["target_documents_max"])

    if from_cache and RAW_CACHE_PATH.exists():
        raw_docs: list[dict[str, Any]] = []
        with RAW_CACHE_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    raw_docs.append(json.loads(line))
        total_snapshot = 241787
        if STATS_PATH.exists():
            try:
                total_snapshot = json.loads(STATS_PATH.read_text(encoding="utf-8")).get(
                    "snapshot_total_docs", total_snapshot
                )
            except Exception:
                pass
        print(f"[INFO] Завантажено {len(raw_docs)} документів із локального кешу {RAW_CACHE_PATH}")
        return raw_docs, total_snapshot

    from datasets import load_dataset

    source = ds_cfg["source"]
    config_name = ds_cfg["config"]
    print(f"[INFO] Завантаження датасету {source} ({config_name}) з Hugging Face Hub...")
    dataset = load_dataset(source, config_name, split="train")
    total_snapshot = len(dataset)
    print(f"[INFO] Повний знімок містить {total_snapshot} статей. Виконується тематичний відбір CS & IT...")

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in dataset:
        doc_id = str(row["id"])
        if doc_id in seen_ids:
            continue
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        if is_cs_article(title, text):
            seen_ids.add(doc_id)
            selected.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "url": str(row.get("url") or ""),
                    "text": text,
                }
            )
            if len(selected) >= max_docs:
                break

    RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_CACHE_PATH.open("w", encoding="utf-8") as fh:
        for doc in selected:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[INFO] Відібрано {len(selected)} сирих тематичних статей (збережено в {RAW_CACHE_PATH})")
    return selected, total_snapshot


def compute_top_words(chunks: list[dict[str, Any]], top_n: int = 20) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Повертає топ-N слів (загальний та без англійських стоп-слів)."""
    all_counter: Counter[str] = Counter()
    content_counter: Counter[str] = Counter()

    for ch in chunks:
        for token in WORD_RE.findall(ch["text"].lower()):
            all_counter[token] += 1
            if token not in STOPWORDS and len(token) >= 3:
                content_counter[token] += 1

    return all_counter.most_common(top_n), content_counter.most_common(top_n)


def write_data_card(cfg: dict[str, Any], stats: dict[str, Any]) -> None:
    """Формує стислу data card у файлі reports/lab-2/data-card.md."""
    ds_cfg = cfg["dataset"]
    ch_cfg = cfg["chunking"]

    top_rows = "\n".join(
        f"| {i} | `{word}` | {count} |"
        for i, (word, count) in enumerate(stats["top_20_content_words"], start=1)
    )

    content = f"""# Data Card: CS-SimpleWiki Corpus (`data/processed/chunks.jsonl`)

## 1. Джерело та ліцензія

- **Назва корпусу:** {ds_cfg['name']}
- **Джерело:** Hugging Face Hub – датасет [`{ds_cfg['source']}`](https://huggingface.co/datasets/{ds_cfg['source']}), конфігурація `{ds_cfg['config']}`, вибірка `train`
- **Мова:** англійська (Simple English)
- **Ліцензія:** {ds_cfg['license']} (відкрита ліцензія, що дозволяє вільне копіювання, обробку, створення похідних індексів та повторне розповсюдження із зазначенням авторства)
- **Спосіб формування:** автоматичний тематичний відбір статей предметної області комп'ютерних наук та інформаційних технологій (Computer Science & IT) за ключовими термінами у заголовку та вступному абзаці

## 2. Розмір корпусу до та після очищення

| Показник | Значення |
|---|---|
| Загальна кількість статей у знімку `20231101.simple` | {stats['snapshot_total_docs']} |
| Відібрано тематичних статей CS & IT (до очищення) | {stats['docs_before_cleaning']} |
| Видалено під час фільтрації (коротші за {stats['min_char_length']} символів або порожні) | {stats['docs_removed_short']} |
| Кількість документів після очищення | {stats['docs_after_cleaning']} |
| Сумарна кількість символів до очищення | {stats['total_chars_before']} |
| Сумарна кількість символів після очищення | {stats['total_chars_after']} |
| Середня довжина очищеного документа (символів / слів) | {stats['avg_doc_chars']} / {stats['avg_doc_words']} |

Пайплайн очищення (`src/ingest/clean.py`) виконує нормалізацію кодування у UTF-8 (Unicode NFC), видалення HTML-розмітки (`BeautifulSoup`) та HTML-сутностей, усунення керуючих символів і зайвих пробілів, а також відсікання документів коротших за {stats['min_char_length']} символів.

## 3. Кількість чанків та середній розмір чанка

- **Стратегія чанкування:** фіксоване вікно за словами (`chunk_size_words = {ch_cfg['chunk_size_words']}`, перекриття `{int(ch_cfg['overlap_ratio'] * 100)}%` – {stats['overlap_words']} слів, крок вікна {stats['step_words']} слів)
- **Файл корпусу:** `{ch_cfg['output_path']}` (формат JSONL, UTF-8, розмір на диску: **{stats['jsonl_size_mb']} МБ**)
- **Загальна кількість чанків:** **{stats['total_chunks']}**
- **Середня кількість чанків на документ:** {stats['avg_chunks_per_doc']}
- **Середній розмір чанка:** **{stats['avg_chunk_words']} слів** ({stats['avg_chunk_chars']} символів)
- **Медіана / мінімум / максимум розміру чанка (слів):** {stats['median_chunk_words']} / {stats['min_chunk_words']} / {stats['max_chunk_words']}
- **Метадані кожного чанка:** `chunk_id`, `doc_id`, `chunk_index`, `title`, `url`, `word_count`, `char_count`, `text`

## 4. Найчастотніші слова корпусу (топ-20 змістовних слів)

Нижче наведено 20 найчастотніших змістовних слів корпусу після виключення базових англійських службових слів (стоп-слів на кшталт `the`, `of`, `and`, `in`, `is`, `to`):

| № | Слово | Частота у чанках |
|---|---|---|
{top_rows}

Для довідки: без фільтрації стоп-слів перші позиції традиційно посідають службові слова (`the` – {stats['top_20_all_words'][0][1]}, `of` – {stats['top_20_all_words'][1][1]}, `to` – {stats['top_20_all_words'][2][1]}, `and` – {stats['top_20_all_words'][3][1]}, `is` – {stats['top_20_all_words'][4][1]}). Висока частотність термінів `computer`, `software`, `windows`, `system`, `data`, `internet`, `program`, `operating`, `programming`, `linux`, `code`, `network` підтверджує тематичну релевантність відібраної підмножини.

## 5. Відомі обмеження корпусу

- **Часове обмеження (зріз 01.11.2023):** корпус не містить подій, версій програмного забезпечення та стандартів, що з'явилися після листопада 2023 року.
- **Спрощений енциклопедичний стиль:** статті Simple English Wikipedia орієнтовані на базове пояснення понять і не містять глибоких формальних доведень чи детальної специфікації API.
- **Залишкові службові секції Wikipedia:** у текстах збереглися заголовки секцій `References` (734 входження, 14-те місце у топ-20), `Related pages` та `Other websites`, що частково додає шум у кінцеві чанки статей.
- **Втрата математичного форматування у вихідному дампі:** під час ручної перевірки виявлено сплющення верхніх індексів у plain-text дампі Wikimedia (наприклад, `2^50` та `10^15` у статті `Pebibyte` записані як `250` та `1015`).
- **Поодинокі сторінки неоднозначностей без суфікса `(disambiguation)`:** ручна перевірка виявила статтю `Grid` (`"Grid may mean:..."`), що потрапила через згадку терміна `graphical user interface`.
- **Нерівномірна довжина статей та мовне обмеження:** частина коротких статей формує один чанк на 16–60 слів; усі тексти англомовні, тому запити до RAG-системи формулюються англійською.
"""
    DATA_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_CARD_PATH.write_text(content, encoding="utf-8")
    print(f"[INFO] Data card збережено у {DATA_CARD_PATH}")


def run_pipeline(from_cache: bool = False) -> dict[str, Any]:
    cfg = load_config()
    min_chars = int(cfg["dataset"]["min_char_length"])
    chunk_size = int(cfg["chunking"]["chunk_size_words"])
    overlap_ratio = float(cfg["chunking"]["overlap_ratio"])
    output_path = ROOT_DIR / cfg["chunking"]["output_path"]

    raw_docs, snapshot_total = fetch_raw_subset(cfg, from_cache=from_cache)

    total_chars_before = sum(len(d["text"]) for d in raw_docs)
    cleaned_docs: list[dict[str, Any]] = []
    removed_short = 0

    for doc in raw_docs:
        cleaned = clean_text(doc["text"])
        if not is_valid_document(cleaned, min_char_length=min_chars):
            removed_short += 1
            continue
        cleaned_docs.append(
            {
                "doc_id": doc["doc_id"],
                "title": clean_text(doc["title"]),
                "url": doc["url"],
                "text": cleaned,
            }
        )

    total_chars_after = sum(len(d["text"]) for d in cleaned_docs)
    doc_word_counts = [len(d["text"].split()) for d in cleaned_docs]

    all_chunks: list[dict[str, Any]] = []
    for doc in cleaned_docs:
        all_chunks.extend(
            chunk_document(
                doc,
                chunk_size_words=chunk_size,
                overlap_ratio=overlap_ratio,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for ch in all_chunks:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")

    jsonl_size_bytes = output_path.stat().st_size
    chunk_words = [c["word_count"] for c in all_chunks]
    chunk_chars = [c["char_count"] for c in all_chunks]
    top_all, top_content = compute_top_words(all_chunks, top_n=20)

    # Відбір 10 випадкових очищених документів для ручної перевірки (фіксований seed=42)
    rng = random.Random(42)
    sample_10 = rng.sample(cleaned_docs, k=min(10, len(cleaned_docs)))
    sample_records = [
        {
            "doc_id": d["doc_id"],
            "title": d["title"],
            "url": d["url"],
            "char_count": len(d["text"]),
            "word_count": len(d["text"].split()),
            "preview": d["text"][:320].replace("\n", " "),
        }
        for d in sample_10
    ]
    SAMPLE_10_PATH.write_text(
        json.dumps(sample_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    overlap_words = max(1, int(round(chunk_size * overlap_ratio)))
    stats = {
        "snapshot_total_docs": snapshot_total,
        "docs_before_cleaning": len(raw_docs),
        "docs_removed_short": removed_short,
        "docs_after_cleaning": len(cleaned_docs),
        "min_char_length": min_chars,
        "total_chars_before": total_chars_before,
        "total_chars_after": total_chars_after,
        "avg_doc_chars": round(total_chars_after / len(cleaned_docs), 1) if cleaned_docs else 0.0,
        "avg_doc_words": round(sum(doc_word_counts) / len(doc_word_counts), 1) if doc_word_counts else 0.0,
        "total_chunks": len(all_chunks),
        "avg_chunks_per_doc": round(len(all_chunks) / len(cleaned_docs), 2) if cleaned_docs else 0.0,
        "chunk_size_words": chunk_size,
        "overlap_words": overlap_words,
        "step_words": max(1, chunk_size - overlap_words),
        "avg_chunk_words": round(sum(chunk_words) / len(chunk_words), 1) if chunk_words else 0.0,
        "avg_chunk_chars": round(sum(chunk_chars) / len(chunk_chars), 1) if chunk_chars else 0.0,
        "median_chunk_words": int(statistics.median(chunk_words)) if chunk_words else 0,
        "min_chunk_words": min(chunk_words) if chunk_words else 0,
        "max_chunk_words": max(chunk_words) if chunk_words else 0,
        "jsonl_size_bytes": jsonl_size_bytes,
        "jsonl_size_mb": round(jsonl_size_bytes / (1024 * 1024), 2),
        "top_20_all_words": top_all,
        "top_20_content_words": top_content,
    }

    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_data_card(cfg, stats)

    print(f"[OK] Збережено {stats['total_chunks']} чанків у {output_path} ({stats['jsonl_size_mb']} МБ)")
    print(f"[OK] Документів: до очищення={stats['docs_before_cleaning']}, відфільтровано={stats['docs_removed_short']}, після очищення={stats['docs_after_cleaning']}")
    print(f"[OK] Середній розмір чанка: {stats['avg_chunk_words']} слів ({stats['avg_chunk_chars']} символів)")
    return stats


def inspect_outputs() -> None:
    """Виводить статистику, 10 випадкових очищених документів та приклади чанків."""
    if not STATS_PATH.exists():
        print("[ERROR] Спочатку запустіть пайплайн: python -m src.ingest.pipeline")
        return

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    sample_10 = json.loads(SAMPLE_10_PATH.read_text(encoding="utf-8"))

    print("=== СТАТИСТИКА КОРПУСУ (ЛБ-2) ===")
    print(f"Знімок Simple Wikipedia:    {stats['snapshot_total_docs']} статей")
    print(f"Відібрано до очищення:      {stats['docs_before_cleaning']} документів")
    print(f"Відфільтровано (<100 симв): {stats['docs_removed_short']} документів")
    print(f"Залишилось після очищення:  {stats['docs_after_cleaning']} документів")
    print(f"Усього чанків (JSONL):      {stats['total_chunks']} ({stats['jsonl_size_mb']} МБ)")
    print(f"Середній розмір чанка:      {stats['avg_chunk_words']} слів / {stats['avg_chunk_chars']} симв.")
    print(f"Медіана / мін / макс слів:  {stats['median_chunk_words']} / {stats['min_chunk_words']} / {stats['max_chunk_words']}")

    print("\n=== ТОП-20 ЗМІСТОВНИХ СЛІВ ===")
    for i, (w, c) in enumerate(stats["top_20_content_words"], start=1):
        print(f"  {i:2d}. {w:<18s} {c}")

    print("\n=== 10 ВИПАДКОВИХ ОЧИЩЕНИХ ДОКУМЕНТІВ (seed=42) ===")
    for i, d in enumerate(sample_10, start=1):
        print(f"  {i:2d}. [doc_id={d['doc_id']}] {d['title']} ({d['word_count']} слів, {d['char_count']} симв.)")
        print(f"      {d['preview'][:160]}...")

    chunks_path = ROOT_DIR / load_config()["chunking"]["output_path"]
    if chunks_path.exists():
        print("\n=== ПЕРШІ 3 ЧАНКИ З data/processed/chunks.jsonl ===")
        with chunks_path.open("r", encoding="utf-8") as fh:
            for i in range(3):
                line = fh.readline()
                if not line:
                    break
                ch = json.loads(line)
                print(
                    f"  {i+1}. chunk_id={ch['chunk_id']} | doc_id={ch['doc_id']} | "
                    f"idx={ch['chunk_index']} | words={ch['word_count']} | title={ch['title']}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Пайплайн збору та підготовки даних (ЛБ-2)")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Використати локальний кеш data/raw/cs_simplewiki_raw.jsonl замість завантаження з HF Hub",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Вивести зведену статистику, 10 випадкових документів та приклади чанків",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_outputs()
    else:
        run_pipeline(from_cache=args.from_cache)


if __name__ == "__main__":
    main()
