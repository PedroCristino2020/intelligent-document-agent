import argparse
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from src.retriever import retrieve


QUESTIONS_PATH = Path("eval") / "questions.json"
DATABASE_PATH = Path("data") / "index.db"
TOP_K_DEFAULT = 3


def load_index(database_path: Path) -> tuple[list[dict], list[list[float]]]:
    """Load indexed chunks and their embeddings from SQLite."""
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT source, page, chunk_index, text, embedding
            FROM indexed_chunks
            """
        ).fetchall()

    chunks = []
    embeddings = []

    for source, page, chunk_index, text, embedding_json in rows:
        chunks.append({
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "text": text,
        })
        embeddings.append(json.loads(embedding_json))

    return chunks, embeddings


def _normalize_key(key) -> tuple:
    """Normalize key (tuple or list) to tuple."""
    if isinstance(key, (list, tuple)):
        # Ensure we have 3 elements
        if len(key) == 2:
            return (key[0], key[1], None)
        elif len(key) == 3:
            return (key[0], key[1], key[2])
    return tuple(key)


def get_relevant_keys(question: dict) -> set[tuple]:
    """Get set of relevant keys (source, page, chunk_index) when available."""
    relevant_keys = set()
    for item in question.get("relevant_chunks", []):
        source = item["source"]
        page = item["page"]
        if "chunk_index" in item and item["chunk_index"] is not None:
            relevant_keys.add((source, page, int(item["chunk_index"]) if isinstance(item["chunk_index"], str) is False else item["chunk_index"]))
        else:
            relevant_keys.add((source, page, None))
    return relevant_keys


def has_chunk_level(relevant_keys: set[tuple]) -> bool:
    """Check if any relevant key uses chunk-level granularity."""
    for k in relevant_keys:
        nk = _normalize_key(k)
        if nk[2] is not None:
            return True
    return False


def is_relevant_result(result: dict, relevant_keys: set[tuple]) -> bool:
    """Check if a result is relevant (chunk-level if available, else page-level)."""
    source = result["source"]
    page = result["page"]
    chunk_index = result.get("chunk_index")

    # Normalize relevant keys
    norm_relevant = {_normalize_key(k) for k in relevant_keys}

    # Try chunk-level match first
    if (source, page, chunk_index) in norm_relevant:
        return True

    # Fall back to page-level match
    if (source, page, None) in norm_relevant:
        return True

    return False


def calculate_precision_at_k(results: list[dict], relevant_keys: set[tuple], k: int) -> float:
    """Calculate Precision@k."""
    if k <= 0:
        return 0.0
    relevant_count = 0
    for i in range(min(k, len(results))):
        if is_relevant_result(results[i], relevant_keys):
            relevant_count += 1
    return relevant_count / k


def calculate_hit_at_k(results: list[dict], relevant_keys: set[tuple], k: int) -> float:
    """Calculate Hit@k (1 if at least one relevant in top-k)."""
    for i in range(min(k, len(results))):
        if is_relevant_result(results[i], relevant_keys):
            return 1.0
    return 0.0


def calculate_recall_at_k(results: list[dict], relevant_keys: set[tuple], k: int) -> float:
    """Calculate Recall@k at appropriate level (chunk-level if any chunk_index specified)."""
    if not relevant_keys:
        return 0.0

    norm_relevant = {_normalize_key(k) for k in relevant_keys}
    chunk_level = any(k[2] is not None for k in norm_relevant)

    if chunk_level:
        relevant_found = set()
        for i in range(min(k, len(results))):
            result = results[i]
            key_res = (result["source"], result["page"], result.get("chunk_index"))
            if key_res in norm_relevant:
                relevant_found.add(key_res)
        total_relevant = len(norm_relevant)
        if total_relevant == 0:
            return 0.0
        return len(relevant_found) / total_relevant
    else:
        target_pages = {(k[0], k[1]) for k in norm_relevant}
        relevant_found_pages = set()
        for i in range(min(k, len(results))):
            result = results[i]
            rp = (result["source"], result["page"])
            if rp in target_pages:
                relevant_found_pages.add(rp)
        total_relevant_pages = len(target_pages)
        if total_relevant_pages == 0:
            return 0.0
        return len(relevant_found_pages) / total_relevant_pages


def calculate_mrr(results: list[dict], relevant_keys: set[tuple]) -> float:
    """Calculate Reciprocal Rank."""
    for rank, result in enumerate(results, start=1):
        if is_relevant_result(result, relevant_keys):
            return 1.0 / rank
    return 0.0


def calculate_map_at_k(results: list[dict], relevant_keys: set[tuple], k: int) -> float:
    """Calculate Mean Average Precision@k."""
    if not relevant_keys:
        return 0.0

    norm_relevant = {_normalize_key(k) for k in relevant_keys}
    chunk_level = any(k[2] is not None for k in norm_relevant)

    if chunk_level:
        target_set = set(norm_relevant)
        total = len(target_set)
    else:
        target_set = {(k[0], k[1]) for k in norm_relevant}
        total = len(target_set)

    relevant_count = 0
    precision_sum = 0.0
    seen_relevant = set()

    for i in range(min(k, len(results))):
        result = results[i]
        if chunk_level:
            key_res = (result["source"], result["page"], result.get("chunk_index"))
        else:
            key_res = (result["source"], result["page"])

        if is_relevant_result(result, relevant_keys) and key_res not in seen_relevant:
            if key_res in target_set:
                relevant_count += 1
                seen_relevant.add(key_res)
                precision_sum += relevant_count / (i + 1)

    if relevant_count == 0 or total == 0:
        return 0.0
    return precision_sum / total


def calculate_ndcg_at_k(results: list[dict], relevant_keys: set[tuple], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain@k (binary relevance)."""
    if not relevant_keys:
        return 0.0

    norm_relevant = {_normalize_key(k) for k in relevant_keys}
    chunk_level = any(k[2] is not None for k in norm_relevant)

    dcg = 0.0
    for i in range(min(k, len(results))):
        result = results[i]
        if chunk_level:
            key_res = (result["source"], result["page"], result.get("chunk_index"))
            is_rel = key_res in norm_relevant or is_relevant_result(result, relevant_keys)
        else:
            key_res = (result["source"], result["page"])
            is_rel = key_res in {(k[0], k[1]) for k in norm_relevant} or is_relevant_result(result, relevant_keys)

        if is_rel:
            dcg += 1.0 / ((i + 1) + 1 - 1)  # log2(rank+1)

    if chunk_level:
        num_relevant = len(norm_relevant)
    else:
        num_relevant = len({(k[0], k[1]) for k in norm_relevant})

    idcg = 0.0
    for i in range(min(num_relevant, k)):
        idcg += 1.0 / ((i + 1) + 1 - 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_question(
    question: dict,
    chunks: list[dict],
    embeddings: list[list[float]],
    top_k: int = TOP_K_DEFAULT,
) -> dict:
    """Calculate all retrieval metrics for a question."""
    results = retrieve(
        question["question"],
        chunks,
        embeddings,
        top_k=top_k,
    )

    relevant_keys = get_relevant_keys(question)
    norm_relevant_for_save = [list(_normalize_key(k)) for k in relevant_keys]

    recall = calculate_recall_at_k(results, relevant_keys, top_k)
    precision = calculate_precision_at_k(results, relevant_keys, top_k)
    hit = calculate_hit_at_k(results, relevant_keys, top_k)
    mrr = calculate_mrr(results, relevant_keys)
    map_at_k = calculate_map_at_k(results, relevant_keys, top_k)
    ndcg = calculate_ndcg_at_k(results, relevant_keys, top_k)

    return {
        "question": question["question"],
        "relevant_keys": norm_relevant_for_save,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "hit_at_k": hit,
        "mrr": mrr,
        "map_at_k": map_at_k,
        "ndcg_at_k": ndcg,
        "retrieved": [
            {
                "rank": i + 1,
                "source": r["source"],
                "page": r["page"],
                "chunk_index": r.get("chunk_index"),
                "similarity": r["similarity"],
                "is_relevant": is_relevant_result(r, relevant_keys),
                "text_preview": r["text"][:200],
            }
            for i, r in enumerate(results)
        ],
    }


def inspect_rag_questions(
    chunks: list[dict],
    embeddings: list[list[float]],
    questions_to_inspect: list[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> None:
    if questions_to_inspect is None:
        questions_to_inspect = [
            "¿Qué son las necesidades de información en recuperación de información?",
            "¿Qué es la recuperación densa o dense retrieval?",
            "¿Por qué se utilizan sistemas de recuperación de información junto a los modelos de lenguaje?",
        ]

    for question in questions_to_inspect:
        print("\n" + "=" * 80)
        print(f"PREGUNTA: {question}")
        print("=" * 80)

        results = retrieve(question, chunks, embeddings, top_k=top_k)

        for rank, result in enumerate(results, start=1):
            print(
                f"\nResultado {rank}: "
                f"{result['source']} | "
                f"Página {result['page']} | "
                f"Chunk {result['chunk_index']} | "
                f"Similitud: {result['similarity']:.3f}"
            )
            print(result["text"][:500])


def save_results(results_summary: dict, output_dir: Path) -> Path:
    """Save evaluation results with timestamp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"results_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval with comprehensive metrics."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help=f"Number of chunks to retrieve (default: {TOP_K_DEFAULT})",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect retrieval for sample questions without full evaluation",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Inspect a specific question",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval") / "results",
        help="Directory to save results (default: eval/results)",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    chunks, embeddings = load_index(DATABASE_PATH)

    if not chunks:
        raise ValueError("The database contains no indexed chunks.")

    # Inspection mode
    if args.inspect or args.question:
        if args.question:
            inspect_rag_questions(chunks, embeddings, [args.question], top_k=args.top_k)
        else:
            inspect_rag_questions(chunks, embeddings, top_k=args.top_k)
        return

    # Full evaluation
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        questions = json.load(file)

    per_question_results = []
    recalls = []
    precisions = []
    hits = []
    reciprocal_ranks = []
    maps = []
    ndcgs = []

    print(f"Evaluating {len(questions)} questions...")
    print(f"Retrieving top {args.top_k} chunks per question.\n")

    for index, question in enumerate(questions, start=1):
        result = evaluate_question(question, chunks, embeddings, top_k=args.top_k)
        per_question_results.append(result)

        recalls.append(result["recall_at_k"])
        precisions.append(result["precision_at_k"])
        hits.append(result["hit_at_k"])
        reciprocal_ranks.append(result["mrr"])
        maps.append(result["map_at_k"])
        ndcgs.append(result["ndcg_at_k"])

        print(
            f"{index:02d}. R@{args.top_k}: {result['recall_at_k']:.2f} | "
            f"P@{args.top_k}: {result['precision_at_k']:.2f} | "
            f"Hit@{args.top_k}: {result['hit_at_k']:.2f} | "
            f"MRR: {result['mrr']:.2f} | "
            f"MAP@{args.top_k}: {result['map_at_k']:.2f} | "
            f"nDCG@{args.top_k}: {result['ndcg_at_k']:.2f} | "
            f"{question['question'][:60]}..."
        )

    mean_recall = sum(recalls) / len(recalls)
    mean_precision = sum(precisions) / len(precisions)
    mean_hit = sum(hits) / len(hits)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    mean_map = sum(maps) / len(maps)
    mean_ndcg = sum(ndcgs) / len(ndcgs)

    print("\n" + "=" * 70)
    print(f"SUMMARY (Mean across all questions)")
    print("=" * 70)
    print(f"Recall@{args.top_k}:    {mean_recall:.3f}")
    print(f"Precision@{args.top_k}: {mean_precision:.3f}")
    print(f"Hit@{args.top_k}:       {mean_hit:.3f}")
    print(f"MRR:                  {mrr:.3f}")
    print(f"MAP@{args.top_k}:      {mean_map:.3f}")
    print(f"nDCG@{args.top_k}:     {mean_ndcg:.3f}")
    print("=" * 70)

    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": args.top_k,
            "database_path": str(DATABASE_PATH),
            "questions_path": str(QUESTIONS_PATH),
            "embedding_model": embedding_model,
            "retrieval_method": "cosine_similarity",
        },
        "summary": {
            "num_questions": len(questions),
            "recall_at_k": round(mean_recall, 6),
            "precision_at_k": round(mean_precision, 6),
            "hit_at_k": round(mean_hit, 6),
            "mrr": round(mrr, 6),
            "map_at_k": round(mean_map, 6),
            "ndcg_at_k": round(mean_ndcg, 6),
        },
        "per_question": per_question_results,
    }

    if not args.no_save:
        output_path = save_results(results_summary, args.output_dir)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nResults not saved (--no-save)")


if __name__ == "__main__":
    main()
