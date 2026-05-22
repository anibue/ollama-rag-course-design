import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from config import (
    DEFAULT_RETRIEVAL_MODE,
    EVAL_FILE,
    LLM_MODEL,
    RESULT_DIR,
    RETRIEVAL_MODES,
    TOP_K,
)


def token_overlap(reference: str, answer: str) -> float:
    ref_tokens = set(reference.lower().split())
    ans_tokens = set(answer.lower().split())
    if len(ref_tokens) <= 1:
        ref_tokens = {char for char in reference if char.strip()}
        ans_tokens = {char for char in answer if char.strip()}
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & ans_tokens) / len(ref_tokens)


def contains_expected_source(result: Dict[str, object], expected_source: str) -> bool:
    if not expected_source:
        return False
    return any(
        expected_source.lower() in str(source["source"]).lower()
        for source in result["sources"]
    )


def load_questions(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _average(rows: List[Dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    return round(mean(values), 4) if values else 0.0


def _safe_answer(
    row: Dict[str, str],
    model: str,
    top_k: int,
    retrieval_mode: str,
) -> Dict[str, object]:
    try:
        from rag_chain import answer_question

        result = answer_question(
            row["question"],
            top_k=top_k,
            model=model,
            retrieval_mode=retrieval_mode,
        )
        reference = row.get("reference_answer", "")
        expected_source = row.get("expected_source", "")
        retrieval_hit = contains_expected_source(result, expected_source)
        overlap = token_overlap(reference, result["answer"])

        return {
            "success": 1,
            "answer": result["answer"],
            "top_sources": "; ".join(
                f"{source['source']}({source['score']:.4f})"
                for source in result["sources"]
            ),
            "retrieval_hit": int(retrieval_hit),
            "reference_overlap": round(overlap, 4),
            "retrieve_time": round(result["retrieve_time"], 4),
            "generate_time": round(result["generate_time"], 4),
            "total_time": round(result["total_time"], 4),
            "error": "",
        }
    except Exception as exc:
        return {
            "success": 0,
            "answer": "",
            "top_sources": "",
            "retrieval_hit": "",
            "reference_overlap": "",
            "retrieve_time": "",
            "generate_time": "",
            "total_time": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_compare(
    question_file: Path,
    top_k: int,
    modes: List[str],
    model: str,
    limit: int | None = None,
) -> Dict[str, object]:
    unsupported_modes = [mode for mode in modes if mode not in RETRIEVAL_MODES]
    if unsupported_modes:
        raise ValueError(
            f"Unsupported retrieval modes: {unsupported_modes}. "
            f"Expected modes from {RETRIEVAL_MODES}"
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_questions(question_file)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"Evaluation file is empty: {question_file}")

    result_rows = []
    for mode in modes:
        print(f"\n=== Retrieval mode: {mode} ===")
        for index, row in enumerate(rows, start=1):
            print(f"[{index}/{len(rows)}] {row['question']}")
            answer_row = _safe_answer(
                row,
                model=model,
                top_k=top_k,
                retrieval_mode=mode,
            )
            result_rows.append(
                {
                    "retrieval_mode": mode,
                    "model": model,
                    "top_k": top_k,
                    "id": row.get("id", index),
                    "category": row.get("category", ""),
                    "question": row["question"],
                    "reference_answer": row.get("reference_answer", ""),
                    "expected_source": row.get("expected_source", ""),
                    **answer_row,
                }
            )

    output_csv = RESULT_DIR / "rag_strategy_compare_results.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    mode_summaries = []
    for mode in modes:
        mode_rows = [row for row in result_rows if row["retrieval_mode"] == mode]
        success_rows = [row for row in mode_rows if row["success"] == 1]
        mode_summaries.append(
            {
                "retrieval_mode": mode,
                "model": model,
                "sample_count": len(mode_rows),
                "success_count": len(success_rows),
                "success_rate": round(len(success_rows) / len(mode_rows), 4),
                "retrieval_hit_rate": _average(success_rows, "retrieval_hit"),
                "average_reference_overlap": _average(
                    success_rows, "reference_overlap"
                ),
                "average_retrieve_time": _average(success_rows, "retrieve_time"),
                "average_generate_time": _average(success_rows, "generate_time"),
                "average_total_time": _average(success_rows, "total_time"),
            }
        )

    summary = {
        "question_file": str(question_file),
        "question_count": len(rows),
        "top_k": top_k,
        "model": model,
        "modes": modes,
        "default_retrieval_mode": DEFAULT_RETRIEVAL_MODE,
        "mode_summaries": mode_summaries,
        "result_file": str(output_csv),
    }
    summary_file = RESULT_DIR / "rag_strategy_compare_summary.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vector and hybrid RAG retrieval strategies"
    )
    parser.add_argument("--file", type=Path, default=EVAL_FILE, help="evaluation CSV")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="retrieved chunks")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=RETRIEVAL_MODES,
        choices=RETRIEVAL_MODES,
        help="retrieval modes to compare",
    )
    parser.add_argument("--model", default=LLM_MODEL, help="Ollama generation model")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit question count for quick tests; omit for full evaluation",
    )
    args = parser.parse_args()

    summary = run_compare(args.file, args.top_k, args.modes, args.model, args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
