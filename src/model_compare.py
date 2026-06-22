import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from config import COMPARE_MODELS, EVAL_FILE, RESULT_DIR, TOP_K


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
    num_predict: int | None = None,
    timeout: int | None = None,
    expand_context_enabled: bool | None = None,
) -> Dict[str, object]:
    try:
        from rag_chain import answer_question

        result = answer_question(
            row["question"],
            top_k=top_k,
            model=model,
            num_predict=num_predict,
            timeout=timeout,
            expand_context_enabled=expand_context_enabled,
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
    models: List[str],
    limit: int | None = None,
    num_predict: int | None = None,
    timeout: int | None = None,
    expand_context_enabled: bool | None = None,
    output_suffix: str = "",
) -> Dict[str, object]:
    # 双模型对比沿用同一套检索流程，重点看生成模型自己的差异。
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_questions(question_file)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"Evaluation file is empty: {question_file}")

    result_rows = []
    for model in models:
        print(f"\n=== Model: {model} ===")
        for index, row in enumerate(rows, start=1):
            print(f"[{index}/{len(rows)}] {row['question']}")
            answer_row = _safe_answer(
                row,
                model=model,
                top_k=top_k,
                num_predict=num_predict,
                timeout=timeout,
                expand_context_enabled=expand_context_enabled,
            )
            result_rows.append(
                {
                    "model": model,
                    "id": row.get("id", index),
                    "category": row.get("category", ""),
                    "question": row["question"],
                    "reference_answer": row.get("reference_answer", ""),
                    "expected_source": row.get("expected_source", ""),
                    **answer_row,
                }
            )

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_csv = RESULT_DIR / f"model_compare_results{suffix}.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    model_summaries = []
    for model in models:
        model_rows = [row for row in result_rows if row["model"] == model]
        success_rows = [row for row in model_rows if row["success"] == 1]
        model_summaries.append(
            {
                "model": model,
                "sample_count": len(model_rows),
                "success_count": len(success_rows),
                "success_rate": round(len(success_rows) / len(model_rows), 4),
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
        "models": models,
        "num_predict": num_predict,
        "timeout": timeout,
        "context_expansion": expand_context_enabled,
        "model_summaries": model_summaries,
        "result_file": str(output_csv),
    }
    summary_file = RESULT_DIR / f"model_compare_summary{suffix}.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG answers across models")
    parser.add_argument("--file", type=Path, default=EVAL_FILE, help="evaluation CSV")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="retrieved chunks")
    parser.add_argument(
        "--models",
        nargs="+",
        default=COMPARE_MODELS,
        help="Ollama model names to compare",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit question count for quick tests; omit for full evaluation",
    )
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--no-context-expansion", action="store_true")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    summary = run_compare(
        args.file,
        args.top_k,
        args.models,
        args.limit,
        args.num_predict,
        args.timeout,
        not args.no_context_expansion,
        args.output_suffix,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




