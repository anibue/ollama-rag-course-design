"""Three-prompt strategy comparison experiment.

固定题目、模型和检索方式，只换 baseline / stepwise / reflective。
这样结果更容易解释，不会把 Prompt 差异和检索差异混在一起。
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from config import EVAL_FILE, LLM_MODEL, RESULT_DIR, TOP_K
from rag_chain import answer_question
from eval import token_overlap, contains_expected_source, load_questions, reasoning_faithfulness


PROMPT_VARIANTS = ["baseline", "stepwise", "reflective"]


def _safe_answer(
    row: Dict[str, str],
    model: str,
    top_k: int,
    prompt_variant: str,
    num_predict: int | None = None,
    timeout: int | None = None,
    expand_context_enabled: bool | None = None,
) -> Dict[str, object]:
    try:
        result = answer_question(
            row["question"],
            top_k=top_k,
            model=model,
            prompt_variant=prompt_variant,
            num_predict=num_predict,
            timeout=timeout,
            expand_context_enabled=expand_context_enabled,
        )
        reference = row.get("reference_answer", "")
        expected_source = row.get("expected_source", "")
        retrieval_hit = contains_expected_source(result, expected_source)
        overlap = token_overlap(reference, result["answer"])
        contexts = [s["content"] for s in result["sources"]]
        rf = reasoning_faithfulness(result["answer"], contexts)

        return {
            "success": 1,
            "answer": result["answer"],
            "retrieval_hit": int(retrieval_hit),
            "reference_overlap": round(overlap, 4),
            "reasoning_faithfulness": round(rf, 4),
            "retrieve_time": round(result["retrieve_time"], 4),
            "generate_time": round(result["generate_time"], 4),
            "total_time": round(result["total_time"], 4),
            "error": "",
        }
    except Exception as exc:
        return {
            "success": 0,
            "answer": "",
            "retrieval_hit": "",
            "reference_overlap": "",
            "reasoning_faithfulness": "",
            "retrieve_time": "",
            "generate_time": "",
            "total_time": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_prompt_compare(
    question_file: Path,
    top_k: int,
    model: str,
    variants: List[str] = None,
    limit: int | None = None,
    num_predict: int | None = None,
    timeout: int | None = None,
    expand_context_enabled: bool | None = None,
    output_suffix: str = "",
) -> Dict[str, object]:
    if variants is None:
        variants = PROMPT_VARIANTS

    # 检索和模型不动，只换 Prompt，避免把策略差异和检索差异混在一起。
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_questions(question_file)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"Evaluation file is empty: {question_file}")

    def _avg(rows_: List[Dict], field: str) -> float:
        vals = [float(r[field]) for r in rows_ if r.get(field) not in ("", None)]
        return round(mean(vals), 4) if vals else 0.0

    result_rows: List[Dict[str, object]] = []
    for variant in variants:
        print(f"\n=== Prompt variant: {variant} ===")
        for idx, row in enumerate(rows, start=1):
            print(f"[{idx}/{len(rows)}] {row['question']}")
            ar = _safe_answer(row, model=model, top_k=top_k, prompt_variant=variant, num_predict=num_predict, timeout=timeout, expand_context_enabled=expand_context_enabled)
            result_rows.append(
                {
                    "prompt_variant": variant,
                    "id": row.get("id", idx),
                    "category": row.get("category", ""),
                    "question": row["question"],
                    "reference_answer": row.get("reference_answer", ""),
                    **ar,
                }
            )

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_csv = RESULT_DIR / f"prompt_compare_results{suffix}.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    variant_summaries = []
    for variant in variants:
        vrows = [r for r in result_rows if r["prompt_variant"] == variant]
        srows = [r for r in vrows if r["success"] == 1]
        variant_summaries.append(
            {
                "prompt_variant": variant,
                "sample_count": len(vrows),
                "success_count": len(srows),
                "retrieval_hit_rate": _avg(srows, "retrieval_hit"),
                "average_reference_overlap": _avg(srows, "reference_overlap"),
                "average_reasoning_faithfulness": _avg(srows, "reasoning_faithfulness"),
                "average_generate_time": _avg(srows, "generate_time"),
                "average_total_time": _avg(srows, "total_time"),
            }
        )

    summary = {
        "question_file": str(question_file),
        "question_count": len(rows),
        "top_k": top_k,
        "model": model,
        "variants": variants,
        "num_predict": num_predict,
        "timeout": timeout,
        "context_expansion": expand_context_enabled,
        "variant_summaries": variant_summaries,
        "result_file": str(output_csv),
    }
    summary_file = RESULT_DIR / f"prompt_compare_summary{suffix}.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="对比三种 Prompt 策略（baseline/stepwise/reflective）")
    parser.add_argument("--file", type=Path, default=EVAL_FILE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=PROMPT_VARIANTS,
        choices=PROMPT_VARIANTS,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--no-context-expansion", action="store_true")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    summary = run_prompt_compare(
        args.file, args.top_k, args.model, args.variants, args.limit, args.num_predict, args.timeout, not args.no_context_expansion, args.output_suffix
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



