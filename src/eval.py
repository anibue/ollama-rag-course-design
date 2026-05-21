import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from config import EVAL_FILE, RESULT_DIR, TOP_K
from rag_chain import answer_question


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


def run_eval(question_file: Path, top_k: int) -> Dict[str, object]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_questions(question_file)
    if not rows:
        raise RuntimeError(f"评测文件为空：{question_file}")

    result_rows = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['question']}")
        result = answer_question(row["question"], top_k=top_k)
        reference = row.get("reference_answer", "")
        expected_source = row.get("expected_source", "")
        retrieval_hit = contains_expected_source(result, expected_source)
        overlap = token_overlap(reference, result["answer"])

        result_rows.append(
            {
                "id": row.get("id", index),
                "category": row.get("category", ""),
                "question": row["question"],
                "reference_answer": reference,
                "answer": result["answer"],
                "expected_source": expected_source,
                "top_sources": "; ".join(
                    f"{source['source']}({source['score']:.4f})"
                    for source in result["sources"]
                ),
                "retrieval_hit": int(retrieval_hit),
                "reference_overlap": round(overlap, 4),
                "retrieve_time": round(result["retrieve_time"], 4),
                "generate_time": round(result["generate_time"], 4),
                "total_time": round(result["total_time"], 4),
            }
        )

    output_csv = RESULT_DIR / "eval_results.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    summary = {
        "sample_count": len(result_rows),
        "top_k": top_k,
        "retrieval_hit_rate": round(
            mean(row["retrieval_hit"] for row in result_rows), 4
        ),
        "average_reference_overlap": round(
            mean(row["reference_overlap"] for row in result_rows), 4
        ),
        "average_retrieve_time": round(
            mean(row["retrieve_time"] for row in result_rows), 4
        ),
        "average_generate_time": round(
            mean(row["generate_time"] for row in result_rows), 4
        ),
        "average_total_time": round(mean(row["total_time"] for row in result_rows), 4),
        "result_file": str(output_csv),
    }
    summary_file = RESULT_DIR / "eval_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="批量评测 RAG 问答效果")
    parser.add_argument("--file", type=Path, default=EVAL_FILE, help="评测问题 CSV")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="检索片段数量")
    args = parser.parse_args()

    summary = run_eval(args.file, args.top_k)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
