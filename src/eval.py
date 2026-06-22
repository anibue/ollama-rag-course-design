import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from config import EVAL_FILE, LLM_MODEL, RESULT_DIR, TOP_K
from rag_chain import answer_question, generate_with_ollama


# ---------------------------------------------------------------------------
# Existing metrics
# ---------------------------------------------------------------------------


def token_overlap(reference: str, answer: str) -> float:
    # 粗略覆盖率，不等同于人工语义评分，只适合看整体趋势。
    ref_tokens = set(reference.lower().split())
    ans_tokens = set(answer.lower().split())
    if len(ref_tokens) <= 1:
        ref_tokens = {char for char in reference if char.strip()}
        ans_tokens = {char for char in answer if char.strip()}
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & ans_tokens) / len(ref_tokens)


def contains_expected_source(result: Dict[str, object], expected_source: str) -> bool:
    # 命中率只看来源文件是否进了 Top-K，先把检索环节单独量出来。
    if not expected_source:
        return False
    return any(
        expected_source.lower() in str(source["source"]).lower()
        for source in result["sources"]
    )


def load_questions(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


# ---------------------------------------------------------------------------
# New multi-dimensional metrics (RAG Survey Gao / A-RAG / Self-Correcting RAG)
# ---------------------------------------------------------------------------


def faithfulness_score(
    answer: str, contexts: List[str], model: str = LLM_MODEL
) -> Optional[float]:
    """LLM-judged faithfulness: is the answer grounded in the retrieved contexts?
    Returns a 0-1 float, or None if parsing fails.
    Basis: RAG Survey (Gao) answer_faithfulness / Self-Correcting RAG忠实度指标.
    """
    excerpt = " ".join(c[:200] for c in contexts[:3])
    prompt = (
        f"判断以下回答是否完全基于参考资料。评分 0-10，0=完全编造，10=完全有据可查。\n\n"
        f"参考资料：{excerpt}\n\n"
        f"回答：{answer[:500]}\n\n"
        f"评分（仅输出数字）："
    )
    try:
        raw = generate_with_ollama(prompt, model=model)
        m = re.search(r"\d+", raw)
        if m:
            return min(int(m.group()), 10) / 10.0
    except Exception:
        pass
    return None


def reasoning_faithfulness(answer: str, contexts: List[str]) -> float:
    """Fraction of [片段N] citations in the answer that map to valid context indices.
    Basis: ProRAG / ReasonRAG 过程监督——推理步骤是否引用了检索片段.
    """
    citations = re.findall(r"\[(?:片段)?(\d+)\]", answer)
    if not citations:
        return 0.0
    valid = [c for c in citations if 1 <= int(c) <= len(contexts)]
    return len(valid) / len(citations)


def test_rejection(model: str = LLM_MODEL) -> float:
    """Measures the system's ability to refuse out-of-scope questions.
    Basis: RAG: Architectures & Robustness robustness evaluation.
    """
    rejection_questions = [
        "今天天气怎么样？",
        "你喜欢吃什么水果？",
        "2026年世界杯冠军是谁？",
    ]
    correct = 0
    for q in rejection_questions:
        try:
            ans = answer_question(q)["answer"]
            if ("没有找到明确答案" in ans) or ("无法回答" in ans) or ("知识库中没有" in ans):
                correct += 1
        except Exception:
            pass
    return correct / len(rejection_questions) if rejection_questions else 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_eval(
    question_file: Path,
    top_k: int,
    model: str = LLM_MODEL,
    prompt_variant: str = "stepwise",
    include_faithfulness: bool = False,
    include_rejection: bool = False,
    num_predict: int | None = None,
    timeout: int | None = None,
    expand_context_enabled: bool | None = None,
    output_suffix: str = "",
    limit: int | None = None,
) -> Dict[str, object]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_questions(question_file)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"评测文件为空：{question_file}")

    result_rows = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['question']}")
        # 每条问题单独跑完整链路，耗时会长，但更接近真实问答过程。
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
        row_out: Dict[str, object] = {
            "id": row.get("id", idx),
            "category": row.get("category", ""),
            "question": row["question"],
            "reference_answer": reference,
            "answer": result["answer"],
            "expected_source": expected_source,
            "top_sources": "; ".join(
                f"{s['source']}({s['score']:.4f})" for s in result["sources"]
            ),
            "retrieval_hit": int(retrieval_hit),
            "reference_overlap": round(overlap, 4),
            "reasoning_faithfulness": round(reasoning_faithfulness(result["answer"], contexts), 4),
            "retrieve_time": round(result["retrieve_time"], 4),
            "generate_time": round(result["generate_time"], 4),
            "total_time": round(result["total_time"], 4),
        }

        if include_faithfulness:
            fs = faithfulness_score(result["answer"], contexts, model=model)
            row_out["faithfulness_score"] = round(fs, 4) if fs is not None else ""

        result_rows.append(row_out)

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_csv = RESULT_DIR / f"eval_results{suffix}.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    def _mean(field: str) -> float:
        vals = [float(r[field]) for r in result_rows if r.get(field) not in ("", None)]
        return round(mean(vals), 4) if vals else 0.0

    summary: Dict[str, object] = {
        "question_count": len(rows),
        "sample_count": len(result_rows),
        "top_k": top_k,
        "model": model,
        "prompt_variant": prompt_variant,
        "num_predict": num_predict,
        "timeout": timeout,
        "context_expansion": expand_context_enabled,
        "retrieval_hit_rate": _mean("retrieval_hit"),
        "average_reference_overlap": _mean("reference_overlap"),
        "average_reasoning_faithfulness": _mean("reasoning_faithfulness"),
        "average_retrieve_time": _mean("retrieve_time"),
        "average_generate_time": _mean("generate_time"),
        "average_total_time": _mean("total_time"),
        "result_file": str(output_csv),
    }

    if include_faithfulness:
        summary["average_faithfulness_score"] = _mean("faithfulness_score")

    if include_rejection:
        print("测试拒答准确率...")
        summary["rejection_accuracy"] = round(test_rejection(model=model), 4)

    summary_file = RESULT_DIR / f"eval_summary{suffix}.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="批量评测 RAG 问答效果")
    parser.add_argument("--file", type=Path, default=EVAL_FILE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument(
        "--prompt",
        choices=["baseline", "stepwise", "reflective"],
        default="stepwise",
        help="Prompt 策略",
    )
    parser.add_argument("--faithfulness", action="store_true", help="启用 LLM 忠实度评分（增加耗时）")
    parser.add_argument("--rejection", action="store_true", help="测试拒答准确率")
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--no-context-expansion", action="store_true")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = run_eval(
        args.file,
        args.top_k,
        model=args.model,
        prompt_variant=args.prompt,
        include_faithfulness=args.faithfulness,
        include_rejection=args.rejection,
        num_predict=args.num_predict,
        timeout=args.timeout,
        expand_context_enabled=not args.no_context_expansion,
        output_suffix=args.output_suffix,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




