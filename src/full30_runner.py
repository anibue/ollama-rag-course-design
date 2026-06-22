"""Full 30-sample evaluation runner with incremental persistence.

完整跑 30 条样本，包括单模型、检索策略、Prompt、双模型和权重调优。
本地 7B 模型一轮很慢，所以每完成一条就写盘，断了也能用 --resume 接着跑。
"""
import argparse
import csv
import importlib
import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import config
from config import COMPARE_MODELS, EVAL_FILE, LLM_MODEL, RESULT_DIR, TOP_K
from eval import contains_expected_source, load_questions, reasoning_faithfulness, token_overlap


SUITE_VERSION = "full30_20260621"
PROMPT_VARIANTS = ["baseline", "stepwise", "reflective"]
STRATEGY_MODES = ["vector", "hybrid", "adaptive"]
WEIGHT_CANDIDATES = [
    (0.30, 0.55, 0.15),
    (0.35, 0.50, 0.15),
    (0.40, 0.45, 0.15),
    (0.45, 0.40, 0.15),
    (0.50, 0.35, 0.15),
]


def _suffix_path(stem: str, suffix: str, ext: str) -> Path:
    return RESULT_DIR / f"{stem}_{suffix}.{ext}"


def _read_existing_keys(path: Path, key_fields: List[str]) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {tuple(row.get(field, "") for field in key_fields) for row in csv.DictReader(file)}


def _append_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    # 评测时间太长，不能等整组跑完再写文件。
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def _load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _avg(rows: List[Dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    return round(mean(values), 4) if values else 0.0


def _safe_answer(
    question: str,
    model: str,
    top_k: int,
    retrieval_mode: str,
    prompt_variant: str,
    num_predict: int,
    timeout: int,
    context_expansion: bool,
) -> Dict[str, object]:
    import rag_chain

    try:
        started = time.perf_counter()
        if retrieval_mode == "adaptive":
            result = rag_chain.answer_question_adaptive(
                question,
                top_k=top_k,
                model=model,
                num_predict=num_predict,
                timeout=timeout,
                expand_context_enabled=context_expansion,
            )
        else:
            result = rag_chain.answer_question(
                question,
                top_k=top_k,
                model=model,
                retrieval_mode=retrieval_mode,
                prompt_variant=prompt_variant,
                num_predict=num_predict,
                timeout=timeout,
                expand_context_enabled=context_expansion,
            )
        result["runner_wall_time"] = time.perf_counter() - started
        return {"success": 1, "error": "", **result}
    except Exception as exc:
        return {
            "success": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "answer": "",
            "sources": [],
            "retrieve_time": "",
            "generate_time": "",
            "total_time": "",
            "retrieval_rounds": "",
            "runner_wall_time": "",
        }


def _metrics(row: Dict[str, str], answer_result: Dict[str, object]) -> Dict[str, object]:
    # 这些指标都偏自动化，只适合做趋势判断，最终答案仍需要抽样人工看。
    answer = str(answer_result.get("answer", ""))
    sources = answer_result.get("sources", []) or []
    reference = row.get("reference_answer", "")
    expected = row.get("expected_source", "")
    retrieval_hit = contains_expected_source({"sources": sources}, expected)
    contexts = [str(source.get("content", "")) for source in sources]
    return {
        "answer": answer,
        "top_sources": "; ".join(
            f"{source.get('source')}({float(source.get('score', 0.0)):.4f})" for source in sources
        ),
        "retrieval_hit": int(retrieval_hit),
        "reference_overlap": round(token_overlap(reference, answer), 4),
        "reasoning_faithfulness": round(reasoning_faithfulness(answer, contexts), 4),
        "retrieve_time": _round_or_blank(answer_result.get("retrieve_time")),
        "generate_time": _round_or_blank(answer_result.get("generate_time")),
        "total_time": _round_or_blank(answer_result.get("total_time")),
        "retrieval_rounds": answer_result.get("retrieval_rounds", 1),
        "error": answer_result.get("error", ""),
        "success": answer_result.get("success", 0),
    }


def _round_or_blank(value: object) -> object:
    if value in (None, ""):
        return ""
    return round(float(value), 4)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_eval_full(rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    # 单模型主实验固定走 hybrid，和后面的策略对比区分开。
    csv_path = _suffix_path("eval_results", args.suffix, "csv")
    summary_path = _suffix_path("eval_summary", args.suffix, "json")
    fieldnames = [
        "id", "category", "question", "reference_answer", "expected_source", "model",
        "top_k", "prompt_variant", "num_predict", "timeout", "context_expansion",
        "success", "answer", "top_sources", "retrieval_hit", "reference_overlap",
        "reasoning_faithfulness", "retrieval_rounds", "retrieve_time", "generate_time",
        "total_time", "error",
    ]
    done = _read_existing_keys(csv_path, ["id"] if args.resume else []) if args.resume else set()
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id", str(index))
        if args.resume and (row_id,) in done:
            continue
        print(f"[eval {index}/{len(rows)}] {row['question']}", flush=True)
        answer = _safe_answer(
            row["question"], args.model, args.top_k, "hybrid", args.prompt,
            args.num_predict, args.timeout, args.context_expansion,
        )
        out = {
            "id": row_id,
            "category": row.get("category", ""),
            "question": row["question"],
            "reference_answer": row.get("reference_answer", ""),
            "expected_source": row.get("expected_source", ""),
            "model": args.model,
            "top_k": args.top_k,
            "prompt_variant": args.prompt,
            "num_predict": args.num_predict,
            "timeout": args.timeout,
            "context_expansion": args.context_expansion,
            **_metrics(row, answer),
        }
        _append_row(csv_path, fieldnames, out)
        _write_eval_summary(summary_path, csv_path, rows, args)


def _write_eval_summary(summary_path: Path, csv_path: Path, rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    result_rows = _load_rows(csv_path)
    success_rows = [row for row in result_rows if row.get("success") == "1"]
    _write_json(summary_path, {
        "question_count": len(rows),
        "sample_count": len(result_rows),
        "success_count": len(success_rows),
        "top_k": args.top_k,
        "model": args.model,
        "prompt_variant": args.prompt,
        "num_predict": args.num_predict,
        "timeout": args.timeout,
        "context_expansion": args.context_expansion,
        "retrieval_hit_rate": _avg(success_rows, "retrieval_hit"),
        "average_reference_overlap": _avg(success_rows, "reference_overlap"),
        "average_reasoning_faithfulness": _avg(success_rows, "reasoning_faithfulness"),
        "average_retrieve_time": _avg(success_rows, "retrieve_time"),
        "average_generate_time": _avg(success_rows, "generate_time"),
        "average_total_time": _avg(success_rows, "total_time"),
        "result_file": str(csv_path),
        "complete": len(result_rows) == len(rows),
    })


def run_strategy_full(rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    # vector 是优化前基线，hybrid/adaptive 是优化后的两个方案。
    csv_path = _suffix_path("rag_strategy_compare_results", args.suffix, "csv")
    summary_path = _suffix_path("rag_strategy_compare_summary", args.suffix, "json")
    fieldnames = [
        "retrieval_mode", "id", "category", "question", "reference_answer", "expected_source",
        "model", "top_k", "prompt_variant", "num_predict", "timeout", "context_expansion",
        "success", "answer", "top_sources", "retrieval_hit", "reference_overlap",
        "reasoning_faithfulness", "retrieval_rounds", "retrieve_time", "generate_time",
        "total_time", "error",
    ]
    done = _read_existing_keys(csv_path, ["retrieval_mode", "id"]) if args.resume else set()
    for mode in STRATEGY_MODES:
        for index, row in enumerate(rows, start=1):
            row_id = row.get("id", str(index))
            if args.resume and (mode, row_id) in done:
                continue
            print(f"[strategy:{mode} {index}/{len(rows)}] {row['question']}", flush=True)
            answer = _safe_answer(
                row["question"], args.model, args.top_k, mode, args.prompt,
                args.num_predict, args.timeout, args.context_expansion,
            )
            out = {
                "retrieval_mode": mode,
                "id": row_id,
                "category": row.get("category", ""),
                "question": row["question"],
                "reference_answer": row.get("reference_answer", ""),
                "expected_source": row.get("expected_source", ""),
                "model": args.model,
                "top_k": args.top_k,
                "prompt_variant": args.prompt,
                "num_predict": args.num_predict,
                "timeout": args.timeout,
                "context_expansion": args.context_expansion,
                **_metrics(row, answer),
            }
            _append_row(csv_path, fieldnames, out)
            _write_strategy_summary(summary_path, csv_path, rows, args)


def _write_strategy_summary(summary_path: Path, csv_path: Path, rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    result_rows = _load_rows(csv_path)
    summaries = []
    for mode in STRATEGY_MODES:
        mode_rows = [row for row in result_rows if row.get("retrieval_mode") == mode]
        success_rows = [row for row in mode_rows if row.get("success") == "1"]
        summaries.append({
            "retrieval_mode": mode,
            "sample_count": len(mode_rows),
            "success_count": len(success_rows),
            "success_rate": round(len(success_rows) / len(mode_rows), 4) if mode_rows else 0.0,
            "retrieval_hit_rate": _avg(success_rows, "retrieval_hit"),
            "average_reference_overlap": _avg(success_rows, "reference_overlap"),
            "average_reasoning_faithfulness": _avg(success_rows, "reasoning_faithfulness"),
            "average_retrieval_rounds": _avg(success_rows, "retrieval_rounds"),
            "average_retrieve_time": _avg(success_rows, "retrieve_time"),
            "average_generate_time": _avg(success_rows, "generate_time"),
            "average_total_time": _avg(success_rows, "total_time"),
        })
    _write_json(summary_path, {
        "question_count": len(rows),
        "top_k": args.top_k,
        "model": args.model,
        "prompt_variant": args.prompt,
        "num_predict": args.num_predict,
        "timeout": args.timeout,
        "context_expansion": args.context_expansion,
        "modes": STRATEGY_MODES,
        "mode_summaries": summaries,
        "result_file": str(csv_path),
        "complete": len(result_rows) == len(rows) * len(STRATEGY_MODES),
    })


def run_prompt_full(rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    csv_path = _suffix_path("prompt_compare_results", args.suffix, "csv")
    summary_path = _suffix_path("prompt_compare_summary", args.suffix, "json")
    fieldnames = [
        "prompt_variant", "id", "category", "question", "reference_answer", "expected_source",
        "model", "top_k", "num_predict", "timeout", "context_expansion", "success",
        "answer", "top_sources", "retrieval_hit", "reference_overlap", "reasoning_faithfulness",
        "retrieval_rounds", "retrieve_time", "generate_time", "total_time", "error",
    ]
    done = _read_existing_keys(csv_path, ["prompt_variant", "id"]) if args.resume else set()
    for variant in PROMPT_VARIANTS:
        for index, row in enumerate(rows, start=1):
            row_id = row.get("id", str(index))
            if args.resume and (variant, row_id) in done:
                continue
            print(f"[prompt:{variant} {index}/{len(rows)}] {row['question']}", flush=True)
            answer = _safe_answer(
                row["question"], args.model, args.top_k, "hybrid", variant,
                args.num_predict, args.timeout, args.context_expansion,
            )
            out = {
                "prompt_variant": variant,
                "id": row_id,
                "category": row.get("category", ""),
                "question": row["question"],
                "reference_answer": row.get("reference_answer", ""),
                "expected_source": row.get("expected_source", ""),
                "model": args.model,
                "top_k": args.top_k,
                "num_predict": args.num_predict,
                "timeout": args.timeout,
                "context_expansion": args.context_expansion,
                **_metrics(row, answer),
            }
            _append_row(csv_path, fieldnames, out)
            _write_prompt_summary(summary_path, csv_path, rows, args)


def _write_prompt_summary(summary_path: Path, csv_path: Path, rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    result_rows = _load_rows(csv_path)
    summaries = []
    for variant in PROMPT_VARIANTS:
        variant_rows = [row for row in result_rows if row.get("prompt_variant") == variant]
        success_rows = [row for row in variant_rows if row.get("success") == "1"]
        summaries.append({
            "prompt_variant": variant,
            "sample_count": len(variant_rows),
            "success_count": len(success_rows),
            "success_rate": round(len(success_rows) / len(variant_rows), 4) if variant_rows else 0.0,
            "retrieval_hit_rate": _avg(success_rows, "retrieval_hit"),
            "average_reference_overlap": _avg(success_rows, "reference_overlap"),
            "average_reasoning_faithfulness": _avg(success_rows, "reasoning_faithfulness"),
            "average_generate_time": _avg(success_rows, "generate_time"),
            "average_total_time": _avg(success_rows, "total_time"),
        })
    _write_json(summary_path, {
        "question_count": len(rows),
        "top_k": args.top_k,
        "model": args.model,
        "variants": PROMPT_VARIANTS,
        "num_predict": args.num_predict,
        "timeout": args.timeout,
        "context_expansion": args.context_expansion,
        "variant_summaries": summaries,
        "result_file": str(csv_path),
        "complete": len(result_rows) == len(rows) * len(PROMPT_VARIANTS),
    })


def run_model_full(rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    csv_path = _suffix_path("model_compare_results", args.suffix, "csv")
    summary_path = _suffix_path("model_compare_summary", args.suffix, "json")
    fieldnames = [
        "model", "id", "category", "question", "reference_answer", "expected_source",
        "top_k", "prompt_variant", "num_predict", "timeout", "context_expansion", "success",
        "answer", "top_sources", "retrieval_hit", "reference_overlap", "reasoning_faithfulness",
        "retrieval_rounds", "retrieve_time", "generate_time", "total_time", "error",
    ]
    done = _read_existing_keys(csv_path, ["model", "id"]) if args.resume else set()
    for model in args.models:
        for index, row in enumerate(rows, start=1):
            row_id = row.get("id", str(index))
            if args.resume and (model, row_id) in done:
                continue
            print(f"[model:{model} {index}/{len(rows)}] {row['question']}", flush=True)
            answer = _safe_answer(
                row["question"], model, args.top_k, "hybrid", args.prompt,
                args.num_predict, args.timeout, args.context_expansion,
            )
            out = {
                "model": model,
                "id": row_id,
                "category": row.get("category", ""),
                "question": row["question"],
                "reference_answer": row.get("reference_answer", ""),
                "expected_source": row.get("expected_source", ""),
                "top_k": args.top_k,
                "prompt_variant": args.prompt,
                "num_predict": args.num_predict,
                "timeout": args.timeout,
                "context_expansion": args.context_expansion,
                **_metrics(row, answer),
            }
            _append_row(csv_path, fieldnames, out)
            _write_model_summary(summary_path, csv_path, rows, args)


def _write_model_summary(summary_path: Path, csv_path: Path, rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    result_rows = _load_rows(csv_path)
    summaries = []
    for model in args.models:
        model_rows = [row for row in result_rows if row.get("model") == model]
        success_rows = [row for row in model_rows if row.get("success") == "1"]
        summaries.append({
            "model": model,
            "sample_count": len(model_rows),
            "success_count": len(success_rows),
            "success_rate": round(len(success_rows) / len(model_rows), 4) if model_rows else 0.0,
            "retrieval_hit_rate": _avg(success_rows, "retrieval_hit"),
            "average_reference_overlap": _avg(success_rows, "reference_overlap"),
            "average_reasoning_faithfulness": _avg(success_rows, "reasoning_faithfulness"),
            "average_retrieve_time": _avg(success_rows, "retrieve_time"),
            "average_generate_time": _avg(success_rows, "generate_time"),
            "average_total_time": _avg(success_rows, "total_time"),
        })
    _write_json(summary_path, {
        "question_count": len(rows),
        "top_k": args.top_k,
        "models": args.models,
        "prompt_variant": args.prompt,
        "num_predict": args.num_predict,
        "timeout": args.timeout,
        "context_expansion": args.context_expansion,
        "model_summaries": summaries,
        "result_file": str(csv_path),
        "complete": len(result_rows) == len(rows) * len(args.models),
    })


def run_weight_full(rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    output_path = _suffix_path("weight_tune_results", args.suffix, "json")
    results = []
    for emb, lex, entity in WEIGHT_CANDIDATES:
        # 权重调优不生成答案，只比较排序是否把目标来源推上来。
        config.EMBEDDING_SCORE_WEIGHT_V3 = emb
        config.LEXICAL_SCORE_WEIGHT_V3 = lex
        config.ENTITY_SCORE_WEIGHT = entity
        import rag_chain
        importlib.reload(rag_chain)
        hits = []
        score_margins = []
        print(f"[weight emb={emb:.2f} lex={lex:.2f} entity={entity:.2f}]", flush=True)
        for row in rows:
            retrieved = rag_chain.retrieve(row["question"], top_k=args.top_k, mode="hybrid")
            expected = row.get("expected_source", "").lower()
            hits.append(int(any(expected in str(item["source"]).lower() for item in retrieved)))
            scores = [float(item["score"]) for item in retrieved]
            if scores:
                score_margins.append(scores[0] - scores[-1] if len(scores) > 1 else scores[0])
        results.append({
            "emb": emb,
            "lex": lex,
            "entity": entity,
            "hit_rate": round(mean(hits), 4) if hits else 0.0,
            "avg_score_margin": round(mean(score_margins), 4) if score_margins else 0.0,
        })
        best = max(results, key=lambda item: (item["hit_rate"], item["avg_score_margin"]))
        _write_json(output_path, {
            "question_count": len(rows),
            "top_k": args.top_k,
            "model": args.model,
            "metric": "retrieval_hit_rate_then_average_top_score_margin",
            "best": best,
            "all": results,
            "complete": len(results) == len(WEIGHT_CANDIDATES),
        })
    config.EMBEDDING_SCORE_WEIGHT_V3 = 0.40
    config.LEXICAL_SCORE_WEIGHT_V3 = 0.45
    config.ENTITY_SCORE_WEIGHT = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full 30-sample RAG evaluation suite incrementally")
    parser.add_argument("--file", type=Path, default=EVAL_FILE)
    parser.add_argument("--suffix", default=SUITE_VERSION)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--models", nargs="+", default=COMPARE_MODELS)
    parser.add_argument("--prompt", choices=PROMPT_VARIANTS, default="baseline")
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--context-expansion", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["eval", "strategy", "prompt", "model", "weight"],
        choices=["eval", "strategy", "prompt", "model", "weight"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_questions(args.file)
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 eval questions, got {len(rows)} from {args.file}")
    print(
        f"Full suite: {len(rows)} questions, top_k={args.top_k}, num_predict={args.num_predict}, "
        f"timeout={args.timeout}, context_expansion={args.context_expansion}, suffix={args.suffix}",
        flush=True,
    )
    if "eval" in args.tasks:
        run_eval_full(rows, args)
    if "strategy" in args.tasks:
        run_strategy_full(rows, args)
    if "prompt" in args.tasks:
        run_prompt_full(rows, args)
    if "model" in args.tasks:
        run_model_full(rows, args)
    if "weight" in args.tasks:
        run_weight_full(rows, args)
    print("Full suite tasks finished.", flush=True)


if __name__ == "__main__":
    main()
