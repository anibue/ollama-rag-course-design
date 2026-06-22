"""Retrieval weight grid-search tuning.

只比较三路检索权重，不调用 LLM 生成答案。
这样跑得快一些，也更容易看出排序本身的问题。
"""
import argparse
import importlib
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import config
from config import EVAL_FILE, LLM_MODEL, RESULT_DIR, TOP_K
from eval import load_questions


EMB_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50]
LEX_CANDIDATES = [0.35, 0.40, 0.45, 0.50, 0.55]
ENTITY_CANDIDATES = [0.05, 0.10, 0.15, 0.20]
DEFAULT_WEIGHTS = (0.40, 0.45, 0.15)


def _run_one(
    question_file: Path,
    top_k: int,
    emb: float,
    lex: float,
    entity: float,
    limit: int | None,
) -> Dict[str, object]:
    # 这里直接改配置再 reload rag_chain，省得为了调参改主流程接口。
    config.EMBEDDING_SCORE_WEIGHT_V3 = emb
    config.LEXICAL_SCORE_WEIGHT_V3 = lex
    config.ENTITY_SCORE_WEIGHT = entity

    import rag_chain
    importlib.reload(rag_chain)

    rows = load_questions(question_file)
    if limit is not None:
        rows = rows[:limit]

    hits: List[int] = []
    score_margins: List[float] = []
    for row in rows:
        try:
            # 调权重只看 expected_source 是否进 Top-K，不等待 LLM 生成。
            retrieved = rag_chain.retrieve(row["question"], top_k=top_k, mode="hybrid")
            expected = row.get("expected_source", "").lower()
            hit = any(expected in str(item["source"]).lower() for item in retrieved) if expected else False
            hits.append(int(hit))
            scores = [float(item["score"]) for item in retrieved]
            if scores:
                score_margins.append(scores[0] - scores[-1] if len(scores) > 1 else scores[0])
        except Exception:
            pass

    return {
        "emb": emb,
        "lex": lex,
        "entity": entity,
        "hit_rate": round(mean(hits), 4) if hits else 0.0,
        "avg_score_margin": round(mean(score_margins), 4) if score_margins else 0.0,
    }


def grid_search(
    question_file: Path,
    top_k: int,
    model: str,
    limit: int | None = None,
    entity_fixed: float | None = None,
    output_suffix: str = "",
) -> Tuple[List[Dict], Dict]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []

    entity_range = [entity_fixed] if entity_fixed is not None else ENTITY_CANDIDATES
    candidates = [
        (emb, lex, entity)
        for entity in entity_range
        for emb in EMB_CANDIDATES
        for lex in LEX_CANDIDATES
        if abs(emb + lex + entity - 1.0) <= 0.01
    ]

    for index, (emb, lex, entity) in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] emb={emb:.2f} lex={lex:.2f} entity={entity:.2f}")
        result = _run_one(question_file, top_k, emb, lex, entity, limit)
        results.append(result)
        print(
            f"  hit_rate={result['hit_rate']:.4f}  "
            f"avg_score_margin={result['avg_score_margin']:.4f}"
        )

    if not results:
        raise RuntimeError("No valid weight combinations found.")

    best = max(results, key=lambda item: (item["hit_rate"], item["avg_score_margin"]))
    config.EMBEDDING_SCORE_WEIGHT_V3, config.LEXICAL_SCORE_WEIGHT_V3, config.ENTITY_SCORE_WEIGHT = DEFAULT_WEIGHTS
    import rag_chain
    importlib.reload(rag_chain)

    payload = {
        "question_file": str(question_file),
        "top_k": top_k,
        "model": model,
        "limit": limit,
        "metric": "retrieval_hit_rate_then_average_top_score_margin",
        "best": best,
        "all": results,
    }
    suffix = f"_{output_suffix}" if output_suffix else ""
    output_json = RESULT_DIR / f"weight_tune_results{suffix}.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved: {output_json}")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    return results, best


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval weight grid-search tuning")
    parser.add_argument("--file", type=Path, default=EVAL_FILE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--entity-fixed", type=float, default=None)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    grid_search(
        args.file,
        args.top_k,
        args.model,
        limit=args.limit,
        entity_fixed=args.entity_fixed,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
