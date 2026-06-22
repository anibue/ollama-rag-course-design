import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from langchain_ollama import OllamaEmbeddings
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ADAPTIVE_TOP_K_SECOND,
    ANSWER_MODES,
    CONTEXT_EXPANSION,
    DEFAULT_RETRIEVAL_MODE,
    EMBEDDING_MODEL,
    EMBEDDING_SCORE_WEIGHT,
    EMBEDDING_SCORE_WEIGHT_V3,
    ENABLE_FAITHFULNESS_CHECK,
    ENTITY_SCORE_WEIGHT,
    EXPANSION_WINDOW,
    GENERATE_TIMEOUT,
    INDEX_FILE,
    LEXICAL_SCORE_WEIGHT,
    LEXICAL_SCORE_WEIGHT_V3,
    LLM_MODEL,
    NUM_CTX,
    NUM_PREDICT,
    OLLAMA_BASE_URL,
    RETRIEVAL_MODES,
    TEMPERATURE,
    TOP_K,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """你是一个严谨的计算机专业文档问答助手。
请只依据【参考资料】回答【用户问题】。如果参考资料没有明确依据，请回答"知识库中没有找到明确答案"，不要补充资料外事实。

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 先用一段话直接回答问题；
2. 再分点说明关键概念、步骤或原因；
3. 最后列出依据来源，引用片段编号和文件名。
"""

# 分步版适合评测引用是否对得上，缺点是回答会更长一点。
PROMPT_TEMPLATE_STEPWISE = """你是一个严谨的计算机专业文档问答助手。请按以下步骤回答问题：

## 步骤 1：信息评估
先判断【参考资料】中的信息是否足以回答【用户问题】。
- 如果信息充分，继续步骤 2。
- 如果信息不足，直接回答"知识库中没有找到明确答案"并说明缺少哪方面的信息。

## 步骤 2：关键事实提取
从参考资料中提取与问题相关的关键事实，列出每条事实对应的片段编号。
格式：[片段编号] 事实内容

## 步骤 3：推理与回答
基于提取的关键事实进行推理，给出最终答案。
- 每个结论必须标注来源片段编号（如"[片段1]"）
- 如果推理过程中发现信息不足，明确指出

## 步骤 4：引用来源
列出答案所依据的所有片段编号和文件名。

【参考资料】
{context}

【用户问题】
{question}

请按上述步骤回答："""

# 反思版主要用来观察遗漏和无依据内容，不一定适合日常问答默认开启。
PROMPT_TEMPLATE_REFLECTIVE = """你是一个严谨的计算机专业文档问答助手。

【参考资料】
{context}

【用户问题】
{question}

请分两轮完成：

**第一轮回答：**
根据参考资料直接回答问题。

**自我反思：**
逐条检查第一轮回答中的每个声明是否有参考资料支持。对无依据的声明标注"[无依据]"。

**修正后的最终回答：**
删除或修正无依据的声明，给出最终准确答案，并标注引用来源（片段编号和文件名）。"""

# adaptive 检索会先问一次“资料够不够”，所以耗时会增加。
SUFFICIENCY_PROMPT = """请判断以下参考资料是否足以回答用户问题。
如果信息充分，回答"充分"；如果信息不足或答案不确定，回答"不足"。

【参考资料】
{context}

【用户问题】
{question}

判断（仅回答"充分"或"不足"）："""

# 忠实度自检只是辅助指标，不能替代人工看答案。
FAITHFULNESS_CHECK_PROMPT = """请检查以下回答中的每个事实性声明是否被参考资料支持。

【参考资料】
{context}

【回答】
{answer}

对回答中的每个事实性声明，判断：
- "支持"：声明有明确的参考资料依据
- "部分支持"：声明与参考资料相关但细节不完全一致
- "不支持"：声明在参考资料中找不到依据

请逐条列出判断结果，最后给出整体忠实度评分（0-10）。
格式：
1. [声明] → [支持/部分支持/不支持]
...
整体忠实度：X/10"""

# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def load_index(path: Path = INDEX_FILE) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"知识库索引不存在：{path}。请先运行 python src/build_kb.py")
    return json.loads(path.read_text(encoding="utf-8"))


def min_max_normalize(values) -> List[float]:
    low = float(min(values))
    high = float(max(values))
    if high == low:
        return [0.0 for _ in values]
    return [(float(v) - low) / (high - low) for v in values]


# ---------------------------------------------------------------------------
# Entity matching (A-RAG keyword_search / LightRAG entity-level retrieval)
# ---------------------------------------------------------------------------


def entity_match_score(question: str, chunk_text: str) -> float:
    """Fraction of question entities/keywords that appear in chunk_text."""
    # 这里没有上复杂分词，课程材料里的术语用规则抽取已经够用。
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[一-鿿]{2,}|\d+", question)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t.lower() in chunk_text.lower()) / len(tokens)


# ---------------------------------------------------------------------------
# Semantic deduplication (Self-Correcting RAG MMKP冗余惩罚简化版)
# ---------------------------------------------------------------------------


def deduplicate_by_similarity(
    chunks: List[Dict[str, object]], similarity_threshold: float = 0.85
) -> List[Dict[str, object]]:
    """Keep only the highest-scoring chunk per similarity cluster."""
    if len(chunks) <= 1:
        return chunks
    vectors = [c.get("embedding") for c in chunks]
    if any(v is None for v in vectors):
        return chunks

    sim = cosine_similarity(vectors)
    try:
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            metric="precomputed",
            linkage="average",
        ).fit_predict(1 - sim)
    except ValueError:
        return chunks

    selected: Dict[int, Dict[str, object]] = {}
    for chunk, label in zip(chunks, labels):
        label = int(label)
        if label not in selected or chunk["score"] > selected[label]["score"]:
            selected[label] = chunk
    return sorted(selected.values(), key=lambda x: x["score"], reverse=True)


# ---------------------------------------------------------------------------
# Context expansion (A-RAG chunk_read)
# ---------------------------------------------------------------------------


def expand_context(
    retrieved_chunks: List[Dict[str, object]], window: int = EXPANSION_WINDOW
) -> List[Dict[str, object]]:
    """Append neighbouring chunks (±window positions) to retrieved results."""
    if window <= 0:
        return retrieved_chunks
    index = load_index()
    all_chunks = index["chunks"]
    pos = {c["id"]: i for i, c in enumerate(all_chunks)}

    neighbour_ids: set = set()
    for chunk in retrieved_chunks:
        idx = pos.get(chunk["id"])
        if idx is None:
            continue
        for offset in range(-window, window + 1):
            j = idx + offset
            if 0 <= j < len(all_chunks):
                neighbour_ids.add(all_chunks[j]["id"])

    original_ids = {c["id"] for c in retrieved_chunks}
    neighbours = [
        {
            "id": c["id"],
            "content": c["content"],
            "source": c["source"],
            "page": c.get("page"),
            "score": 0.0,
            "embedding_score": 0.0,
            "lexical_score": 0.0,
            "entity_score": 0.0,
            "retrieval_mode": "expansion",
            "embedding": c.get("embedding"),
        }
        for c in all_chunks
        if c["id"] in neighbour_ids and c["id"] not in original_ids
    ]
    return retrieved_chunks + neighbours


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    question: str,
    top_k: int = TOP_K,
    mode: str = DEFAULT_RETRIEVAL_MODE,
) -> List[Dict[str, object]]:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"Unsupported retrieval mode: {mode}. Expected one of {RETRIEVAL_MODES}"
        )

    index = load_index()
    chunks = index["chunks"]
    texts = [chunk["content"] for chunk in chunks]

    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    question_embedding = embedder.embed_query(question)
    matrix = [chunk["embedding"] for chunk in chunks]
    embedding_scores = cosine_similarity([question_embedding], matrix)[0]

    # 中文专业词经常是短词或缩写，字符 n-gram 比依赖分词更省心。
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(texts)
    lexical_scores = cosine_similarity(vectorizer.transform([question]), tfidf_matrix)[0]

    normalized_embedding_scores = min_max_normalize(embedding_scores)
    normalized_lexical_scores = min_max_normalize(lexical_scores)

    if mode == "vector":
        # 纯向量检索作为基线，方便和 hybrid 做严格对照。
        scores = [float(s) for s in embedding_scores]
        entity_scores_list = [0.0] * len(chunks)
    else:
        # 当前 full30 用三路融合：向量看语义，TF-IDF 和实体项负责拉住术语。
        entity_scores_raw = [entity_match_score(question, t) for t in texts]
        entity_scores_norm = min_max_normalize(entity_scores_raw) if max(entity_scores_raw) > 0 else entity_scores_raw
        entity_scores_list = entity_scores_norm
        scores = [
            EMBEDDING_SCORE_WEIGHT_V3 * e + LEXICAL_SCORE_WEIGHT_V3 * l + ENTITY_SCORE_WEIGHT * n
            for e, l, n in zip(
                normalized_embedding_scores, normalized_lexical_scores, entity_scores_norm
            )
        ]

    ranked = sorted(
        (
            {
                "id": chunk["id"],
                "content": chunk["content"],
                "source": chunk["source"],
                "page": chunk.get("page"),
                "score": float(score),
                "embedding_score": float(embedding_score),
                "lexical_score": float(lexical_score),
                "entity_score": float(entity_score),
                "retrieval_mode": mode,
                "embedding": chunk["embedding"],  # retained for dedup; stripped in sources output
            }
            for chunk, score, embedding_score, lexical_score, entity_score in zip(
                chunks, scores, embedding_scores, lexical_scores, entity_scores_list
            )
        ),
        key=lambda item: item["score"],
        reverse=True,
    )

    # 先多取一些再去重，避免 Top-K 里塞进几段内容几乎一样的片段。
    candidates = deduplicate_by_similarity(ranked[: top_k * 2])
    return candidates[:top_k]


def format_context(chunks: List[Dict[str, object]]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        page = f"，页码：{chunk['page']}" if chunk.get("page") else ""
        parts.append(
            f"[片段{index}] 文件：{chunk['source']}{page}，综合分：{chunk['score']:.4f}\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


def _sources_from_chunks(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Strip internal-only fields (embedding) before returning to callers."""
    # embedding 太长，返回给 Web 和 CSV 只会让结果文件变得很难看。
    return [
        {
            "id": chunk["id"],
            "source": chunk["source"],
            "page": chunk.get("page"),
            "score": chunk["score"],
            "embedding_score": chunk.get("embedding_score", 0.0),
            "lexical_score": chunk.get("lexical_score", 0.0),
            "entity_score": chunk.get("entity_score", 0.0),
            "content": chunk["content"][:500],
        }
        for chunk in chunks
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_with_ollama(
    prompt: str,
    model: str = LLM_MODEL,
    num_predict: Optional[int] = None,
    timeout: Optional[int] = None,
) -> str:
    effective_num_predict = NUM_PREDICT if num_predict is None else num_predict
    effective_timeout = GENERATE_TIMEOUT if timeout is None else timeout
    # 不使用 stream，批量评测时拿完整响应更好写入 CSV。
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": effective_num_predict,
                "num_ctx": NUM_CTX,
            },
        },
        timeout=effective_timeout,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# Faithfulness check (Self-Correcting RAG NLI / Self-RAG [IsSup])
# ---------------------------------------------------------------------------


def check_faithfulness(
    answer: str,
    context: str,
    model: str = LLM_MODEL,
    num_predict: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, object]:
    """LLM-based faithfulness self-check; returns check_result and 0-1 score."""
    check = generate_with_ollama(
        FAITHFULNESS_CHECK_PROMPT.format(context=context, answer=answer),
        model=model,
        num_predict=num_predict,
        timeout=timeout,
    )
    m = re.search(r"整体忠实度[：:]\s*(\d+)", check)
    score = int(m.group(1)) / 10.0 if m else None
    return {"check_result": check, "faithfulness_score": score}


# ---------------------------------------------------------------------------
# Standard answer (baseline / stepwise / reflective prompt selectable)
# ---------------------------------------------------------------------------


def answer_question(
    question: str,
    top_k: int = TOP_K,
    model: str = LLM_MODEL,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    prompt_variant: str = "stepwise",
    num_predict: Optional[int] = None,
    timeout: Optional[int] = None,
    expand_context_enabled: Optional[bool] = None,
) -> Dict[str, object]:
    """Single-round QA with stepwise prompt by default.

    prompt_variant: "baseline" | "stepwise" | "reflective"
    retrieval_mode: "vector" | "hybrid" — for "adaptive" use answer_question_adaptive().
    """
    started = time.perf_counter()
    retrieved = retrieve(question, top_k=top_k, mode=retrieval_mode)
    retrieve_time = time.perf_counter() - started

    # 日常问答可以补相邻片段；控制实验时可关掉，避免变量混在一起。
    should_expand = CONTEXT_EXPANSION if expand_context_enabled is None else expand_context_enabled
    if should_expand:
        retrieved = expand_context(retrieved, window=EXPANSION_WINDOW)

    context = format_context(retrieved)
    template = {
        "baseline": PROMPT_TEMPLATE,
        "stepwise": PROMPT_TEMPLATE_STEPWISE,
        "reflective": PROMPT_TEMPLATE_REFLECTIVE,
    }.get(prompt_variant, PROMPT_TEMPLATE_STEPWISE)

    generate_started = time.perf_counter()
    answer = generate_with_ollama(
        template.format(context=context, question=question),
        model=model,
        num_predict=num_predict,
        timeout=timeout,
    )
    generate_time = time.perf_counter() - generate_started

    result: Dict[str, object] = {
        "question": question,
        "answer": answer,
        "sources": _sources_from_chunks(retrieved),
        "top_k": top_k,
        "model": model,
        "retrieval_mode": retrieval_mode,
        "prompt_variant": prompt_variant,
        "embedding_model": EMBEDDING_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "retrieve_time": retrieve_time,
        "generate_time": generate_time,
        "total_time": time.perf_counter() - started,
    }

    if ENABLE_FAITHFULNESS_CHECK:
        result["faithfulness"] = check_faithfulness(answer, context, model=model, num_predict=num_predict, timeout=timeout)

    return result


# ---------------------------------------------------------------------------
# Adaptive multi-round answer (SIM-RAG / ReaLM-Retrieve / R-Search)
# ---------------------------------------------------------------------------


def answer_question_adaptive(
    question: str,
    top_k: int = TOP_K,
    model: str = LLM_MODEL,
    num_predict: Optional[int] = None,
    timeout: Optional[int] = None,
    expand_context_enabled: Optional[bool] = None,
) -> Dict[str, object]:
    """Two-round retrieval: expand to ADAPTIVE_TOP_K_SECOND if first round is insufficient."""
    started = time.perf_counter()

    # 第一轮先用普通 hybrid。只有判断“不足”时才扩大检索范围。
    retrieved = retrieve(question, top_k=top_k, mode="hybrid")
    context = format_context(retrieved)

    sufficiency = generate_with_ollama(
        SUFFICIENCY_PROMPT.format(context=context, question=question),
        model=model,
        num_predict=16,
        timeout=timeout,
    ).strip()
    retrieve_time = time.perf_counter() - started

    rounds = 1
    if "不足" in sufficiency:
        retrieved = retrieve(question, top_k=ADAPTIVE_TOP_K_SECOND, mode="hybrid")
        rounds = 2

    # adaptive 默认也会补相邻片段，和普通问答保持同一套上下文处理。
    if CONTEXT_EXPANSION:
        retrieved = expand_context(retrieved, window=EXPANSION_WINDOW)

    # 扩展后再压回 top_k，避免 Prompt 被邻居片段撑得太长。
    retrieved = deduplicate_by_similarity(retrieved)[:top_k]
    context = format_context(retrieved)

    generate_started = time.perf_counter()
    answer = generate_with_ollama(
        PROMPT_TEMPLATE_STEPWISE.format(context=context, question=question),
        model=model,
        num_predict=num_predict,
        timeout=timeout,
    )
    generate_time = time.perf_counter() - generate_started

    result: Dict[str, object] = {
        "question": question,
        "answer": answer,
        "sources": _sources_from_chunks(retrieved),
        "top_k": top_k,
        "model": model,
        "retrieval_mode": "adaptive",
        "prompt_variant": "stepwise",
        "retrieval_rounds": rounds,
        "sufficiency_judgment": sufficiency,
        "embedding_model": EMBEDDING_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "retrieve_time": retrieve_time,
        "generate_time": generate_time,
        "total_time": time.perf_counter() - started,
    }

    if ENABLE_FAITHFULNESS_CHECK:
        result["faithfulness"] = check_faithfulness(answer, context, model=model, num_predict=num_predict, timeout=timeout)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="命令行 RAG 问答")
    parser.add_argument("--question", "-q", help="要提问的问题")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--mode", choices=ANSWER_MODES, default=DEFAULT_RETRIEVAL_MODE)
    parser.add_argument(
        "--prompt",
        choices=["baseline", "stepwise", "reflective"],
        default="stepwise",
        help="Prompt 策略",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--no-context-expansion", action="store_true")
    args = parser.parse_args()

    question = args.question or input("请输入问题：").strip()

    if args.mode == "adaptive":
        result = answer_question_adaptive(
            question,
            top_k=args.top_k,
            num_predict=args.num_predict,
            timeout=args.timeout,
            expand_context_enabled=not args.no_context_expansion,
        )
    else:
        result = answer_question(
            question,
            top_k=args.top_k,
            retrieval_mode=args.mode,
            prompt_variant=args.prompt,
            num_predict=args.num_predict,
            timeout=args.timeout,
            expand_context_enabled=not args.no_context_expansion,
        )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("\n模型回答：")
    print(result["answer"])
    print("\n来源片段：")
    for source in result["sources"]:
        page = f"，页码：{source['page']}" if source.get("page") else ""
        print(
            f"- {source['source']}{page}  综合分：{source['score']:.4f}  "
            f"向量：{source['embedding_score']:.4f}  词汇：{source['lexical_score']:.4f}  "
            f"实体：{source['entity_score']:.4f}"
        )
    rounds = result.get("retrieval_rounds", 1)
    print(
        f"\n耗时：检索 {result['retrieve_time']:.2f}s  生成 {result['generate_time']:.2f}s  "
        f"总计 {result['total_time']:.2f}s  检索轮数：{rounds}"
    )
    if "faithfulness" in result and result["faithfulness"]:
        score = result["faithfulness"].get("faithfulness_score")
        print(f"忠实度评分：{score}")


if __name__ == "__main__":
    main()

