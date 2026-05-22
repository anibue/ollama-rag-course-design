import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import requests
from langchain_ollama import OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DEFAULT_RETRIEVAL_MODE,
    EMBEDDING_MODEL,
    EMBEDDING_SCORE_WEIGHT,
    GENERATE_TIMEOUT,
    INDEX_FILE,
    LEXICAL_SCORE_WEIGHT,
    LLM_MODEL,
    NUM_CTX,
    NUM_PREDICT,
    OLLAMA_BASE_URL,
    RETRIEVAL_MODES,
    TEMPERATURE,
    TOP_K,
)


PROMPT_TEMPLATE = """你是一个严谨的计算机专业文档问答助手。
请只依据【参考资料】回答【用户问题】。如果参考资料没有明确依据，请回答“知识库中没有找到明确答案”，不要补充资料外事实。

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 先用一段话直接回答问题；
2. 再分点说明关键概念、步骤或原因；
3. 最后列出依据来源，引用片段编号和文件名。
"""


def load_index(path: Path = INDEX_FILE) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"知识库索引不存在：{path}。请先运行 python src/build_kb.py")
    return json.loads(path.read_text(encoding="utf-8"))


def min_max_normalize(values) -> List[float]:
    low = float(min(values))
    high = float(max(values))
    if high == low:
        return [0.0 for _ in values]
    return [(float(value) - low) / (high - low) for value in values]


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

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(texts)
    lexical_scores = cosine_similarity(vectorizer.transform([question]), tfidf_matrix)[0]

    normalized_embedding_scores = min_max_normalize(embedding_scores)
    normalized_lexical_scores = min_max_normalize(lexical_scores)
    if mode == "vector":
        scores = [float(score) for score in embedding_scores]
    else:
        scores = [
            EMBEDDING_SCORE_WEIGHT * embedding_score
            + LEXICAL_SCORE_WEIGHT * lexical_score
            for embedding_score, lexical_score in zip(
                normalized_embedding_scores, normalized_lexical_scores
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
                "retrieval_mode": mode,
            }
            for chunk, score, embedding_score, lexical_score in zip(
                chunks, scores, embedding_scores, lexical_scores
            )
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return ranked[:top_k]


def format_context(chunks: List[Dict[str, object]]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        page = f"，页码：{chunk['page']}" if chunk.get("page") else ""
        parts.append(
            f"[片段{index}] 文件：{chunk['source']}{page}，综合分：{chunk['score']:.4f}\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


def generate_with_ollama(prompt: str, model: str = LLM_MODEL) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT,
                "num_ctx": NUM_CTX,
            },
        },
        timeout=GENERATE_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "").strip()


def answer_question(
    question: str,
    top_k: int = TOP_K,
    model: str = LLM_MODEL,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> Dict[str, object]:
    started = time.perf_counter()
    retrieved = retrieve(question, top_k=top_k, mode=retrieval_mode)
    retrieve_time = time.perf_counter() - started

    prompt = PROMPT_TEMPLATE.format(context=format_context(retrieved), question=question)

    generate_started = time.perf_counter()
    answer = generate_with_ollama(prompt, model=model)
    generate_time = time.perf_counter() - generate_started

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "page": chunk.get("page"),
                "score": chunk["score"],
                "embedding_score": chunk["embedding_score"],
                "lexical_score": chunk["lexical_score"],
                "content": chunk["content"][:500],
            }
            for chunk in retrieved
        ],
        "top_k": top_k,
        "model": model,
        "retrieval_mode": retrieval_mode,
        "embedding_model": EMBEDDING_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "retrieve_time": retrieve_time,
        "generate_time": generate_time,
        "total_time": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="命令行 RAG 问答")
    parser.add_argument("--question", "-q", help="要提问的问题")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="检索片段数量")
    parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODES,
        default=DEFAULT_RETRIEVAL_MODE,
        help="检索策略：vector 或 hybrid",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    args = parser.parse_args()

    question = args.question or input("请输入问题：").strip()
    result = answer_question(question, top_k=args.top_k, retrieval_mode=args.mode)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("\n模型回答：")
    print(result["answer"])
    print("\n来源片段：")
    for source in result["sources"]:
        page = f"，页码：{source['page']}" if source.get("page") else ""
        print(
            f"- {source['source']}{page}，综合分：{source['score']:.4f}，"
            f"向量分：{source['embedding_score']:.4f}，关键词分：{source['lexical_score']:.4f}"
        )
    print(f"\n耗时：检索 {result['retrieve_time']:.2f}s，生成 {result['generate_time']:.2f}s")


if __name__ == "__main__":
    main()
