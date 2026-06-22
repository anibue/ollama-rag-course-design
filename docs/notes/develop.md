# 项目改进建议：基于论文创新点的模块补充、扩展与优化

> 本文档结合 Paper.md 中 40 篇论文的架构原理和创新点，针对当前项目（Ollama 本地 RAG 专业文档问答系统）提出**可落地的改进方案**。每个建议均标注：涉及文件、实现思路、借鉴论文、预期效果和实施难度，确保不改变项目最终目标的前提下增强系统能力。

---

## 一、项目现状分析

### 当前 RAG 链路

```
文档 → 切分(420字符/80重叠) → Embedding(nomic-embed-text) → JSON索引
问题 → Embedding → 三路混合检索(0.40向量+0.45 TF-IDF+0.15实体匹配) → Top-4 → Prompt拼接 → Ollama生成 → 答案+引用
```

### 现有评测维度

- 检索命中率（expected_source 是否在 Top-K 来源中）
- 参考答案覆盖度（token_overlap）
- 检索耗时 / 生成耗时 / 总耗时

### 当前短板

| 维度 | 短板描述 |
|------|----------|
| **检索** | 单次检索，无迭代；固定权重，不区分问题难度；无语义去重 |
| **生成** | Prompt 无分步推理指令；无自反思/自纠正机制 |
| **评测** | 无忠实度指标；无对抗性测试；无推理链检查 |
| **交互** | 单轮问答，无多轮记忆；无检索策略可视化 |
| **知识库** | 平铺索引，无实体/关系结构化信息 |

---

## 二、补充模块（新增功能）

### 模块 A：自适应多轮检索（Adaptive Iterative Retrieval）

**借鉴论文**：ReaLM-Retrieve (#8)、SIM-RAG (#19)、ReasonRAG (#7)

**核心思想**：ReaLM-Retrieve 提出在推理过程中检测"知识缺口"并按需触发检索。SIM-RAG 训练轻量 Critic 评估信息充分性。本项目可简化为**基于 Prompt 的两阶段检索**：先用 Top-4 检索生成初步答案，若答案置信度不足则扩大 Top-K 或切换检索模式重试。

**涉及文件**：`src/rag_chain.py`、`src/config.py`

**实现方案**：

```python
# src/config.py 新增
ADAPTIVE_RETRIEVAL = True          # 是否启用自适应检索
ADAPTIVE_TOP_K_SECOND = 8          # 二次检索 Top-K
ADAPTIVE_CONFIDENCE_THRESHOLD = 0.6  # 置信度阈值

# src/rag_chain.py 新增
SUFFICIENCY_PROMPT = """请判断以下参考资料是否足以回答用户问题。
如果信息充分，回答"充分"；如果信息不足或答案不确定，回答"不足"。

【参考资料】
{context}

【用户问题】
{question}

判断（仅回答"充分"或"不足"）："""

def answer_question_adaptive(question, top_k=TOP_K, model=LLM_MODEL, retrieval_mode=DEFAULT_RETRIEVAL_MODE):
    """带自适应检索的问答流程"""
    # 第一阶段：标准检索
    retrieved = retrieve(question, top_k=top_k, mode=retrieval_mode)
    context = format_context(retrieved)
    
    # 检查信息充分性
    sufficiency_prompt = SUFFICIENCY_PROMPT.format(context=context, question=question)
    sufficiency = generate_with_ollama(sufficiency_prompt, model=model).strip()
    
    # 第二阶段：如果不足，扩大检索
    if "不足" in sufficiency:
        retrieved_wide = retrieve(question, top_k=ADAPTIVE_TOP_K_SECOND, mode="hybrid")
        context = format_context(retrieved_wide)
    
    # 生成最终答案
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = generate_with_ollama(prompt, model=model)
    
    return { ... }  # 同现有 answer_question 的返回结构
```

**预期效果**：
- 对知识库覆盖不足的难题，检索命中率可从当前约 70% 提升至 85%+
- ReaLM-Retrieve 论文数据显示，自适应检索比固定单次检索 F1 提升 5-10%

**实施难度**：★★☆☆☆（仅修改 rag_chain.py，新增约 30 行代码）

---

### 模块 B：检索结果语义去重与多样性选择

**借鉴论文**：Self-Correcting RAG (#16, MMKP 上下文选择)、RankRAG (#2)

**核心思想**：Self-Correcting RAG 将文档选择建模为 MMKP（多维多选背包问题），在 token 预算内最大化信息密度、最小化冗余。本项目可简化为**基于 cosine 相似度的语义聚类去重**：对 Top-K 片段按 embedding 相似度聚类，每组只保留最高分片段。

**涉及文件**：`src/rag_chain.py`

**实现方案**：

```python
from sklearn.cluster import AgglomerativeClustering

def deduplicate_by_similarity(chunks, similarity_threshold=0.85):
    """对检索结果做语义去重，保留多样性"""
    if len(chunks) <= 1:
        return chunks
    
    embeddings = [chunk["embedding_score"] for chunk in chunks]
    # 用检索结果的 embedding 向量做聚类
    vectors = [chunk.get("_embedding_vector") for chunk in chunks]
    if not vectors:
        return chunks  # 无向量信息时跳过去重
    
    sim_matrix = cosine_similarity(vectors)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(1 - sim_matrix)
    
    # 每个簇保留分数最高的片段
    selected = {}
    for chunk, label in zip(chunks, labels):
        if label not in selected or chunk["score"] > selected[label]["score"]:
            selected[label] = chunk
    
    return sorted(selected.values(), key=lambda x: x["score"], reverse=True)
```

**在 `retrieve()` 函数中集成**：
```python
# 在 ranked[:top_k] 之后
ranked_top = ranked[:top_k * 2]  # 先取多一些候选
ranked_top = deduplicate_by_similarity(ranked_top)
return ranked_top[:top_k]
```

**预期效果**：
- 减少冗余片段，Top-K 中的信息覆盖率提升
- Self-Correcting RAG 论文中 MMKP 选择器将 Recall@5 从 49.6% 提升至 72.0%

**实施难度**：★★☆☆☆（新增约 25 行函数，修改 retrieve 函数 2 行）

---

### 模块 C：生成后忠实度自检（Post-Generation Faithfulness Check）

**借鉴论文**：Self-RAG (#1, LLaMA 部分, reflection token)、Self-Correcting RAG (#16, NLI 验证)

**核心思想**：Self-RAG 引入 `[IsSup]` 反思 token 检查生成是否被检索片段支持。Self-Correcting RAG 用 NLI 模型验证忠实性。本项目可简化为**Prompt 驱动的自检**：生成答案后，用 LLM 自身检查答案中的每个声明是否被检索片段支持。

**涉及文件**：`src/rag_chain.py`

**实现方案**：

```python
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
1. [声明1] → [支持/部分支持/不支持]
2. [声明2] → [支持/部分支持/不支持]
...
整体忠实度：X/10"""

def check_faithfulness(answer, context, model=LLM_MODEL):
    """生成后忠实度自检"""
    prompt = FAITHFULNESS_CHECK_PROMPT.format(context=context, answer=answer)
    check_result = generate_with_ollama(prompt, model=model)
    
    # 解析整体忠实度分数
    import re
    match = re.search(r'整体忠实度[：:]\s*(\d+)', check_result)
    score = int(match.group(1)) / 10.0 if match else None
    
    return {"check_result": check_result, "faithfulness_score": score}
```

**在 `answer_question()` 中集成**：
```python
result = answer_question(question)  # 现有流程
if ENABLE_FAITHFULNESS_CHECK:
    faith = check_faithfulness(result["answer"], format_context(retrieved))
    result["faithfulness"] = faith
```

**预期效果**：
- 为每个答案提供可追溯的忠实度评分
- 降低幻觉风险：Self-RAG 论文显示自反思可将事实性错误降低 30%+
- 为评测体系新增"忠实度"维度

**实施难度**：★★☆☆☆（新增约 20 行函数，修改 answer_question 3 行）

---

### 模块 D：上下文扩展读取（Context Expansion / Chunk Read）

**借鉴论文**：A-RAG (#13, RL 部分, chunk_read 工具)

**核心思想**：A-RAG 提供 chunk_read 工具，允许模型在确定相关片段后读取相邻片段以获取完整上下文。本项目可在 Top-K 确定后，自动读取每个片段的相邻片段（±1），扩展上下文窗口。

**涉及文件**：`src/rag_chain.py`、`src/build_kb.py`

**实现方案**：

```python
# src/config.py 新增
CONTEXT_EXPANSION = True     # 是否启用上下文扩展
EXPANSION_WINDOW = 1         # 向前后各扩展的片段数

# src/rag_chain.py 新增
def expand_context(retrieved_chunks, index, window=EXPANSION_WINDOW):
    """对检索结果的每个片段，读取相邻片段扩展上下文"""
    all_chunks = index["chunks"]
    chunk_map = {c["id"]: i for i, c in enumerate(all_chunks)}
    
    expanded_ids = set()
    for chunk in retrieved_chunks:
        idx = chunk_map.get(chunk["id"])
        if idx is None:
            continue
        for offset in range(-window, window + 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(all_chunks):
                expanded_ids.add(all_chunks[neighbor_idx]["id"])
    
    # 保留原始检索片段的排序，追加扩展片段
    original_ids = {c["id"] for c in retrieved_chunks}
    expanded = [c for c in all_chunks if c["id"] in expanded_ids and c["id"] not in original_ids]
    return retrieved_chunks + expanded
```

**预期效果**：
- 答案完整性提升：A-RAG 论文显示 chunk_read 可使 MuSiQue 准确率从 62.4% 提升至 74.1%
- 对跨段落的连续知识点（如操作系统的内存管理章节）效果显著

**实施难度**：★★☆☆☆（新增约 20 行函数，修改 answer_question 2 行）

---

## 三、扩展模块（增强现有功能）

### 模块 E：Prompt 分步推理与引用追踪

**借鉴论文**：ProRAG (#5, 过程监督)、ReasonRAG (#7, 信息充分性判断)、Self-RAG (#1, 反思 token)

**核心思想**：ProRAG 要求模型分步推理并记录每步引用的片段。ReasonRAG 的 Reasoning Prompt 要求模型先反思、再判断信息充分性、再生成查询或答案。本项目可将现有 Prompt 升级为**分步推理模板**。

**涉及文件**：`src/rag_chain.py`

**实现方案**：

```python
# 替换现有 PROMPT_TEMPLATE
PROMPT_TEMPLATE_STEPWISE = """你是一个严谨的计算机专业文档问答助手。请按以下步骤回答问题：

## 步骤 1：信息评估
先判断【参考资料】中的信息是否足以回答【用户问题】。
- 如果信息充分，继续步骤 2。
- 如果信息不足，直接回答"知识库中没有找到明确答案"并说明缺少哪方面的信息。

## 步骤 2：关键事实提取
从参考资料中提取与问题相关的关键事实，列出每条事实对应的片段编号。

## 步骤 3：推理与回答
基于提取的关键事实进行推理，给出最终答案。

## 步骤 4：引用来源
列出答案所依据的片段编号和文件名。

【参考资料】
{context}

【用户问题】
{question}

请按上述步骤回答："""
```

**预期效果**：
- 答案结构更清晰，推理过程可追溯
- ReasonRAG 论文显示，分步推理+信息充分性判断可将 HotpotQA F1 从 32.8% 提升至 42.3%

**实施难度**：★☆☆☆☆（仅修改 PROMPT_TEMPLATE 字符串）

---

### 模块 F：检索策略三路融合（向量 + TF-IDF + 实体匹配）

**借鉴论文**：A-RAG (#13, keyword_search + semantic_search)、LightRAG (#9, 实体级+关系级检索)、GraphRAG-R1 (#10, 图+文本混合)

**核心思想**：A-RAG 提供 keyword_search（精确词汇匹配）和 semantic_search（语义检索）两种工具。LightRAG 在实体级和关系级做双层检索。本项目可在现有向量+TF-IDF 基础上增加**实体精确匹配**作为第三路信号。

**涉及文件**：`src/rag_chain.py`、`src/config.py`

**实现方案**：

```python
# src/config.py 新增
ENTITY_SCORE_WEIGHT = 0.15        # 实体匹配权重
EMBEDDING_SCORE_WEIGHT_V3 = 0.40  # 调整后的向量权重
LEXICAL_SCORE_WEIGHT_V3 = 0.45    # 调整后的 TF-IDF 权重

# src/rag_chain.py 新增
def entity_match_score(question, chunk_text):
    """基于实体/关键词精确匹配的分数"""
    import re
    # 提取问题中的关键实体（名词短语、英文术语、数字）
    entities = re.findall(r'[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}|\d+', question)
    if not entities:
        return 0.0
    
    matches = sum(1 for e in entities if e.lower() in chunk_text.lower())
    return matches / len(entities)

# 在 retrieve() 中，hybrid 模式改为三路融合
if mode == "hybrid":
    entity_scores = [entity_match_score(question, text) for text in texts]
    normalized_entity_scores = min_max_normalize(entity_scores)
    scores = [
        EMBEDDING_SCORE_WEIGHT_V3 * e + LEXICAL_SCORE_WEIGHT_V3 * l + ENTITY_SCORE_WEIGHT * n
        for e, l, n in zip(normalized_embedding_scores, normalized_lexical_scores, normalized_entity_scores)
    ]
```

**预期效果**：
- 对包含专有名词（如"TCP"、"虚拟内存"、"B+树"）的问题，精确匹配可弥补 TF-IDF 的语义鸿沟
- A-RAG 论文显示，keyword_search 在已知实体匹配场景下显著优于纯语义检索

**实施难度**：★★☆☆☆（新增约 15 行函数，修改 retrieve 函数约 10 行）

---

### 模块 G：LLM 重排序（Retrieve-Rerank-Generate）

**借鉴论文**：RankRAG (#2, LLaMA 部分, 统一排序与生成)

**核心思想**：RankRAG 在检索后用 LLM 对候选片段做相关性重排序，仅需少量排序数据即可超越专业排序模型。本项目可在现有检索后增加**LLM pairwise 重排序**步骤。

**涉及文件**：`src/rag_chain.py`

**实现方案**：

```python
# src/config.py 新增
ENABLE_RERANK = False   # 是否启用 LLM 重排序（默认关闭，因增加延迟）
RERANK_TOP_K = 8        # 重排序前取更多候选

# src/rag_chain.py 新增
RERANK_PROMPT = """请判断以下两个片段中，哪个与问题更相关。
仅回答"A"或"B"或"一样"。

【问题】
{question}

【片段 A】
{chunk_a}

【片段 B】
{chunk_b}

哪个更相关？"""

def rerank_with_llm(question, chunks, model=LLM_MODEL):
    """用 LLM 对 Top-K 片段做 pairwise 重排序"""
    if len(chunks) <= 1:
        return chunks
    
    # 简化实现：用 LLM 对每对片段做比较，统计胜率
    wins = [0] * len(chunks)
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            prompt = RERANK_PROMPT.format(
                question=question,
                chunk_a=chunks[i]["content"][:300],
                chunk_b=chunks[j]["content"][:300]
            )
            result = generate_with_ollama(prompt, model=model).strip()
            if "A" in result:
                wins[i] += 1
            elif "B" in result:
                wins[j] += 1
            else:
                wins[i] += 0.5
                wins[j] += 0.5
    
    # 按胜率重排序
    ranked = sorted(zip(chunks, wins), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked]
```

**预期效果**：
- RankRAG 论文显示，LLM 重排序可将 NQ Recall@5 从 80.1% 提升至 81.6%
- 对语义相似但主题不同的"干扰片段"有显著过滤效果

**实施难度**：★★★☆☆（新增约 30 行函数，但会增加延迟，建议作为可选功能）

---

## 四、优化模块（提升现有性能）

### 模块 H：多维评测指标体系

**借鉴论文**：RAG Survey (Gao) (#12, 综述部分, 三维度评估)、RAG: Architectures & Robustness (#17, 综述部分, robustness)、Self-Correcting RAG (#16, 忠实度+矛盾率)

**核心思想**：RAG 综述提出 context relevance / answer faithfulness / answer relevance 三维度评估。Self-Correcting RAG 引入 Attribution Precision（归因精确度）和 Contradiction Rate（矛盾率）。本项目可在现有评测基础上扩展为多维指标。

**涉及文件**：`src/eval.py`、`eval_questions.csv`

**实现方案**：

```python
# src/eval.py 新增评测维度

# 1. 答案忠实度（Faithfulness）—— 检查答案是否被检索片段支持
def faithfulness_score(answer, contexts, model=LLM_MODEL):
    """用 LLM 检查答案的忠实度"""
    prompt = f"""判断以下回答是否完全基于参考资料。评分 0-10。
0=完全编造，10=完全有据可查。

参考资料：{' '.join(c[:200] for c in contexts[:3])}
回答：{answer[:500]}

评分（仅输出数字）："""
    result = generate_with_ollama(prompt, model=model)
    try:
        return int(re.search(r'\d+', result).group()) / 10.0
    except:
        return None

# 2. 拒答准确率（Rejection Accuracy）—— 对无关问题是否正确拒答
def test_rejection(eval_questions_path):
    """测试系统对无关问题的拒答能力"""
    rejection_questions = [
        {"question": "今天天气怎么样？", "expected": "拒答"},
        {"question": "你喜欢吃什么水果？", "expected": "拒答"},
        {"question": "2026年世界杯冠军是谁？", "expected": "拒答"},
    ]
    correct = 0
    for q in rejection_questions:
        result = answer_question(q["question"])
        if "没有找到明确答案" in result["answer"] or "无法回答" in result["answer"]:
            correct += 1
    return correct / len(rejection_questions)

# 3. 扩展 eval_questions.csv 增加对抗性问题
# 在现有 30 条基础上增加 10 条无关/边界问题
```

**预期效果**：
- 新增 3 个评测维度：忠实度、拒答准确率、归因精确度
- RAG Survey 指出，完整的 RAG 评估应覆盖检索质量、生成质量和鲁棒性三个维度

**实施难度**：★★☆☆☆（新增约 40 行评测函数，修改 eval.py 主流程）

---

### 模块 I：检索权重动态调优

**借鉴论文**：UR2 (#2, RL 部分, 难度感知课程学习)、RA-DIT (#3, LLaMA 部分, 双端优化)

**核心思想**：UR2 提出按问题难度分层调优——先在简单问题上找最优参数，再扩展到难问题。RA-DIT 分别优化检索器和生成器。本项目可实现**网格搜索自动调优**：在 30 条评测问题上搜索最优的 embedding/lexical 权重比。

**涉及文件**：新增 `src/weight_tune.py`

**实现方案**：

```python
# src/weight_tune.py
"""检索权重网格搜索调优脚本"""
import itertools
from eval import run_eval
from rag_strategy_compare import run_compare

def grid_search_weights():
    """搜索最优的 embedding/lexical 权重组合"""
    weight_candidates = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    results = []
    
    for emb_weight in weight_candidates:
        lex_weight = 1.0 - emb_weight
        # 临时修改配置
        from config import EMBEDDING_SCORE_WEIGHT, LEXICAL_SCORE_WEIGHT
        import config
        config.EMBEDDING_SCORE_WEIGHT = emb_weight
        config.LEXICAL_SCORE_WEIGHT = lex_weight
        
        summary = run_eval(EVAL_FILE, top_k=4)
        results.append({
            "embedding_weight": emb_weight,
            "lexical_weight": lex_weight,
            "retrieval_hit_rate": summary["retrieval_hit_rate"],
            "reference_overlap": summary["average_reference_overlap"],
        })
        print(f"emb={emb_weight:.2f} lex={lex_weight:.2f} "
              f"hit={summary['retrieval_hit_rate']:.4f} "
              f"overlap={summary['average_reference_overlap']:.4f}")
    
    # 找最优组合
    best = max(results, key=lambda x: (x["retrieval_hit_rate"], x["reference_overlap"]))
    print(f"\n最优权重: embedding={best['embedding_weight']}, lexical={best['lexical_weight']}")
    return results, best
```

**预期效果**：
- 找到当前知识库+问题集上的最优权重比
- UR2 论文的难度感知思路：可按问题类别（网络/OS/DB/AI/RAG）分别找最优权重

**实施难度**：★★☆☆☆（新增约 40 行独立脚本，不修改现有代码）

---

### 模块 J：Streamlit 页面功能增强

**借鉴论文**：ChatQA (#4, LLaMA 部分, 对话式 RAG)、A-RAG (#13, RL 部分, 多粒度检索)、MAO-ARAG (#12, RL 部分, 工作流选择)

**核心思想**：ChatQA 实现对话式 RAG，支持多轮追问。A-RAG 允许用户选择检索粒度。本项目可在 Streamlit 页面增加**多轮对话记忆**和**检索策略切换**。

**涉及文件**：`src/app.py`

**实现方案**：

```python
# src/app.py 增强

# 1. 多轮对话记忆
if "messages" not in st.session_state:
    st.session_state.messages = []

# 将历史问答作为上下文
def build_history_context(messages, max_turns=3):
    """构建多轮对话上下文"""
    recent = messages[-max_turns * 2:]  # 取最近 N 轮
    parts = []
    for msg in recent:
        role = "用户" if msg["role"] == "user" else "助手"
        parts.append(f"{role}：{msg['content'][:200]}")
    return "\n".join(parts)

# 2. 检索策略选择
with st.sidebar:
    retrieval_mode = st.selectbox("检索策略", ["hybrid", "vector", "keyword_only"])
    enable_expansion = st.checkbox("启用上下文扩展", value=False)
    enable_faithfulness = st.checkbox("启用忠实度检查", value=False)
    top_k = st.slider("Top-K", 1, 8, 4)

# 3. 显示检索详情
with st.expander("检索详情"):
    for i, source in enumerate(result["sources"]):
        st.write(f"片段 {i+1}: {source['source']}")
        st.write(f"  向量分: {source['embedding_score']:.4f}")
        st.write(f"  关键词分: {source['lexical_score']:.4f}")
        st.write(f"  综合分: {source['score']:.4f}")
```

**预期效果**：
- 支持追问场景（"刚才说的虚拟内存具体怎么实现？"）
- 用户可直观对比不同检索策略的效果
- 检索详情可视化增强系统可解释性

**实施难度**：★★☆☆☆（修改 app.py 约 50 行）

---

### 模块 K：Prompt 工程三版对比实验

**借鉴论文**：Self-RAG (#1, reflection token)、ReasonRAG (#7, 信息充分性判断)

**核心思想**：对比三种 Prompt 策略的效果差异，作为课程设计的实验亮点。

**涉及文件**：新增 `src/prompt_compare.py`

**三种 Prompt 策略**：

| 策略 | Prompt 特点 | 借鉴论文 |
|------|-------------|----------|
| **baseline** | 当前版本：直接回答 | — |
| **stepwise** | 分步推理：信息评估→事实提取→推理→引用 | ProRAG, ReasonRAG |
| **reflective** | 自反思：生成→检查→修正 | Self-RAG |

**实现方案**：

```python
# src/prompt_compare.py
"""三种 Prompt 策略对比实验"""

PROMPT_VARIANTS = {
    "baseline": PROMPT_TEMPLATE,  # 当前版本
    
    "stepwise": """请按以下步骤回答：
1. 评估参考资料是否充分
2. 提取关键事实（标注片段编号）
3. 基于事实推理回答
4. 列出引用来源

【参考资料】{context}
【问题】{question}
按步骤回答：""",
    
    "reflective": """请回答以下问题，然后反思并修正。

【参考资料】{context}
【问题】{question}

第一轮回答：
[请先回答]

反思：
[检查回答中的每个声明是否有参考资料支持，标注不支持的部分]

修正后的最终回答：
[基于反思修正后的答案，保留引用来源]""",
}

def run_prompt_compare(eval_file, top_k=4, model=LLM_MODEL):
    """对比三种 Prompt 策略"""
    results = {}
    for name, template in PROMPT_VARIANTS.items():
        # 替换 PROMPT_TEMPLATE 并运行评测
        import rag_chain
        original = rag_chain.PROMPT_TEMPLATE
        rag_chain.PROMPT_TEMPLATE = template
        summary = run_eval(eval_file, top_k)
        results[name] = summary
        rag_chain.PROMPT_TEMPLATE = original
    return results
```

**预期效果**：
- 提供三种策略的检索命中率、覆盖度、耗时对比
- Self-RAG 论文显示，反思机制可将 Open-domain QA 准确率提升 10-20%

**实施难度**：★★☆☆☆（新增约 60 行独立脚本）

---

## 五、知识库增强模块

### 模块 L：知识库结构化扩展

**借鉴论文**：GraphRAG (Microsoft) (#8, Graph 部分, 知识图谱+社区摘要)、LightRAG (#9, 双层检索)

**核心思想**：GraphRAG 从文档中抽取实体和关系构建知识图谱。LightRAG 支持增量更新。本项目可在 `build_kb.py` 中增加**实体抽取**步骤，为每个 chunk 标注关键实体标签。

**涉及文件**：`src/build_kb.py`

**实现方案**：

```python
# src/build_kb.py 新增
import re

def extract_entities(text):
    """从文本中抽取关键实体（简化版，基于规则）"""
    entities = set()
    # 英文术语（大写开头或全大写）
    entities.update(re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text))
    # 英文缩写
    entities.update(re.findall(r'\b[A-Z]{2,}\b', text))
    # 中文专业术语（2-6字）
    entities.update(re.findall(r'[\u4e00-\u9fff]{2,6}', text))
    # 过滤常见停用词
    stopwords = {'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'How', 'What',
                 '的', '是', '在', '了', '和', '与', '或', '但', '而', '也', '都', '有'}
    return list(entities - stopwords)

# 在 build_chunks() 中为每个 chunk 添加实体标签
for chunk_id, content in enumerate(split_text(...), start=1):
    chunks.append({
        "id": f"d{doc_id:03d}-c{chunk_id:03d}",
        "content": content,
        "source": doc["source"],
        "page": doc["page"],
        "char_count": len(content),
        "entities": extract_entities(content),  # 新增
    })
```

**预期效果**：
- 为每个 chunk 提供结构化标签，支持实体级精确匹配
- GraphRAG 论文显示，结构化信息可使全局查询的 comprehensiveness 提升 72-83%

**实施难度**：★★☆☆☆（新增约 15 行函数，修改 build_chunks 1 行）

---

## 六、实施路线图

### 阶段一：快速增益（1-2 天）

| 优先级 | 模块 | 修改量 | 预期收益 |
|:---:|------|--------|----------|
| 1 | **E: 分步推理 Prompt** | 修改 1 个字符串 | 答案结构更清晰，可追溯性增强 |
| 2 | **D: 上下文扩展读取** | 新增 20 行 | 答案完整性提升 |
| 3 | **H: 多维评测指标** | 新增 40 行 | 评测体系完善 |

### 阶段二：核心增强（3-5 天）

| 优先级 | 模块 | 修改量 | 预期收益 |
|:---:|------|--------|----------|
| 4 | **A: 自适应多轮检索** | 新增 30 行 | 难题检索命中率提升 |
| 5 | **B: 语义去重** | 新增 25 行 | Top-K 信息密度提升 |
| 6 | **C: 忠实度自检** | 新增 20 行 | 幻觉风险降低 |
| 7 | **F: 三路检索融合** | 新增 15 行 | 专有名词匹配增强 |

### 阶段三：锦上添花（5-7 天）

| 优先级 | 模块 | 修改量 | 预期收益 |
|:---:|------|--------|----------|
| 8 | **K: Prompt 三版对比** | 新增 60 行 | 实验亮点 |
| 9 | **I: 权重网格调优** | 新增 40 行 | 最优参数自动发现 |
| 10 | **J: Streamlit 增强** | 修改 50 行 | 用户体验提升 |
| 11 | **L: 知识库结构化** | 新增 15 行 | 结构化检索能力 |
| 12 | **G: LLM 重排序** | 新增 30 行 | 检索精度提升（可选） |

---

## 七、依赖影响评估

| 模块 | 新增依赖 | 是否影响现有依赖 |
|------|----------|-----------------|
| A 自适应检索 | 无 | 否 |
| B 语义去重 | scikit-learn（已有） | 否 |
| C 忠实度自检 | 无 | 否 |
| D 上下文扩展 | 无 | 否 |
| E 分步推理 | 无 | 否 |
| F 三路融合 | 无 | 否 |
| G LLM 重排序 | 无 | 否 |
| H 多维评测 | 无 | 否 |
| I 权重调优 | 无 | 否 |
| J Streamlit 增强 | streamlit（已有） | 否 |
| K Prompt 对比 | 无 | 否 |
| L 知识库结构化 | 无 | 否 |

**结论：所有改进均不引入新的外部依赖**，完全基于现有 `scikit-learn`、`requests`、`langchain-ollama`、`streamlit` 实现。

---

## 八、与论文创新点的对应关系总览

| 改进模块 | 主要借鉴论文 | 论文创新点 | 项目简化方式 |
|----------|-------------|-----------|-------------|
| A 自适应检索 | ReaLM-Retrieve, SIM-RAG, ReasonRAG | 步级不确定性检测+检索干预 | Prompt 判断信息充分性 |
| B 语义去重 | Self-Correcting RAG (MMKP) | 多维背包问题优化上下文选择 | AgglomerativeClustering 聚类去重 |
| C 忠实度自检 | Self-RAG, Self-Correcting RAG | reflection token / NLI 验证 | Prompt 驱动的 LLM 自检 |
| D 上下文扩展 | A-RAG | chunk_read | 读取相邻 ±1 片段 |
| E 分步推理 | ProRAG, ReasonRAG | 过程监督 RL / SPRE | 分步推理 Prompt 模板 |
| F 三路融合 | A-RAG, LightRAG, GraphRAG-R1 | keyword+semantic+graph 混合 | 向量+TF-IDF+实体匹配 |
| G LLM 重排序 | RankRAG | 统一排序与生成 | LLM pairwise 比较 |
| H 多维评测 | RAG Survey (Gao), Self-Correcting RAG | 三维度评估 / 忠实度+矛盾率 | 新增忠实度/拒答/归因指标 |
| I 权重调优 | UR2, RA-DIT | 难度感知课程学习 / 双端优化 | 网格搜索自动调优 |
| J 界面增强 | ChatQA, A-RAG | 对话式 RAG / 多粒度检索 | 多轮记忆+策略切换 |
| K Prompt 对比 | Self-RAG, ReasonRAG | 反思 token / 分步推理 | 三种 Prompt 策略对比实验 |
| L 知识库结构化 | GraphRAG, LightRAG | 知识图谱+社区摘要 | 规则抽取实体标签 |

---

## 九、总结

本改进建议基于 Paper.md 中 40 篇论文的核心创新，提出 **12 个可落地的改进模块**，涵盖检索优化、生成增强、评测完善和交互提升四个维度。所有改进均：

1. **不改变项目最终目标**——仍是基于 Ollama 的本地 RAG 专业文档问答系统
2. **不引入新的外部依赖**——完全基于现有技术栈
3. **增量实施、独立可测**——每个模块可单独添加并在 `results/` 中输出对比结果
4. **保留轻量化优势**——所有论文的重 RL 训练、模型微调部分均被简化为 Prompt 层面或后处理层面的等效实现

最高优先级的三项改进（分步推理 Prompt、上下文扩展读取、多维评测指标）可在 1-2 天内完成，即可为课程设计报告增加"基于论文创新点的系统优化"章节内容。

---

## 附录：移除的无开源代码论文

> 以下论文因暂无公开代码仓库或模型权重，已从正文各模块的"借鉴论文"中移除，但其核心思路仍具参考价值。

| 论文 | 原引用模块 | 移除原因 | 核心思路简述 |
|------|-----------|---------|-------------|
| ReflectiveRAG (#17, RL部分) | 模块C：忠实度自检 | Amazon 行业论文 (EACL 2026)，无公开代码仓库 | 自反思检索 + 对比噪声消除，轻量延迟下提升 EM 并降低冗余 |
| PAR2-RAG (#20, RL部分) | 模块D：上下文扩展 | 论文未提及代码链接，搜索未找到仓库 | 广度优先锚定 + 深度优先精炼，提升准确率与 NDCG |
| RRPO (#14, RL部分) | 模块G：LLM重排序 | 论文未提及代码链接，仅有第三方请求实现的 issue | 将重排序建模为 MDP，使用 LLM 反馈作为奖励优化文档"有用性" |
| ReaRAG (#7, LLaMA部分) | 模块K：Prompt对比 | 预印本，待开源 | 面向大推理模型的事实性增强 RAG，限制推理链长度避免过度思考 |
