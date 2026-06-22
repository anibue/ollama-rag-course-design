# Feature: 回答回写知识库（Answer Write-Back）

> 功能目标：将系统生成的高质量回答导入知识库，便于后续类似提问快速检索复用，实现 RAG 系统的自我增强。

---

## 一、功能设计

### 1.1 核心思路

传统 RAG 是**只读系统**：知识库在初始化时构建一次，之后模型只从中检索，生成的回答在返回用户后即被丢弃。

**Answer Write-Back** 打破这一单向模式，增加一条**反向通路**：

```
                    ┌─────────────────────────────────┐
                    │         知识库 (Knowledge Base)    │
                    │  chunks + embeddings + metadata   │
                    └──────────┬──────────────────┬────┘
                               │ 检索 (Read)       │ 回写 (Write)
                               ▼                  │
    用户提问 → Embedding → Hybrid Search → Top-K ─┤
                               │                  │
                               ▼                  │
                          Prompt → Ollama → 回答 ─┤
                               │                  │
                               ▼                  │
                        [验证层] ──── 通过？ ──→ 写入知识库
                               │
                               └── 不通过 → 丢弃
```

### 1.2 需要解决的核心问题

| 问题 | 风险 | 解决方案 |
|------|------|----------|
| **幻觉污染** | 错误回答写入后会被后续检索到，形成恶性循环 | 多阶段验证（Grounding + Attribution + Novelty） |
| **重复膨胀** | 相似回答大量堆积，知识库体积无控制增长 | 语义去重 + 容量上限 |
| **知识过时** | 写入的回答基于旧文档，文档更新后回答过时 | 来源版本追踪 + TTL 过期机制 |
| **检索干扰** | 回写内容质量低于原始文档，降低检索精度 | 质量分级标记 + 检索时可选排除 |

---

## 二、相关论文调研

### 2.1 Bidirectional RAG ⭐⭐⭐（最直接相关）

**论文**：*Bidirectional RAG: Safe Self-Improving Retrieval-Augmented Generation Through Multi-Stage Validation*（2024）

**核心贡献**：首个系统性提出"安全回写"的 RAG 架构。

**架构设计**：

```
生成的回答 y
    ↓
┌─ Gate 1: Grounding Verification (NLI 蕴含检测) ─┐
│  对 y 中每个句子 s，计算与检索文档的最大蕴含概率：  │
│  G(s) = max_{d ∈ D} P(entail | s, d)            │
│  要求 min_s G(s) ≥ θ_g (默认 0.7)               │
└─────────────────────────────────────────────────┘
    ↓ 通过
┌─ Gate 2: Attribution Check (来源归因) ──────────┐
│  回答中每个事实声明必须能追溯到具体检索片段        │
│  使用 citation 信息验证归因完整性                  │
└─────────────────────────────────────────────────┘
    ↓ 通过
┌─ Gate 3: Novelty Detection (新颖性检测) ────────┐
│  计算回答与现有知识库的最大语义相似度：            │
│  novel(y) = 1 - max_{d ∈ KB} sim(emb(y), emb(d))│
│  要求 novel(y) ≥ θ_n (避免近重复写入)            │
└─────────────────────────────────────────────────┘
    ↓ 通过
写入知识库 ✅
```

**关键实验结果**：
- 4 个数据集（NQ, TriviaQA, HotpotQA, Stack Overflow），12 组实验
- 覆盖率从 20.33%（Standard RAG）提升至 **40.58%**（近乎翻倍）
- 写入量仅 140 篇 vs 朴素回写的 500 篇（**减少 72%**）
- 拒绝约 72% 的候选写入，以保守策略保证知识库质量

**额外机制 — Experience Store**：
- 被拒绝的回答也会保存 **critique log**（拒绝原因）
- 查询时检索 critique，引导模型避免重复犯错（"负面学习"）

---

### 2.2 GroundedCache ⭐⭐⭐（缓存安全复用）

**论文**：*Grounded Cache Routing for Retrieval-Augmented Generation: When Is It Safe to Reuse an Answer?*（2025）

**核心贡献**：回答缓存的安全路由，4 个门控确保缓存复用不会返回错误答案。

**4 个安全门控**：

| 门控 | 检查内容 | 作用 |
|------|----------|------|
| G1: Query Similarity | 新查询与缓存查询的语义相似度 | 确保问题一致 |
| G2: Evidence Overlap | 新检索片段与缓存片段的 Jaccard 重叠 | 确保证据一致 |
| G3: Source Version | 共享 chunk 的源文档版本号 | 防止文档更新后缓存失效 |
| G4: Lexical Support | 缓存回答的内容 token 是否被新检索证据覆盖 | **最关键的安全门** |

**关键发现**：
- G4（词汇支持门控）是 **load-bearing** 安全机制：移除后 USR 上升 +0.125
- 完整系统在 HotpotQA 上将 Unsafe-Served Rate 降至 **0.0%**
- 延迟仅增加 4-7%（门控检查极轻量）

**与回写功能的关联**：回写时可参考 G4 思路，写入的回答需通过"证据支持验证"后才入库。

---

### 2.3 ROZA Graphs ⭐⭐（推理过程持久化）

**论文**：*ROZA Graphs: Self-Improving Near-Deterministic RAG through Evidence-Centric Feedback*（2025）

**核心贡献**：通过 **Reasoning Graph** 持久化每次推理中对每个证据片段的评判结果。

**架构**：
```
Reasoning Graph G_R
    ├── 节点：证据片段（chunk）
    ├── 边：对证据的评判（有用/无用/误导）
    └── 遍历：新查询时反向遍历，获取历史评判

Retrieval Graph G_P
    ├── 节点：候选检索项
    ├── 边：检索历史
    └── 用途：剪枝 consistently-rejected 候选项
```

**关键结果**：
- MuSiQue 上，50%+ 覆盖率时准确率提升 +10.6pp（47% 错误减少）
- 高复用场景下，成本降低 46%，延迟降低 46%
- 决策一致性提升 +8~21pp

**与回写功能的关联**：不仅回写回答，还可以回写**推理路径**——记录"哪些片段对回答有用"，为后续检索提供信号。

---

### 2.4 EvoRAG ⭐⭐（知识图谱自我进化）

**论文**：*EvoRAG: Making Knowledge Graph-based RAG Automatically Evolve through Feedback-driven Backpropagation*（2025）

**核心贡献**：将用户反馈通过 **backpropagation** 传播到知识图谱的三元组级别，持续优化检索。

**机制**：
```
用户反馈（正确/错误）
    ↓
Path Evaluation（评估每条推理路径的贡献度）
    ↓
Gradient Backpropagation（将路径贡献度传播到三元组）
    ↓
更新三元组的 contribution score
    ↓
后续检索优先选择高贡献三元组
```

**关键结果**：
- 比 SOTA KG-RAG 准确率提升 7.34%
- GitHub: `iDC-NEU/EvoRAG`

**与回写功能的关联**：回写时可附带 **utility score**，后续检索时对高 utility 的回答赋予更高权重。

---

### 2.5 Semantic Caching 工业方案

**GPTCache**：开源语义缓存库
- 查询 → 嵌入 → 在缓存中搜索相似查询 → 命中则直接返回缓存回答
- 相似度阈值通常 0.85-0.95

**AWS ElastiCache Semantic Cache**：
- 使用向量数据库（Valkey/Redis）存储 (query_embedding, answer) 对
- 命中时延迟从秒级降至毫秒级，成本降低 86%

**Proximity Cache**（LSH 方案）：
- 使用 Locality-Sensitive Hashing 加速缓存查找
- 数据库调用减少 77.2%，查找延迟恒定 4.8μs

---

## 三、实现方案

### 3.1 方案选择

综合考虑项目约束（Ollama 本地部署、无新依赖、增量实施），推荐 **方案 A**：

| 方案 | 复杂度 | 安全性 | 延迟开销 | 推荐度 |
|------|--------|--------|----------|--------|
| **A: 验证后回写** | 中 | 高 | ~2s/回答 | ⭐⭐⭐ |
| B: 语义缓存 | 低 | 中 | ~0.1s | ⭐⭐ |
| C: 推理图持久化 | 高 | 高 | ~1s | ⭐ |

**方案 A** 借鉴 Bidirectional RAG，实现完整的多阶段验证 + 知识库回写。

### 3.2 详细设计

#### 新增文件：`src/answer_writeback.py`

```python
"""
Answer Write-Back: 将验证通过的回答写入知识库
借鉴: Bidirectional RAG (2024), GroundedCache (2025)
"""

import json
import os
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# === 配置 ===
WRITEBACK_DIR = "data/writeback_kb"       # 回写知识库目录
WRITEBACK_INDEX = "data/writeback_kb/index.json"
MAX_WRITEBACK_SIZE = 200                  # 回写容量上限
GROUNDING_THRESHOLD = 0.7                 # 基础验证阈值
NOVELTY_THRESHOLD = 0.3                   # 新颖性阈值（与现有KB的最大相似度上限）
SIMILARITY_DEDUP_THRESHOLD = 0.92         # 去重阈值

# === Gate 1: Grounding Verification ===
def verify_grounding(answer, retrieved_chunks, llm_model):
    """
    验证回答是否被检索证据支持（简化版 NLI）
    使用 LLM 判断回答中的关键声明是否有检索片段支持
    """
    # 提取回答中的关键声明
    claims_prompt = f"""从以下回答中提取关键事实声明，每行一条：
回答：{answer}
关键声明："""

    # 用 Ollama 提取声明（轻量调用）
    claims = call_ollama(claims_prompt, model=llm_model, max_tokens=200)

    # 逐条检查声明是否有检索片段支持
    context = "\n\n".join([c["content"] for c in retrieved_chunks])
    verify_prompt = f"""判断以下每条声明是否能从参考资料中找到支持。
对每条声明回答"支持"或"不支持"。

参考资料：
{context}

声明列表：
{claims}

逐条判断："""

    results = call_ollama(verify_prompt, model=llm_model, max_tokens=300)

    # 统计支持比例
    lines = [l for l in results.strip().split("\n") if l.strip()]
    supported = sum(1 for l in lines if "支持" in l and "不支持" not in l)
    ratio = supported / max(len(lines), 1)

    return ratio >= GROUNDING_THRESHOLD, ratio

# === Gate 2: Attribution Check ===
def verify_attribution(answer, retrieved_chunks):
    """
    验证回答是否引用了检索片段来源
    检查回答中是否包含引用标记（如 [片段1], [来源:d001] 等）
    """
    import re
    # 检查是否包含引用标记
    citation_patterns = [
        r'\[片段\d+\]',
        r'\[来源:?\w+\]',
        r'\[\d+\]',
        r'根据.*?片段',
        r'参考.*?资料',
    ]
    has_citation = any(re.search(p, answer) for p in citation_patterns)
    return has_citation

# === Gate 3: Novelty Detection ===
def check_novelty(answer_embedding, existing_embeddings):
    """
    检测回答与现有知识库的语义差异
    避免写入与现有内容高度重复的回答
    """
    if not existing_embeddings:
        return True, 0.0

    similarities = cosine_similarity(
        [answer_embedding],
        existing_embeddings
    )[0]
    max_sim = similarities.max()

    # 与已有内容的最大相似度不应太高（否则是重复）
    is_novel = max_sim < (1.0 - NOVELTY_THRESHOLD)
    return is_novel, max_sim

# === 语义去重 ===
def check_duplicate(answer_embedding, writeback_embeddings, threshold=SIMILARITY_DEDUP_THRESHOLD):
    """
    检查回写库中是否已有高度相似的回答
    """
    if not writeback_embeddings:
        return False, None

    similarities = cosine_similarity(
        [answer_embedding],
        writeback_embeddings
    )[0]
    max_idx = similarities.argmax()
    max_sim = similarities[max_idx]

    return max_sim >= threshold, max_idx

# === 主入口 ===
def try_writeback(query, answer, retrieved_chunks, query_embedding, answer_embedding, llm_model):
    """
    尝试将回答写入知识库
    返回: (success: bool, reason: str)
    """
    # 容量检查
    if os.path.exists(WRITEBACK_INDEX):
        with open(WRITEBACK_INDEX, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {"entries": []}

    if len(index["entries"]) >= MAX_WRITEBACK_SIZE:
        return False, "容量上限"

    # Gate 1: 基础验证
    grounded, ratio = verify_grounding(answer, retrieved_chunks, llm_model)
    if not grounded:
        return False, f"基础验证未通过 (支持率={ratio:.2f})"

    # Gate 2: 归因检查
    attributed = verify_attribution(answer, retrieved_chunks)
    if not attributed:
        return False, "归因检查未通过（缺少引用标记）"

    # Gate 3: 新颖性检测（对比原始知识库）
    # 加载原始知识库的 embeddings
    kb_embeddings = load_kb_embeddings()
    novel, max_sim = check_novelty(answer_embedding, kb_embeddings)
    if not novel:
        return False, f"新颖性不足 (max_sim={max_sim:.2f})"

    # 去重检查（对比回写知识库）
    wb_embeddings = [e["embedding"] for e in index["entries"]]
    is_dup, dup_idx = check_duplicate(answer_embedding, wb_embeddings)
    if is_dup:
        return False, f"与已有回写条目重复 (sim={wb_embeddings[dup_idx]:.2f})"

    # === 全部通过，写入 ===
    entry = {
        "id": f"wb_{len(index['entries'])+1:04d}",
        "query": query,
        "answer": answer,
        "embedding": answer_embedding.tolist(),
        "source_chunks": [c["id"] for c in retrieved_chunks],
        "grounding_ratio": ratio,
        "timestamp": datetime.now().isoformat(),
        "ttl_days": 90,  # 90天有效期
        "quality": "verified",
    }
    index["entries"].append(entry)

    os.makedirs(WRITEBACK_DIR, exist_ok=True)
    with open(WRITEBACK_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return True, "写入成功"


# === 检索时整合回写知识库 ===
def search_writeback_kb(query_embedding, top_k=2):
    """
    在回写知识库中检索相关回答
    返回: 匹配的回写条目列表
    """
    if not os.path.exists(WRITEBACK_INDEX):
        return []

    with open(WRITEBACK_INDEX, "r", encoding="utf-8") as f:
        index = json.load(f)

    if not index["entries"]:
        return []

    # 过滤过期条目
    now = datetime.now()
    valid_entries = []
    for e in index["entries"]:
        created = datetime.fromisoformat(e["timestamp"])
        days = (now - created).days
        if days <= e.get("ttl_days", 90):
            valid_entries.append(e)

    if not valid_entries:
        return []

    embeddings = np.array([e["embedding"] for e in valid_entries])
    sims = cosine_similarity([query_embedding], embeddings)[0]

    # 按相似度排序
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    results = []
    for idx, sim in ranked[:top_k]:
        if sim >= 0.85:  # 回写库检索阈值
            entry = valid_entries[idx].copy()
            entry["writeback_score"] = float(sim)
            results.append(entry)

    return results
```

#### 修改 `rag_chain.py` 集成

```python
# rag_chain.py 中新增回写逻辑

from answer_writeback import try_writeback, search_writeback_kb

def rag_answer(query, top_k=4):
    """增强版 RAG：检索时同时查询回写库"""

    # 1. 常规检索
    query_embedding = get_embedding(query)
    retrieved = hybrid_search(query_embedding, top_k=top_k)

    # 2. 同时检索回写知识库
    writeback_results = search_writeback_kb(query_embedding, top_k=2)

    # 3. 合并上下文（回写回答作为补充参考）
    context_parts = []
    for chunk in retrieved:
        context_parts.append(f"[片段{chunk['id']}] {chunk['content']}")

    if writeback_results:
        context_parts.append("\n--- 历史问答参考 ---")
        for wb in writeback_results:
            context_parts.append(
                f"[历史Q] {wb['query']}\n[历史A] {wb['answer']}"
            )

    context = "\n\n".join(context_parts)

    # 4. 生成回答
    answer = generate_answer(query, context)

    # 5. 异步尝试回写（不阻塞主流程）
    try:
        answer_embedding = get_embedding(answer)
        success, reason = try_writeback(
            query, answer, retrieved,
            query_embedding, answer_embedding,
            LLM_MODEL
        )
        if success:
            print(f"✅ 回答已写入知识库")
        else:
            print(f"⏭️ 跳过回写: {reason}")
    except Exception as e:
        print(f"⚠️ 回写异常（不影响回答）: {e}")

    return {
        "answer": answer,
        "sources": retrieved,
        "writeback_hits": writeback_results,
    }
```

### 3.3 Streamlit 界面集成

```python
# app.py 侧边栏新增

with st.sidebar:
    # ... 现有设置 ...

    st.subheader("回答回写")
    enable_writeback = st.checkbox("启用回答回写", value=True)
    writeback_top_k = st.slider("回写库检索数", 0, 4, 2)

    if os.path.exists("data/writeback_kb/index.json"):
        with open("data/writeback_kb/index.json") as f:
            wb = json.load(f)
        st.metric("回写库大小", f"{len(wb['entries'])} 条")

# 主界面显示回写状态
if result.get("writeback_hits"):
    with st.expander(f"📚 命中 {len(result['writeback_hits'])} 条历史问答"):
        for wb in result["writeback_hits"]:
            st.write(f"**Q**: {wb['query']}")
            st.write(f"**A**: {wb['answer'][:200]}...")
            st.write(f"相似度: {wb['writeback_score']:.3f}")
```

---

## 四、数据流总览

```
用户提问
    │
    ├──→ [原始知识库检索] ──→ Top-K chunks
    │                              │
    ├──→ [回写知识库检索] ──→ 历史 Q&A pairs (if any)
    │                              │
    └──→ 合并上下文 → LLM 生成回答
                                │
                        ┌───────┴────────┐
                        │   多阶段验证     │
                        │                │
                        │ Gate1: 基础验证  │──→ LLM 判断声明是否有证据支持
                        │ Gate2: 归因检查  │──→ 正则检查引用标记
                        │ Gate3: 新颖性   │──→ 与原始KB的语义差异
                        │ Gate4: 去重检查  │──→ 与回写库已有条目的相似度
                        │                │
                        └───────┬────────┘
                                │
                    全部通过？──→ 写入 data/writeback_kb/
                    │
                    未通过 ──→ 丢弃（可保存 critique log）
```

---

## 五、评测指标

| 指标 | 定义 | 预期效果 |
|------|------|----------|
| **回写成功率** | 通过验证的回答占比 | 约 25-35%（保守策略） |
| **回写命中率** | 后续查询命中回写库的比例 | 随使用逐步提升 |
| **回答质量提升** | 有回写辅助 vs 无回写的评测分数差异 | 预期 5-15% 提升 |
| **幻觉传播率** | 回写库中错误回答被后续引用的比例 | 目标 < 5% |
| **去重效率** | 回写库中重复条目占比 | 目标 < 3% |

---

## 六、实施路线

### Phase 1: 基础回写（1-2天）
- [x] 创建 `src/answer_writeback.py`
- [ ] 实现 Gate 1-3 验证逻辑
- [ ] 集成到 `rag_chain.py`
- [ ] 基础评测

### Phase 2: 检索整合（1天）
- [ ] `search_writeback_kb()` 集成到主检索流程
- [ ] Streamlit 界面展示回写命中
- [ ] 回写库容量管理 + TTL 过期

### Phase 3: 进阶优化（2-3天）
- [ ] Experience Store（保存被拒绝回答的 critique）
- [ ] 质量分级标记（verified / unverified / user_approved）
- [ ] 用户反馈按钮（"这个回答有用" → 强制写入）
- [ ] 回写库统计仪表板

---

## 七、参考文献

1. **Bidirectional RAG** (2024) — Safe Self-Improving RAG Through Multi-Stage Validation. 首次提出安全回写架构，多阶段验证（NLI + Attribution + Novelty）。
2. **GroundedCache** (2025) — Evidence-Validated Cache Routing for RAG. 4门控缓存安全机制，lexical support gate 是关键。
3. **ROZA Graphs** (2025) — Self-Improving Near-Deterministic RAG through Evidence-Centric Feedback. 推理图持久化，47% 错误减少。GitHub: 未公开。
4. **EvoRAG** (2025) — Making KG-RAG Automatically Evolve through Feedback-driven Backpropagation. 反馈驱动反向传播优化 KG。GitHub: `iDC-NEU/EvoRAG`。
5. **Proximity Cache** (2025) — Approximate Caching for Faster RAG. LSH 加速，数据库调用减少 77.2%。
6. **GPTCache** — 开源语义缓存库，支持 embedding 相似度匹配。
7. **LightRAG** (HKUDS, 36K stars) — 增量更新机制 `apipeline_enqueue_documents`，可参考其文档增量插入流程。
8. **AWS ElastiCache Semantic Cache** — 工业级语义缓存方案，成本降低 86%，延迟降低 88%。
