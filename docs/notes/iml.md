# 基于论文对 `ollama-rag-course-design` 的改进建议（iml）

## 1. 项目现状速览（结合 README 与代码）

当前项目是**本地轻量 RAG**实现，特点是：

- 本地推理：`Ollama + qwen/deepseek + nomic-embed-text`
- 轻依赖索引：`chroma_db/knowledge_index.json`（实际是 JSON 向量库）
- 检索模式：`vector` 与 `hybrid`（向量+TF-IDF）
- 主流程集中在：`src/rag_chain.py`
- 评测维度偏基础：命中率、参考答案覆盖度、耗时（`src/eval.py`, `src/rag_strategy_compare.py`, `src/model_compare.py`）

当前优势：实现完整、可复现、教学友好。
当前瓶颈：

1. 检索仍是“单轮 top-k + 一次生成”主范式；
2. 缺少“何时继续检索/何时停止”的决策机制；
3. 重排与证据质量评估较弱；
4. 缺少“忠实性/矛盾率”约束指标；
5. 实验有策略对比，但缺“过程级行为指标”。

---

## 2. 与本项目最相关的论文机制（已检索）

下列机制和当前代码可直接对接：

1. **R-Search / ReasonRAG / ProRAG**
   - 核心：把 reasoning-search 轨迹当作可优化对象，引入**多奖励/过程奖励**。
   - 对本项目启发：先不做大规模 RL，也可先落地“规则化过程奖励 + 离线打分”。

2. **ReaLM-Retrieve / SIM-RAG**
   - 核心：通过不确定性或 critic 判断“是否继续检索”。
   - 对本项目启发：在 `rag_chain.py` 增加轻量的**继续检索判定器**（无需改模型权重）。

3. **GRIP（Retrieval as Generation）**
   - 核心：将 `[RETRIEVE]/[ANSWER]/[SOLVED]` 作为控制行为。
   - 对本项目启发：可用 prompt + 解析器模拟控制 token，构建“可解释决策轨迹”。

4. **R3-RAG（结果奖励+过程奖励）**
   - 核心：两阶段训练，冷启动迭代推理+检索 → RL 探索检索环境；结果奖励（答案正确性）+ 过程奖励（相关文档验证）。
   - 对本项目启发：在 hybrid 之后加入二级 rerank（LLM 打分或规则打分），引入相关性二分类标注。

5. **Self-Correcting RAG（MMKP + NLI-guided MCTS）**
   - 核心：输入端做预算优化选文；输出端做忠实性自校验。
   - 对本项目启发：
     - 输入端先做简化版“冗余惩罚+预算约束选段”；
     - 输出端加入 NLI 风格“是否与证据矛盾”检查。

6. **A-RAG（分层检索接口）**
   - 核心：keyword_search / semantic_search / chunk_read 三层工具。
   - 对本项目启发：把当前检索封装为分层接口，支持 agent 化扩展。

7. **A-RAG（广覆盖+深精炼）**
   - 核心：chunk_read 读取相邻片段，支持先广泛检索再精选。
   - 对本项目启发：先高召回候选，再局部精炼重排，优于一次性 top-k。

---

## 3. 面向当前代码的可落地改进建议

## 3.1 P0（建议优先，本周可做）

### 建议 A：增加“自适应多轮检索”模式（最关键）

- 目标：不再固定一次检索，改为最多 N 轮（如 2~3 轮）。
- 参考论文：ReaLM-Retrieve、SIM-RAG、R-Search、GRIP。
- 对接文件：`src/rag_chain.py`。
- 最小实现：
  1. 第 1 轮按当前 `hybrid` 检索；
  2. 让模型输出：`是否信息充足`、`若不足则给出下一轮检索子问题`；
  3. 若不足，执行第 2 轮检索并合并候选；
  4. 达到停止条件后再生成最终答案。
- 建议新增参数：
  - `MAX_RETRIEVAL_ROUNDS=3`
  - `SUFFICIENCY_THRESHOLD`
  - `ENABLE_ADAPTIVE_RETRIEVAL=True`

### 建议 B：加入二级重排（utility-aware rerank）

- 目标：让最终上下文更“有用”，而不是仅相关。
- 参考论文：RankRAG、R3-RAG。
- 对接文件：`src/rag_chain.py`。
- 最小实现：
  1. 初检索取 top-12；
  2. 用轻量打分函数对每个 chunk 评分：
     - 语义相关分（已有）
     - 词面覆盖分（已有）
     - 新增：问题关键词覆盖率、实体重合、句式完整度
  3. 综合后截断为最终 top-k（如 4）。

### 建议 C：增加“忠实性检查”后处理

- 目标：降低幻觉并提供可解释拒答。
- 参考论文：Self-Correcting RAG。
- 对接文件：`src/rag_chain.py`, `src/eval.py`。
- 最小实现：
  1. 生成答案后，让模型逐条声明“关键结论-证据片段ID”；
  2. 若结论找不到证据支撑，则触发“保守重答/拒答”；
  3. 输出 `faithfulness_flag`。

---

## 3.2 P1（两周内可做）

### 建议 D：构建分层检索接口（A-RAG 风格）

- 对接文件：可新增 `src/retrievers.py`（或 `src/rag_chain.py` 内先拆函数）。
- 接口建议：
  - `keyword_search(query, k)`
  - `semantic_search(query, k)`
  - `chunk_read(chunk_id)`
- 价值：
  - 提升系统可解释性；
  - 为后续 Agent 化/控制 token 化打基础。

### 建议 E：实现“广覆盖→深精炼”两阶段检索

- 参考：A-RAG（chunk_read + 分层检索）。
- 流程：
  1. Broad stage：不同策略并行取候选（向量、TF-IDF、关键词）；
  2. Refine stage：按问题实体与推理链需求做精炼。
- 对接文件：`src/rag_chain.py`。

### 建议 F：扩展评测指标（过程层）

- 对接文件：`src/eval.py`, `src/rag_strategy_compare.py`。
- 新增指标：
  - `avg_retrieval_rounds`
  - `useful_chunk_rate`（最终答案引用到的chunk占比）
  - `faithfulness_rate`
  - `unsupported_claim_rate`
  - `redundancy_rate`（上下文重复度）

---

## 3.3 P2（课程加分项）

### 建议 G：离线过程奖励数据集（轻量版 ProRAG/ReasonRAG）

- 思路：不直接全量 RL，先积累轨迹数据。
- 对接：新增 `results/trajectory_logs.jsonl`。
- 记录字段建议：
  - question
  - round_id
  - query_generated
  - retrieved_chunks
  - final_answer
  - reward_answer（F1/overlap）
  - reward_evidence（证据可验证性）
- 后续可用于：
  - 学习“是否继续检索”的分类器；
  - 学习 reranker 的监督信号。

### 建议 H：控制 token 轨迹输出（GRIP 风格轻量模拟）

- 不改底模，仅在 prompt 中约束输出：
  - `[INTERMEDIARY]`
  - `[RETRIEVE]`
  - `[ANSWER]`
  - `[SOLVED]`
- 价值：答辩展示直观，解释“系统如何做决策”。

---

## 4. 推荐代码改造顺序（最小风险）

1. **先改 `src/rag_chain.py`**：加入多轮检索与停止判定；
2. **再改 `src/eval.py`**：新增过程指标；
3. **再改 `src/rag_strategy_compare.py`**：加入新模式对照（`single` vs `adaptive`）；
4. **最后改 `src/app.py`**：展示“检索轮次、停止原因、证据校验结果”。

这样可以始终保持“每一步都可运行、可验证”。

---

## 5. 预期收益与风险

### 预期收益

- 命中率和覆盖度继续提升（尤其多跳问题）；
- 幻觉率下降，拒答更合理；
- 评测报告从“结果指标”升级到“过程指标”；
- 答辩亮点更强：有“策略控制、证据校验、可解释轨迹”。

### 风险与应对

- 风险1：多轮检索增加时延。
  - 应对：限制 `MAX_RETRIEVAL_ROUNDS<=3`，并缓存 query->retrieval 结果。
- 风险2：规则过严导致过度拒答。
  - 应对：在 `faithfulness` 阈值上做验证集调参。
- 风险3：实现复杂度升高。
  - 应对：先 P0 最小实现，再逐步模块化。

---

## 6. 结论（针对当前项目的最优策略）

对该课程项目最具性价比的升级路径是：

**“自适应多轮检索 + 二级重排 + 忠实性校验 + 过程评测指标”**。

它直接继承你现有的本地 Ollama 与轻量索引架构，不需要立刻进行高成本 RL 训练，却能吸收 R-Search / ReaLM-Retrieve / RankRAG / Self-Correcting RAG / A-RAG 的关键思想，形成一个更强、更可解释、且更适合答辩展示的 RAG 系统。

---

## 附录：移除的无开源代码论文

> 以下论文因暂无公开代码仓库或模型权重，已从正文中移除，但其核心思路仍具参考价值。

| 论文 | 原引用位置 | 移除原因 | 核心思路简述 |
|------|-----------|---------|-------------|
| Search-P1 (#4, RL部分) | 第2节机制1、建议B参考论文 | 匿名评审期，附录标注 "(anonymized)" | 路径中心奖励塑形：双轨路径评分 + 软结果评分，提升检索路径质量 |
| TreePS-RAG (#6, RL部分) | 第2节机制1 | 论文未提及代码链接，搜索未找到仓库 | 将推理 rollout 建模为树，蒙特卡洛估计节点效用并相似性剪枝 |
| RRPO (#14, RL部分) | 第2节机制4、建议B参考论文、结论 | 论文未提及代码链接，仅有第三方请求实现的 issue | 将重排序建模为 MDP，使用 LLM 反馈作为奖励优化文档"有用性" |
| PAR²-RAG (#20, RL部分) | 第2节机制7、建议E参考 | 论文未提及代码链接，搜索未找到仓库 | 广度优先锚定 + 深度优先精炼，提升准确率与 NDCG |
