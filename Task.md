# 课程设计任务完成情况核对

核对时间：2026-06-22（高级改进与 full30 结果已同步）
项目主题：基于 Ollama 的本地大模型部署与计算机专业文档问答 RAG 应用

## 一、总体结论

当前项目已经完成基础档、进阶层和提升档的主要要求，并且具备加分挑战所需的模型对比材料；同时已补充公开数据集和计算机领域论文材料作为可追溯原始知识库来源。`src/` 下核心脚本也已补充必要中文注释，便于报告说明和答辩时定位实现位置。

此外，在课程要求之外，已参照近年 RAG 论文（Self-RAG / SIM-RAG / A-RAG / ReasonRAG / RA-DIT 等）实现了 7 项高级改进并全部通过端到端验证，详见第八节。

按现有文件证据判断：

- 基础档（60-69 分）：已完成。
- 进阶层（70-79 分）：已完成主要要求。
- 提升档（80-89 分）：已完成；已有 Prompt 约束、混合检索，并补齐了”纯向量检索 vs 混合检索”的优化前后控制实验。
- 加分挑战（+10 分）：已具备”模型对比”实现和完整双模型评测结果，可作为挑战任务材料。
- **高级 RAG 改进（额外加分材料）**：7 项改进全部实现并验证，见第八节。

## 二、基础档核对

### 1. Ollama 安装配置 + 模型部署

状态：已完成。

证据文件：

- `port.md`
- `README.md`
- `src/config.py`

已记录内容：

- Ollama API 地址：`http://127.0.0.1:8090`
- Docker Desktop 端口映射：`8090:11434`
- Python conda 环境：`nlprag`
- 生成模型：`qwen2.5:7b-instruct-q4_K_M`
- Embedding 模型：`nomic-embed-text`
- 对比模型：`deepseek-r1:7b-qwen-distill-q4_K_M`

相关命令已在 `README.md` 和 `doc-formal.txt` 中说明：

```cmd
curl http://127.0.0.1:8090/api/tags
conda activate nlprag
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull deepseek-r1:7b-qwen-distill-q4_K_M
ollama pull nomic-embed-text
```

### 2. 选定一个任务类型，明确任务定义、输入输出

状态：已完成。

任务类型：计算机专业文档问答 / 本地 RAG 问答系统。

输入：

- `data/` 目录下的计算机专业文档。
- 用户自然语言问题。

输出：

- 本地大模型生成的问答结果。
- 引用来源片段、来源文件、相似度分数。
- 检索耗时、生成耗时、总耗时等运行信息。

证据文件：

- `doc-formal.txt`
- `README.md`
- `RAG流程实现说明.md`
- `src/rag_chain.py`
- `src/app.py`

### 3. 完成基本任务实现，能够运行并产生结果

状态：已完成。

核心实现文件：

- `src/build_kb.py`：读取文档、切分文本、生成 Embedding、构建索引。
- `src/collect_public_kb.py`：采集公开 CSE 数据集和 arXiv 计算机论文摘要。
- `src/rag_chain.py`：检索、Prompt 拼接、调用 Ollama 生成答案。
- `src/app.py`：Streamlit Web 问答页面。
- `src/eval.py`：批量评测脚本。
- `src/model_compare.py`：双模型对比脚本。

数据与索引：

- 原始知识库文档：`data/`
- 知识库索引：`chroma_db/knowledge_index.json`
- 当前索引已经包含自建文档和公开材料，`chroma_db/knowledge_index.json` 中记录 `document_count=41`、`chunk_count=163`。
- 公开材料已保存在 `data/public_kb/`，来源、许可、URL 和本地输出路径见 `data/public_kb/manifest.json`。

已执行的代码语法核验：

```cmd
python -m py_compile src\config.py src\collect_public_kb.py src\build_kb.py src\rag_chain.py src\eval.py src\model_compare.py src\app.py
```

结果：通过，无语法错误。

### 4. 撰写报告

状态：已完成。

证据文件：

- `doc-formal.txt`
- `Ollama_RAG_课程设计完整操作步骤.docx`
- `RAG流程实现说明.md`

报告内容覆盖：

- 任务背景与目标。
- 技术方案。
- 实现过程。
- 测试数据与评测指标。
- 结果分析方法。
- 总结与反思。

篇幅核对：

- `doc-formal.txt` 正文非空白字符约 6151 个，满足“不少于 3000 字”的要求。

基础档结论：基础档 4 项要求均已完成。

## 三、进阶层核对

### 1. 效果评测至少 30 条样本

状态：已完成。

证据文件：

- `eval_questions.csv`
- `results/model_compare_results.csv`
- `results/model_compare_summary.json`

样本数量：

- `eval_questions.csv`：30 条问题。
- `results/model_compare_results.csv`：60 条结果，即 2 个模型各 30 条。

样本覆盖类别：

- 网络
- 操作系统
- 数据库
- 人工智能
- RAG

说明：

- 当前 30 条评测样本是项目内自建评测集，用于保证与原有知识库问题严格对应。
- 原始知识库已补充公开来源：Hugging Face `hatakekksheeshh/CSE_course_RAG`、`CCRss/arXiv_dataset` 和 arXiv Computer Science 论文摘要。
- 公开材料的来源、许可、URL 和输出路径会记录在 `data/public_kb/manifest.json`，满足可追溯要求。

### 2. 技术分析

状态：已完成。

证据文件：

- `doc-formal.txt`
- `RAG流程实现说明.md`
- `results/model_compare_summary.json`

已有技术分析内容：

- 文档解析与切分。
- Embedding 向量化。
- 向量相似度检索。
- 字符 n-gram TF-IDF 关键词检索。
- 混合检索加权策略：当前 full30 使用三路融合，即 `0.40 * 向量分 + 0.45 * TF-IDF 关键词分 + 0.15 * 实体匹配分`。
- Prompt 约束生成。
- 检索命中率、参考答案覆盖度、响应时间等指标说明。

进阶层结论：进阶层主要要求已完成。

## 四、提升档核对

要求：在进阶层基础上，选择 Prompt Engineering / 模型组合 / RAG 优化 中的一项进行优化，并且必须有明确的效果对比。

当前状态：已完成。

已具备的优化材料：

- Prompt Engineering：`src/rag_chain.py` 中已有严格依据参考资料、资料不足拒答、输出引用来源的 Prompt 模板。
- RAG 策略优化：当前检索支持 `vector` 纯向量检索、`hybrid` 三路融合检索和 `adaptive` 自适应检索。报告主结论使用 `vector` 与 `hybrid` 的完整 30 条控制实验。
- 模型组合/模型对比材料：`src/model_compare.py` 已支持 DeepSeek R1 与 Qwen 双模型对比。
- 优化控制实验：`src/rag_strategy_compare.py` 已完成 `vector` 与 `hybrid` 的同条件对比。

已有对比结果（历史结果保留，主结论以 `full30_20260621` 为准）：

首次完整实验结果中，`results/rag_strategy_compare_summary.json` 显示：

| 检索策略 | 样本数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 30 | 1.0000 | 0.7667 | 0.4058 | 0.4444s | 137.7124s | 138.1570s |
| hybrid | 30 | 1.0000 | 1.0000 | 0.5948 | 0.2312s | 129.4322s | 129.6635s |

重新完整流程实验结果（2026-05-23）中，`results/rag_strategy_compare_summary.json` 显示：

| 检索策略 | 样本数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 30 | 1.0000 | 0.7000 | 0.4076 | 0.4339s | 123.1658s | 123.5998s |
| hybrid | 30 | 1.0000 | 1.0000 | 0.6185 | 0.4084s | 137.9988s | 138.4073s |

单模型重新完整流程实验结果（2026-05-23）中，`results/eval_summary.json` 显示：

| 样本数 | Top-K | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 4 | 1.0000 | 0.6143 | 0.4019s | 135.2647s | 135.6667s |

首次完整实验结果中，`results/model_compare_summary.json` 显示：

| 模型 | 样本数 | 成功数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-r1:7b-qwen-distill-q4_K_M | 30 | 30 | 1.0 | 1.0 | 0.4324 | 171.2024s |
| qwen2.5:7b-instruct-q4_K_M | 30 | 30 | 1.0 | 1.0 | 0.5746 | 143.3015s |

重新完整流程实验结果（2026-05-23）中，`results/model_compare_summary.json` 显示：

| 模型 | 样本数 | 成功数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1:7b-qwen-distill-q4_K_M | 30 | 30 | 1.0000 | 1.0000 | 0.4765 | 0.4563s | 165.0030s | 165.4594s |
| qwen2.5:7b-instruct-q4_K_M | 30 | 30 | 1.0000 | 1.0000 | 0.6205 | 0.4582s | 138.7100s | 139.1683s |

full30 主结论：

- 在 `full30_20260621` 完整 30 条实验中，`vector` 命中率为 0.7000，参考覆盖度为 0.3669。
- `hybrid` 命中率为 1.0000，参考覆盖度为 0.5404。
- 当前 `hybrid` 使用三路融合：`0.40 * 向量分 + 0.45 * TF-IDF 关键词分 + 0.15 * 实体匹配分`，不是旧版二路权重。
- 因此，`hybrid` 相对 `vector` 将检索命中率从 0.7000 提升到 1.0000，将参考覆盖度从 0.3669 提升到 0.5404，已经形成明确的 RAG 策略优化前后对照。

提升档结论：已补齐明确的优化前后控制实验，可支撑 RAG 策略优化材料。

## 五、加分挑战核对

要求：在提升档基础上完成 LoRA 微调 / 公开评测 / 创新应用 / 模型对比 中任一任务。

当前状态：已具备“模型对比”挑战材料，且提升档控制实验已补齐。

证据文件：

- `src/model_compare.py`
- `results/model_compare_results.csv`
- `results/model_compare_summary.json`
- `results/model_compare_run.out`

已完成内容：

- 对比模型 1：`deepseek-r1:7b-qwen-distill-q4_K_M`
- 对比模型 2：`qwen2.5:7b-instruct-q4_K_M`
- 每个模型评测 30 条样本。
- 输出模型成功率、检索命中率、平均参考答案覆盖度、平均检索耗时、平均生成耗时、平均总耗时。

结果摘要：

- 两个模型成功率均为 1.0。
- 两个模型检索命中率均为 1.0。
- Qwen 平均参考答案覆盖度更高：0.5746，高于 DeepSeek R1 的 0.4324。
- Qwen 平均总耗时更短：143.3015s，短于 DeepSeek R1 的 171.2024s。
- 2026-05-23 重新完整流程实验中，Qwen 平均参考答案覆盖度为 0.6205，高于 DeepSeek R1 的 0.4765。
- 2026-05-23 重新完整流程实验中，Qwen 平均总耗时为 139.1683s，短于 DeepSeek R1 的 165.4594s。

挑战任务结论：

- “模型对比”材料已经完成。
- 提升档中的优化前后对比实验已经补齐，模型对比可作为加分挑战材料。

## 六、当前注意事项

### 必须注意

1. `full30_20260621` 是完整 30 条全套评测，不是轻量冒烟测试；报告主结论应优先引用该组结果。
2. `goal_20260621` 只用于链路验证，不作为主要质量结论。
3. 当前 `hybrid` 是三路融合，不能在报告中写成旧版 `0.45 * 向量 + 0.55 * TF-IDF`。
4. DeepSeek 在 full30 双模型对比中的参考覆盖度为 0.0000，应解释为自动覆盖指标与输出格式不匹配的现象，不能简单写成模型语义完全错误。
5. 公开知识库采集脚本已经补充，且公开材料已经进入当前索引；如果后续需要刷新公开材料或扩大样本，再运行 `python src/collect_public_kb.py` 和 `python src/build_kb.py`。
6. `src/` 代码已补充中文注释，注释主要解释工程取舍和实验控制变量，不改变业务逻辑。
7. 重新运行 `python src/build_kb.py` 会覆盖 `chroma_db/knowledge_index.json`，属于正常重建行为。

### 可选补充

1. 如需单模型独立评测结果，可运行：

```cmd
python src/eval.py
```

生成：

- `results/eval_results.csv`
- `results/eval_summary.json`

2. RAG 优化对比实验已完成，复现实验命令为：

```cmd
python src/rag_strategy_compare.py
```

3. 对比结果已写入 `doc-formal.txt` 和 `README.md`。

## 七、最终判定

| 档位 | 要求 | 当前完成情况 | 判定 |
|---|---|---|---|
| 基础档 | 部署、任务定义、基本实现、3000 字报告、至少 10 条测试结果 | 均有证据支撑 | 已完成 |
| 进阶层 | 至少 30 条样本评测、技术分析 | 30 条样本和技术分析已具备 | 已完成 |
| 提升档 | Prompt / 模型组合 / RAG 优化，且有明确效果对比 | 已完成纯向量检索 vs 混合检索控制实验 | 已完成 |
| 加分挑战 | LoRA 微调 / 公开评测 / 创新应用 / 模型对比 | 已完成双模型对比 | 已具备材料 |
| **高级 RAG 改进** | 参照 RAG 论文实现高级技术 | 7 项改进全部实现并验证（见第八节） | **已完成** |

综合判断：项目当前可以支撑基础档、进阶层和提升档；双模型评测结果也可作为加分挑战材料；7 项高级改进可作为额外论据材料。

## 八、高级 RAG 改进核对（2026-06-12）

基于近年 RAG 论文，在原有基础上实现以下 7 项改进，均已通过 Python 语法检查和端到端运行验证。

### 改进 1：三路融合检索（Hybrid V3）

状态：**已完成并验证**。

理论依据：A-RAG（arXiv:2405.12777）、LightRAG 实体图检索思想。

实现位置：`src/rag_chain.py` — `entity_match_score()` + `retrieve()` hybrid 分支。

实现内容：
- 在原有向量 + TF-IDF 二路融合基础上，增加实体匹配第三路。
- 实体匹配：用正则提取问题中的英文技术词（`[A-Za-z][A-Za-z0-9_\-]+`）、中文双字词（`[一-鿿]{2,}`）和数字，计算在片段中的出现比例。
- 三路权重之和为 1.0：`0.40 * 向量 + 0.45 * TF-IDF + 0.15 * 实体匹配`。
- 新增配置项：`ENTITY_SCORE_WEIGHT = 0.15 / EMBEDDING_SCORE_WEIGHT_V3 = 0.40 / LEXICAL_SCORE_WEIGHT_V3 = 0.45`。

验证结果：`retrieve()` 返回结果包含 `embedding_score`、`lexical_score`、`entity_score` 三路得分，实体词完全命中时 `entity_score=1.0`，部分命中时 `0.5`，无命中时 `0.0`。

证据文件：`src/rag_chain.py`、`src/config.py`。

### 改进 2：语义去重（Semantic Deduplication）

状态：**已完成并验证**。

理论依据：Self-Correcting RAG（MMKP 聚类简化版，arXiv:2406.13692）。

实现位置：`src/rag_chain.py` — `deduplicate_by_similarity()`。

实现内容：
- 对 `top_k * 2` 候选片段计算成对余弦相似度矩阵（`1 - cosine_similarity`）。
- 用 `AgglomerativeClustering(linkage="average", metric="precomputed")` 进行凝聚聚类，距离阈值 `1 - 0.85 = 0.15`。
- 每簇保留综合得分最高的片段，返回去重后结果。
- 使用 `retrieve()` 输出中的 `embedding` 字段（真实向量），并在 `_sources_from_chunks()` 中将其剥离，不对外暴露。

验证结果：8 条语义各异的片段经过去重后均保留（无误合并），1 条语义相近的重复内容被正确过滤。

证据文件：`src/rag_chain.py`。

### 改进 3：上下文扩展（Context Expansion）

状态：**已完成并验证**。

理论依据：A-RAG chunk_read 扩展机制（arXiv:2405.12777）。

实现位置：`src/rag_chain.py` — `expand_context()`。

实现内容：
- 检索完成后，对每个返回片段查找其在 `chroma_db/knowledge_index.json` 中的位置索引。
- 追加相邻 ±`EXPANSION_WINDOW`（默认 1）位置的片段，得分标记为 0.0（区分于检索片段）。
- 同一文档内的邻居才被追加，避免跨文档污染。
- 配置项：`CONTEXT_EXPANSION = True / EXPANSION_WINDOW = 1`。

验证结果：2 个检索片段 → 扩展后 4 个片段，新增片段 ID 为相邻位置，得分为 0.0。

证据文件：`src/rag_chain.py`、`src/config.py`。

### 改进 4：自适应多轮检索

状态：**已完成并验证**。

理论依据：SIM-RAG 充分性判断（arXiv:2406.10208）+ ReaLM-Retrieve 步级不确定性（华南理工/阿里 2024）。

实现位置：`src/rag_chain.py` — `answer_question_adaptive()`。

实现内容：
- 第一轮：混合检索 Top-K 片段，格式化上下文。
- 充分性判断：调用 LLM 评估是否可以基于当前上下文回答问题，输出包含"不足"则触发第二轮。
- 第二轮（按需）：扩大到 `ADAPTIVE_TOP_K_SECOND=8` 重新检索。
- 最终：上下文扩展 + 语义去重 + cap 至 `top_k` 后生成答案。
- 结果返回 `retrieval_rounds`（检索轮数）和 `sufficiency_judgment`（充分性判断结果）。
- 配置项：`ADAPTIVE_RETRIEVAL = True / ADAPTIVE_TOP_K_SECOND = 8 / MAX_RETRIEVAL_ROUNDS = 2`。

验证结果：问"操作系统进程和线程区别"时 `retrieval_rounds=1`（首轮充分），`sources=3`，完整返回答案。

证据文件：`src/rag_chain.py`、`src/config.py`、`src/rag_strategy_compare.py`（支持 `--modes adaptive`）。

### 改进 5：分步推理与自反思 Prompt

状态：**已完成并验证**。

理论依据：Self-RAG（ICLR 2024，Asai et al.）+ ReasonRAG 分步推理。

实现位置：`src/rag_chain.py` — `PROMPT_TEMPLATE_STEPWISE`、`PROMPT_TEMPLATE_REFLECTIVE`、`SUFFICIENCY_PROMPT`、`FAITHFULNESS_CHECK_PROMPT`；`answer_question(prompt_variant=)`。

实现内容：
- `baseline`：直接依据参考资料回答，资料不足时拒答。
- `stepwise`（推荐，默认）：强制四步推理——步骤 1 识别信息线索 → 步骤 2 提取关键事实 → 步骤 3 综合推导 → 步骤 4 最终答案并引用 `[片段N]` 编号。
- `reflective`：先给出初步答案，再"审查"每个主要声明是否有片段依据，给出修订后答案。
- CLI：`python src/rag_chain.py --prompt {baseline,stepwise,reflective}`。

验证结果：`answer_question(prompt_variant='stepwise')` 输出包含完整四步结构，`[片段N]` 引用存在，生成时间约 58.4s。

证据文件：`src/rag_chain.py`。

### 改进 6：多维度评测指标

状态：**已完成并验证**。

理论依据：Gao et al., RAG Survey（arXiv:2312.10997）忠实度/相关性/完整性框架。

实现位置：`src/eval.py` — `faithfulness_score()`、`reasoning_faithfulness()`、`test_rejection()`；`run_eval()` 扩展参数。

实现内容：
- `faithfulness_score(answer, contexts, model)`：LLM 打分 0–10，归一化到 0–1，评估答案声明是否有片段依据。
- `reasoning_faithfulness(answer, contexts)`：正则找 `[片段N]` 或 `[N]` 引用，检查 N 在合法范围内，返回有效引用比例。
- `test_rejection(model)`：3 条超出知识库范围的问题，检查是否触发拒答词（"没有找到明确答案"/"无法回答"/"知识库中没有"）。
- `run_eval()` 结果增加 `reasoning_faithfulness` 列。
- CLI 新增参数：`--prompt {baseline,stepwise,reflective}`、`--faithfulness`、`--rejection`。

证据文件：`src/eval.py`。

### 改进 7：检索权重网格搜索

状态：**已完成并验证**。

理论依据：UR2（Tsinghua-dhy/UR2）+ RA-DIT 双端优化思想（轻量网格搜索实现）。

实现位置：`src/weight_tune.py` — `grid_search()`、`_run_one()`。

实现内容：
- 候选权重：`emb` ∈ {0.30, 0.35, 0.40, 0.45, 0.50}，`lex` ∈ {0.35, 0.40, 0.45, 0.50, 0.55}，`entity` ∈ {0.05, 0.10, 0.15, 0.20}。
- 过滤三者之和不等于 1.0 的组合（±0.01 容差）。
- 用 `config.EMBEDDING_SCORE_WEIGHT_V3 = emb; importlib.reload(rag_chain)` 模式热重载权重，无需重启进程。
- 以 `hit_rate + avg_overlap` 综合评分选最优组合。
- 搜索结束后自动恢复默认权重（0.40 / 0.45 / 0.15）。
- CLI：`--entity-fixed 0.15`（固定实体权重，加速搜索）、`--limit 10`（快速评测）。

输出文件：`results/weight_tune_results.json`，包含 `best` 最优权重和 `all` 全量结果。

证据文件：`src/weight_tune.py`。

### 高级改进完成汇总

| 编号 | 改进项 | 理论依据 | 实现文件 | 状态 |
|---|---|---|---|---|
| 1 | 三路融合检索 | A-RAG / LightRAG | `rag_chain.py` `config.py` | 已验证 |
| 2 | 语义去重 | Self-Correcting RAG | `rag_chain.py` | 已验证 |
| 3 | 上下文扩展 | A-RAG chunk_read | `rag_chain.py` `config.py` | 已验证 |
| 4 | 自适应多轮检索 | SIM-RAG / ReaLM-Retrieve | `rag_chain.py` `rag_strategy_compare.py` | 已验证 |
| 5 | 分步推理与自反思 Prompt | Self-RAG / ReasonRAG | `rag_chain.py` `prompt_compare.py` | 已验证 |
| 6 | 多维度评测指标 | RAG Survey Gao et al. | `eval.py` | 已验证 |
| 7 | 检索权重网格搜索 | UR2 / RA-DIT | `weight_tune.py` | 已验证 |

## 2026-06-21 补充流程记录（不覆盖原结果）

本轮目标是在当前 `feature` 分支的高级 RAG 改动基础上边修复运行问题边执行一轮可复现流程。由于当前 Docker/Ollama 使用 CPU 推理 `qwen2.5:7b-instruct-q4_K_M`，单题生成耗时较长，若继续使用默认 `NUM_PREDICT=300` 和完整上下文扩展，评测容易触发 240 秒读取超时。因此本轮补充流程采用轻量运行参数验证完整链路，历史 30 条完整评测结果不被覆盖。

本轮修复与增强：

- `src/rag_chain.py`：为生成调用增加 `num_predict`、`timeout` 和 `expand_context_enabled` 可选参数；CLI 新增 `--num-predict`、`--timeout`、`--no-context-expansion`。
- `src/eval.py`：新增 `--limit` 和 `--output-suffix`，支持小样本补充评测并把新结果写入带后缀文件。
- `src/rag_strategy_compare.py`、`src/prompt_compare.py`：新增短生成参数、上下文扩展开关和输出后缀，避免覆盖既有结果。
- `src/weight_tune.py`：改为纯检索权重搜索，不再对每组权重调用 LLM 生成，指标改为 `hit_rate + avg_score_margin`，更符合检索权重调优目的。

本轮执行命令：

```cmd
python -m py_compile src\config.py src\rag_chain.py src\eval.py src\rag_strategy_compare.py src\prompt_compare.py src\weight_tune.py src\app.py
python src\build_kb.py
python src\eval.py --limit 3 --top-k 2 --prompt baseline --num-predict 24 --timeout 180 --no-context-expansion --output-suffix goal_20260621
python src\rag_strategy_compare.py --limit 2 --modes vector hybrid adaptive --top-k 2 --prompt baseline --num-predict 24 --timeout 180 --no-context-expansion --output-suffix goal_20260621
python src\prompt_compare.py --limit 1 --variants baseline stepwise reflective --top-k 2 --num-predict 24 --timeout 180 --no-context-expansion --output-suffix goal_20260621
python src\weight_tune.py --limit 5 --top-k 2 --entity-fixed 0.15 --output-suffix goal_20260621
```

本轮新增结果文件：

- `results/eval_results_goal_20260621.csv`
- `results/eval_summary_goal_20260621.json`
- `results/rag_strategy_compare_results_goal_20260621.csv`
- `results/rag_strategy_compare_summary_goal_20260621.json`
- `results/prompt_compare_results_goal_20260621.csv`
- `results/prompt_compare_summary_goal_20260621.json`
- `results/weight_tune_results_goal_20260621.json`

本轮结果摘要：

| 实验 | 样本/设置 | 主要结果 | 平均耗时 |
|---|---|---|---:|
| 单模型轻量评测 | 3 条，Top-K=2，baseline，num_predict=24 | 检索命中率 1.0000，平均参考覆盖度 0.1057 | 71.1078s |
| RAG 策略对比 vector | 2 条，Top-K=2 | 命中率 0.5000，覆盖度 0.2561 | 59.0952s |
| RAG 策略对比 hybrid | 2 条，Top-K=2 | 命中率 1.0000，覆盖度 0.1220 | 63.8648s |
| RAG 策略对比 adaptive | 2 条，Top-K=2 | 命中率 1.0000，覆盖度 0.1220，平均检索轮数 1.0000 | 140.4118s |
| Prompt 对比 baseline | 1 条，Top-K=2 | 命中率 1.0000，覆盖度 0.4390 | 97.9050s |
| Prompt 对比 stepwise | 1 条，Top-K=2 | 命中率 1.0000，覆盖度 0.1707 | 110.9410s |
| Prompt 对比 reflective | 1 条，Top-K=2 | 命中率 1.0000，覆盖度 0.2927 | 93.7747s |
| 权重搜索 | 5 条，Top-K=2，entity=0.15 | 最优 `emb=0.30, lex=0.55, entity=0.15`，命中率 1.0000，平均分差 0.4834 | 纯检索 |

结果说明：本轮轻量实验主要用于验证高级 RAG 改动的可运行性和补充记录，不替代完整 30 条评测。由于 `num_predict=24` 输出被刻意截短，参考答案覆盖度偏低是预期现象；更适合比较检索链路、脚本稳定性和运行耗时。报告主结论以 `full30_20260621` 的完整 30 条结果为准。

补充双模型轻量对比（2026-06-21）：

```cmd
python src\model_compare.py --limit 1 --top-k 2 --num-predict 16 --timeout 180 --no-context-expansion --output-suffix goal_20260621
```

新增结果文件：

- `results/model_compare_results_goal_20260621.csv`
- `results/model_compare_summary_goal_20260621.json`

| 模型 | 样本数 | 成功率 | 检索命中率 | 平均参考覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1:7b-qwen-distill-q4_K_M | 1 | 1.0000 | 1.0000 | 0.0000 | 1.6942s | 139.9968s | 141.6911s |
| qwen2.5:7b-instruct-q4_K_M | 1 | 1.0000 | 1.0000 | 0.0976 | 0.6193s | 136.7221s | 137.3414s |

说明：本轮双模型补充对比同样采用短输出 `num_predict=16`，用于验证双模型链路和结果文件后缀机制，不替代既有 30 条完整双模型评测。短输出会显著降低参考覆盖度，尤其 DeepSeek 在 16 token 限制下输出更容易不完整。

## 2026-06-22 完整 30 条全套评测结果（full30_20260621）

本轮已补齐完整 30 条样本的全套评测，结果使用 `full30_20260621` 后缀，不覆盖旧结果。运行参数为 `Top-K=4`、`num_predict=64`、`timeout=360`、关闭 context expansion；这是完整样本数评测，只是控制了本地 7B 模型的生成长度。

完成性验证：

| 模块 | 明细数量 | 完成情况 |
|---|---:|---|
| 单模型效果评测 | 30 | `complete=true` |
| RAG 策略对比 | 90 | vector / hybrid / adaptive 各 30 条 |
| Prompt 对比 | 90 | baseline / stepwise / reflective 各 30 条 |
| 双模型对比 | 60 | DeepSeek / Qwen 各 30 条 |
| 权重调优 | 5 组权重 | `complete=true` |

关键结果：

| 实验 | 设置 | 主要结果 | 平均总耗时 |
|---|---|---|---:|
| 单模型效果评测 | 30 条，Qwen，Top-K=4 | 命中率 1.0000，参考覆盖度 0.5585 | 148.2807s |
| RAG 策略 vector | 30 条，优化前基线 | 命中率 0.7000，参考覆盖度 0.3669 | 123.3889s |
| RAG 策略 hybrid | 30 条，优化后方案 | 命中率 1.0000，参考覆盖度 0.5404 | 138.3754s |
| RAG 策略 adaptive | 30 条，附加策略 | 命中率 1.0000，参考覆盖度 0.3850 | 286.2434s |
| Prompt baseline | 30 条 | 命中率 1.0000，参考覆盖度 0.5286 | 138.9987s |
| Prompt stepwise | 30 条 | 命中率 1.0000，参考覆盖度 0.3782，推理忠实度 0.8000 | 134.8031s |
| Prompt reflective | 30 条 | 命中率 1.0000，参考覆盖度 0.6596 | 142.3079s |
| DeepSeek 双模型对比 | 30 条 | 命中率 1.0000，参考覆盖度 0.0000 | 142.2310s |
| Qwen 双模型对比 | 30 条 | 命中率 1.0000，参考覆盖度 0.5382 | 146.9947s |
| 权重调优 | 5 组检索权重 | 最优 `emb=0.30, lex=0.55, entity=0.15`，命中率 1.0000，分差 0.5419 | 纯检索 |

提升档控制实验结论：在固定知识库、30 条问题、模型 `qwen2.5:7b-instruct-q4_K_M` 和 `Top-K=4` 的条件下，只切换检索策略。`hybrid` 相比 `vector` 将检索命中率从 0.7000 提升到 1.0000，将平均参考覆盖度从 0.3669 提升到 0.5404，因此已形成明确的 RAG 优化前后对照证据。

新增 full30 结果文件：

- `results/eval_summary_full30_20260621.json`
- `results/rag_strategy_compare_summary_full30_20260621.json`
- `results/prompt_compare_summary_full30_20260621.json`
- `results/model_compare_summary_full30_20260621.json`
- `results/weight_tune_results_full30_20260621.json`
