# 基于 Ollama 的本地大模型部署与计算机专业文档问答应用开发

本项目是一个面向课程设计的本地 RAG（Retrieval-Augmented Generation，检索增强生成）专业文档问答系统。系统基于本地 Ollama 服务运行大模型和 Embedding 模型，读取 `data/` 目录下的计算机专业资料，构建可追溯知识库索引，并在用户提问时完成“检索相关片段 -> 组织 Prompt -> 本地模型生成答案 -> 返回引用来源和耗时统计”的完整问答链路。

项目没有完全照搬参考文档中的 LangChain/ChromaDB 示例，而是保留 RAG 的关键流程，用 JSON 保存向量索引，并通过 `scikit-learn` 完成余弦相似度和字符 n-gram TF-IDF 相似度计算。这样依赖更轻，便于解释代码、复现实验和在报告中说明每个技术环节。

## 项目完成度

- 基础功能：已完成 Ollama 本地部署、文档解析、文本切分、Embedding、检索、Prompt 拼接、本地模型生成和 Streamlit 页面。
- 进阶评测：已提供 30 条评测问题和批量评测脚本。
- RAG 优化：已补齐”纯向量检索 vs 混合检索”的严格控制实验。
- 模型对比：已支持 `deepseek-r1:7b-qwen-distill-q4_K_M` 与 `qwen2.5:7b-instruct-q4_K_M` 双模型对比。
- 公开知识库：已采集 Hugging Face CSE 课程 RAG 材料和 arXiv 计算机领域论文摘要，当前公开材料已保存在本地 `data/public_kb/`。
- 代码可读性：`src/` 下核心脚本已补充必要中文注释，主要解释端口配置、文档切分、检索融合、评测控制变量和增量保存逻辑。
- **高级 RAG 改进（已全部实现并验证）**：
  - 三路融合检索：向量 + TF-IDF + 实体匹配，权重和为 1.0（参照 A-RAG / LightRAG）。
  - 语义去重：基于真实 Embedding 向量的凝聚聚类去重（参照 Self-Correcting RAG）。
  - 上下文扩展：检索后追加相邻片段（±1 窗口，参照 A-RAG chunk_read）。
  - 自适应多轮检索：充分性判断 + 必要时扩大二轮检索（参照 SIM-RAG / ReaLM-Retrieve）。
  - 分步推理 / 自反思 Prompt：三种 Prompt 策略可切换（参照 Self-RAG / ReasonRAG）。
  - 多维度评测：忠实度评分、推理引用率、拒答准确率（参照 RAG Survey Gao et al.）。
  - 权重网格搜索：自动寻优三路检索权重（参照 UR2 / RA-DIT）。

## 功能特性

- 本地化运行：生成模型、Embedding 模型和知识库索引均在本地运行，不依赖外部云 API。
- 计算机专业知识库：覆盖计算机网络、操作系统、数据库、人工智能、RAG，以及公开 CSE/arXiv 材料。
- 三路融合混合检索：`0.40 * 向量分 + 0.45 * TF-IDF 关键词分 + 0.15 * 实体匹配分`，兼顾语义召回、关键词匹配和专业术语识别。
- 自适应多轮检索：LLM 判断首轮检索是否充分，不足时自动扩大第二轮检索范围（Top-K 从 4 扩至 8）。
- 上下文扩展：返回检索片段时自动追加相邻片段（±1 个位置），保留段落上下文完整性。
- 语义去重：对检索结果按真实 Embedding 向量做凝聚聚类，过滤高度重复片段，每簇保留得分最高者。
- 三种 Prompt 策略：`baseline`（直接回答）、`stepwise`（四步分步推理，默认）、`reflective`（生成后自我审查）。
- 忠实度自检：可选的 LLM 忠实度评分，检测答案是否有参考资料依据。
- 多轮对话记忆：Streamlit 页面使用 `st.session_state` 保存完整对话历史，支持多轮问答。
- 多维度评测：命中率、参考覆盖度、推理引用率（`[片段N]` 标注）、忠实度评分、拒答准确率。
- 命令行与 Web 双入口：既可通过 CLI 快速问答，也可通过 Streamlit 页面交互。
- 可复现实验：评测结果输出到 `results/`，包含明细 CSV 和汇总 JSON。

## 环境要求

### 1. Ollama 服务

Ollama 通过 Docker Desktop 或本机服务暴露在：

```cmd
http://127.0.0.1:8090
```

如果使用 Docker Desktop，端口映射为：

```text
8090:11434
```

检查服务是否可用：

```cmd
curl http://127.0.0.1:8090/api/tags
```

### 2. Python 环境

建议使用 conda 环境：

```cmd
conda activate nlprag
pip install -r requirements.txt
```

如需重新冻结依赖：

```cmd
pip freeze > requirements.txt
```

### 3. 模型

默认生成模型：

```cmd
ollama pull qwen2.5:7b-instruct-q4_K_M
```

双模型对比模型：

```cmd
ollama pull deepseek-r1:7b-qwen-distill-q4_K_M
```

Embedding 模型：

```cmd
ollama pull nomic-embed-text
```

当前代码中的模型和端口统一配置在 `src/config.py`。

## 数据来源

知识库由两部分组成。

第一部分是项目内自建资料，位于 `data/`：

- `computer_networks.md`：计算机网络基础知识。
- `operating_systems.md`：操作系统基础知识。
- `database_systems.md`：数据库系统基础知识。
- `ai_rag_notes.md`：人工智能与 RAG 基础知识。

第二部分是公开知识库材料，位于 `data/public_kb/`：

- Hugging Face 数据集 `hatakekksheeshh/CSE_course_RAG` 的 CSE/RAG 课程材料。
- Hugging Face 数据集 `CCRss/arXiv_dataset` 的 arXiv 元数据说明。
- arXiv Computer Science 论文摘要和链接。
- `data/public_kb/manifest.json` 记录来源、许可、URL、标题和本地输出路径。

当前本地已经保存公开材料并重建索引；除非需要刷新公开材料或扩大采样规模，否则不必重新运行采集脚本。

## 完整运行流程

### 1. 可选：采集公开知识库材料

默认采集小样本并写入 `data/public_kb/`：

```cmd
python src/collect_public_kb.py
```

快速抽样测试：

```cmd
python src/collect_public_kb.py --max-arxiv 3 --max-cse 3
```

如需额外下载少量论文 PDF：

```cmd
python src/collect_public_kb.py --max-arxiv 20 --download-pdf
```

注意：重新运行采集脚本会覆盖或更新 `data/public_kb/` 及其 `manifest.json`，适合在需要刷新公开材料时使用。

### 2. 构建知识库索引

```cmd
python src/build_kb.py
```

该命令会递归读取 `data/` 中支持的文档类型，生成 Embedding，并写入：

```text
chroma_db/knowledge_index.json
```

当前索引已包含公开材料，最近一次索引状态为：

| 指标 | 数值 |
|---|---:|
| 文档数 | 41 |
| 片段数 | 163 |
| Embedding 模型 | `nomic-embed-text` |
| Ollama 地址 | `http://127.0.0.1:8090` |

重新运行 `python src/build_kb.py` 会覆盖 `chroma_db/knowledge_index.json`。

### 3. 命令行问答

```cmd
python src/rag_chain.py --question "虚拟内存的作用是什么？"
```

支持 `--mode` 和 `--prompt` 参数切换检索策略和 Prompt 策略：

```cmd
python src/rag_chain.py --question "操作系统调度算法有哪些？" --mode adaptive
python src/rag_chain.py --question "TCP 三次握手过程" --mode hybrid --prompt stepwise
python src/rag_chain.py --question "什么是虚拟内存？" --mode hybrid --prompt reflective
```

可用检索模式（`--mode`）：`vector`、`hybrid`（默认）、`adaptive`。
可用 Prompt 策略（`--prompt`）：`baseline`、`stepwise`（默认）、`reflective`。

输出内容包括答案、引用片段及其三路得分（向量分 / 关键词分 / 实体匹配分）、来源文件、检索耗时、生成耗时和总耗时。

### 4. 启动 Web 页面

```cmd
streamlit run src/app.py
```

页面支持：

- 多轮对话历史（`st.session_state`，清空按钮可重置）。
- 侧栏切换检索策略（`vector` / `hybrid` / `adaptive`）、Prompt 策略、Top-K。
- 侧栏可选开启"忠实度自检"（额外一次 LLM 调用，输出 0–10 分）。
- 检索来源展示区：每个片段的综合分、向量分、关键词分和实体匹配分。
- 自适应模式下展示检索轮数和充分性判断结果。

## 评测与实验

### 1. 单模型 30 条样本评测

```cmd
python src/eval.py
```

支持额外参数：

```cmd
python src/eval.py --prompt stepwise              # 使用分步推理 Prompt（默认）
python src/eval.py --faithfulness                  # 开启 LLM 忠实度评分
python src/eval.py --rejection                     # 包含拒答准确率测试
```

输出文件：

- `results/eval_results.csv`（含 `reasoning_faithfulness` 推理引用率列）
- `results/eval_summary.json`

评测样本来自 `eval_questions.csv`，共 30 条，覆盖网络、操作系统、数据库、人工智能和 RAG。

重新完整流程实验结果（2026-05-23）：

| 样本数 | Top-K | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 4 | 1.0000 | 0.6143 | 0.4019s | 135.2647s | 135.6667s |

### 2. RAG 策略优化对照实验

支持三种检索模式对比（含新增的 `adaptive` 自适应模式）：

```cmd
python src/rag_strategy_compare.py                       # 全量 30 条，三种模式
python src/rag_strategy_compare.py --limit 3             # 快速冒烟
python src/rag_strategy_compare.py --modes vector hybrid # 仅对比两种模式
python src/rag_strategy_compare.py --modes hybrid adaptive --limit 5
```

输出文件：

- `results/rag_strategy_compare_results.csv`（含 `retrieval_rounds` 检索轮数列）
- `results/rag_strategy_compare_summary.json`（含 `average_retrieval_rounds`）

三种检索策略说明：

- `vector`：只按 Embedding 向量相似度排序，作为优化前基线。
- `hybrid`：按三路融合权重排序（`0.40 * 向量 + 0.45 * TF-IDF + 0.15 * 实体匹配`），作为优化后方案。
- `adaptive`：在 hybrid 基础上增加充分性判断，不足时自动扩大二轮检索（第二轮 Top-K = 8）。

首次完整实验结果如下：

| 检索策略 | 样本数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 30 | 1.0000 | 0.7667 | 0.4058 | 0.4444s | 137.7124s | 138.1570s |
| hybrid | 30 | 1.0000 | 1.0000 | 0.5948 | 0.2312s | 129.4322s | 129.6635s |

结论：`hybrid` 相比 `vector` 将检索命中率从 `0.7667` 提升到 `1.0000`，平均参考答案覆盖度从 `0.4058` 提升到 `0.5948`，可以作为提升档中 RAG 策略优化的明确控制实验。

重新完整流程实验结果（2026-05-23）如下：

| 检索策略 | 样本数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 30 | 1.0000 | 0.7000 | 0.4076 | 0.4339s | 123.1658s | 123.5998s |
| hybrid | 30 | 1.0000 | 1.0000 | 0.6185 | 0.4084s | 137.9988s | 138.4073s |

复跑结论：`hybrid` 相比 `vector` 仍然明显提升检索命中率和平均参考答案覆盖度。两组数据中的生成耗时存在波动，主要来自本地生成模型非确定性、运行环境负载和 Ollama 推理耗时变化，不影响”同一知识库、同一问题、同一模型、同一 Top-K，只切换检索策略”的控制变量设置。

### 3. Prompt 策略对比实验

对比 `baseline`、`stepwise`、`reflective` 三种 Prompt 策略在同一检索结果下的答案质量差异：

```cmd
python src/prompt_compare.py                                    # 全量 30 条，三种策略
python src/prompt_compare.py --limit 3                          # 快速冒烟
python src/prompt_compare.py --variants stepwise reflective     # 仅对比两种策略
```

输出文件：

- `results/prompt_compare_results.csv`（含 `reasoning_faithfulness` 推理引用率）
- `results/prompt_compare_summary.json`

三种策略说明：

- `baseline`：直接依据参考资料回答，无推理步骤约束。
- `stepwise`（推荐）：要求模型分四步推理：① 识别信息线索 → ② 提取关键事实 → ③ 综合推导 → ④ 给出结论并引用片段编号。
- `reflective`：生成初步答案后自我审查，识别可能的错误或不完整之处并修正输出。

### 4. 检索权重网格搜索

在三路融合权重空间中自动寻找最优 `EMBEDDING_SCORE_WEIGHT_V3 / LEXICAL_SCORE_WEIGHT_V3 / ENTITY_SCORE_WEIGHT` 组合：

```cmd
python src/weight_tune.py --limit 10                  # 快速评测（默认 10 题）
python src/weight_tune.py --entity-fixed 0.15         # 固定实体权重，只搜索向量/词汇比例
```

搜索空间（三者之和须等于 1.0）：

- 向量权重 `emb`：0.30 / 0.35 / 0.40 / 0.45 / 0.50
- 词汇权重 `lex`：0.35 / 0.40 / 0.45 / 0.50 / 0.55
- 实体权重 `entity`：0.05 / 0.10 / 0.15 / 0.20

输出文件：

- `results/weight_tune_results.json`（含 `best` 最优组合和 `all` 全量结果）

### 5. 双模型对比实验

完整实验：

```cmd
python src/model_compare.py
```

快速抽样：

```cmd
python src/model_compare.py --limit 5
```

输出文件：

- `results/model_compare_results.csv`
- `results/model_compare_summary.json`
- `results/model_compare_run.out`

首次完整双模型评测摘要如下：

| 模型 | 样本数 | 成功数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| `deepseek-r1:7b-qwen-distill-q4_K_M` | 30 | 30 | 1.0 | 1.0 | 0.4324 | 171.2024s |
| `qwen2.5:7b-instruct-q4_K_M` | 30 | 30 | 1.0 | 1.0 | 0.5746 | 143.3015s |

结论：两种模型都能完成问答流程，Qwen 在本次评测中的平均参考答案覆盖度更高、平均总耗时更低，可作为加分挑战中的模型对比材料。

重新完整流程实验结果（2026-05-23）如下：

| 模型 | 样本数 | 成功数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `deepseek-r1:7b-qwen-distill-q4_K_M` | 30 | 30 | 1.0000 | 1.0000 | 0.4765 | 0.4563s | 165.0030s | 165.4594s |
| `qwen2.5:7b-instruct-q4_K_M` | 30 | 30 | 1.0000 | 1.0000 | 0.6205 | 0.4582s | 138.7100s | 139.1683s |

复跑结论：Qwen 在重新完整流程实验中仍然取得更高的平均参考答案覆盖度和更低的平均总耗时。两次实验的具体数值存在小幅差异，属于本地模型生成和机器负载波动下的正常现象。

## 项目结构

```text
data/                         自建计算机专业知识库文档
data/public_kb/               公开 CSE 数据集与 arXiv 论文摘要采集结果
chroma_db/                    保存 knowledge_index.json；目录名沿用课程资料习惯
results/                      评测明细、汇总和运行日志
src/config.py                 模型、端口、路径、切分和检索参数（含三路融合权重和自适应参数）
src/collect_public_kb.py      公开知识库轻量采集脚本
src/build_kb.py               文档解析、文本切分、Embedding 与索引构建
src/rag_chain.py              检索（三路融合/语义去重/上下文扩展）、多种 Prompt 策略、Ollama 生成
src/app.py                    Streamlit 多轮对话页面（含自适应模式和忠实度自检）
src/eval.py                   30 条样本批量评测（含忠实度评分和拒答准确率）
src/model_compare.py          DeepSeek R1 与 Qwen 双模型对比
src/rag_strategy_compare.py   vector / hybrid / adaptive 三种检索策略控制实验
src/prompt_compare.py         baseline / stepwise / reflective 三种 Prompt 策略对比
src/weight_tune.py            三路融合检索权重网格搜索调优
eval_questions.csv            评测问题、参考答案和期望命中文档
doc-formal.txt                课程设计正式报告草稿
Task.md                       课程设计完成情况核对
dev.md                        RAG 高级改进方案设计与实施说明
port.md                       端口、环境和模型配置说明
RAG流程实现说明.md            RAG 核心链路说明（含高级改进技术详解）
```

## 核心设计说明

### 文档解析

`src/build_kb.py` 支持读取 `.txt`、`.md`、`.markdown` 和 `.pdf` 文件。普通文本和 Markdown 会按 UTF-8、UTF-8-SIG、GB18030 顺序尝试读取；PDF 使用 `pypdf` 提取页文本并保留页码信息。

### 文本切分

默认切分参数位于 `src/config.py`：

```python
CHUNK_SIZE = 420
CHUNK_OVERLAP = 80
```

切分时优先在段落、换行、句号、分号、逗号和空格等自然边界处截断，减少专业概念被切碎的概率。

### 向量索引

项目保留 `chroma_db/` 目录名，但没有调用 ChromaDB 客户端。索引实际保存为：

```text
chroma_db/knowledge_index.json
```

这种设计适合课程设计和中小规模材料演示，优点是透明、便于检查和讲解；如果后续扩展到更大规模文档，可迁移到 ChromaDB、FAISS 或 Milvus。

### 检索策略

系统支持三种检索模式（`ANSWER_MODES`）：

- `vector`：只使用问题向量与片段向量的余弦相似度。
- `hybrid`（默认）：三路融合——向量相似度 + 字符 n-gram TF-IDF 相似度 + 实体匹配得分。
- `adaptive`：在 hybrid 基础上增加 LLM 充分性判断，不足时自动扩大第二轮检索。

当前三路融合默认权重（V3 权重，和为 1.0）：

```python
EMBEDDING_SCORE_WEIGHT_V3 = 0.40   # 向量语义得分
LEXICAL_SCORE_WEIGHT_V3   = 0.45   # TF-IDF 字符 n-gram 词汇得分
ENTITY_SCORE_WEIGHT       = 0.15   # 专业实体词匹配得分
```

实体匹配使用正则提取问题中的英文技术词、中文双字词和数字，计算在片段中出现的比例，增强对专业术语查询的检索精度。

历史兼容权重（仅 vector/hybrid 无实体匹配时使用）：

```python
EMBEDDING_SCORE_WEIGHT = 0.45
LEXICAL_SCORE_WEIGHT   = 0.55
```

自适应检索参数：

```python
ADAPTIVE_RETRIEVAL    = True
ADAPTIVE_TOP_K_SECOND = 8     # 第二轮扩大检索的 Top-K
MAX_RETRIEVAL_ROUNDS  = 2
```

### Prompt 策略

系统支持三种问答 Prompt：

- **`baseline`**：直接基于参考片段回答，资料不足时拒答，并列出片段来源。
- **`stepwise`**（推荐）：强制四步推理——步骤 1 识别信息线索、步骤 2 提取关键事实、步骤 3 综合推导结论、步骤 4 给出最终答案并引用 `[片段N]` 编号。
- **`reflective`**：先生成初稿，再对照检索片段自我审查是否有不支持的内容，输出修订后答案。

所有策略均要求：仅依据参考资料回答；资料不足时回答”知识库中没有找到明确答案”；引用时标注片段编号和来源文件名。

### 语义去重

`deduplicate_by_similarity()` 对当前候选片段做凝聚聚类（`AgglomerativeClustering`，预计算 1 - 余弦相似度矩阵），相似度 ≥ 0.85 的片段归为同一簇，每簇保留综合得分最高的片段。此步骤消除检索结果中高度重复的冗余内容，避免同一段话在不同切分边界下被计入多次。

### 上下文扩展

`expand_context()` 在检索完成后，对每个返回片段查找其在原始索引中的位置，追加相邻 ±`EXPANSION_WINDOW`（默认 1）个片段（分数设为 0.0 标记为扩展）。扩展片段仅在上下文中追加，不参与排序。

```python
CONTEXT_EXPANSION  = True
EXPANSION_WINDOW   = 1
```

### 忠实度自检

`check_faithfulness()` 调用 LLM 对生成答案和参考片段做比对，判断答案中的主要声明是否有片段依据，并输出 0–10 分忠实度评分。此功能默认关闭（`ENABLE_FAITHFULNESS_CHECK = False`），可在 Streamlit 侧栏或 `eval.py --faithfulness` 中按需开启。

## 常用命令汇总

```cmd
conda activate nlprag
curl http://127.0.0.1:8090/api/tags
python src/collect_public_kb.py
python src/build_kb.py
python src/rag_chain.py --question "RAG 的基本流程是什么？"
python src/rag_chain.py --question "进程和线程的区别？" --mode adaptive --prompt stepwise
streamlit run src/app.py
python src/eval.py --prompt stepwise
python src/eval.py --faithfulness --rejection
python src/rag_strategy_compare.py
python src/rag_strategy_compare.py --modes vector hybrid adaptive --limit 5
python src/prompt_compare.py --limit 5
python src/weight_tune.py --entity-fixed 0.15 --limit 10
python src/model_compare.py
```

语法检查：

```cmd
python -m py_compile src\config.py src\collect_public_kb.py src\build_kb.py src\rag_chain.py src\eval.py src\model_compare.py src\rag_strategy_compare.py src\prompt_compare.py src\weight_tune.py src\app.py
```

## 重新运行时是否会覆盖

- `python src/collect_public_kb.py`：会更新 `data/public_kb/` 和 `data/public_kb/manifest.json`。
- `python src/build_kb.py`：会重新生成并覆盖 `chroma_db/knowledge_index.json`。
- `python src/eval.py`：会写入或覆盖 `results/eval_results.csv` 和 `results/eval_summary.json`。
- `python src/rag_strategy_compare.py`：会写入或覆盖 `results/rag_strategy_compare_results.csv` 和 `results/rag_strategy_compare_summary.json`。
- `python src/prompt_compare.py`：会写入或覆盖 `results/prompt_compare_results.csv` 和 `results/prompt_compare_summary.json`。
- `python src/weight_tune.py`：会写入或覆盖 `results/weight_tune_results.json`。
- `python src/model_compare.py`：会写入或覆盖 `results/model_compare_results.csv` 和 `results/model_compare_summary.json`。

如果只是演示或答辩，当前本地材料和索引已经可用，可以直接运行问答、Web 页面或评测脚本；如果需要刷新公开知识库或扩大样本，再从采集脚本开始重新跑完整流程。

## 常见问题

### 1. Ollama 连接失败

先检查端口是否正确：

```cmd
curl http://127.0.0.1:8090/api/tags
```

如果返回失败，确认 Docker Desktop 或本机 Ollama 服务是否启动，以及端口映射是否为 `8090:11434`。

### 2. 模型不存在

运行：

```cmd
ollama list
```

确认是否存在：

- `qwen2.5:7b-instruct-q4_K_M`
- `deepseek-r1:7b-qwen-distill-q4_K_M`
- `nomic-embed-text`

缺失时使用 `ollama pull` 拉取。

### 3. 知识库索引不存在

运行：

```cmd
python src/build_kb.py
```

生成 `chroma_db/knowledge_index.json` 后再进行问答或评测。

### 4. 生成速度慢

本项目使用本地 7B 量化模型，生成速度与 CPU、内存、显卡和 Ollama 后端配置有关。可以减少 Top-K、降低 `NUM_PREDICT`，或临时改用更小模型进行演示。

## 报告材料对应关系

- 任务背景、技术路线、系统设计、实验分析：见 `doc-formal.txt`。
- RAG 三个关键环节的代码体现：见 `RAG流程实现说明.md`。
- 评分项完成情况：见 `Task.md`。
- 端口和环境说明：见 `port.md`。
- 完整操作步骤：见 `Ollama_RAG_课程设计完整操作步骤.docx`。

## 参考资料

- DataWhale：动手学 Ollama / handy-ollama。
- Ollama 官方文档。
- LangChain 与 langchain-ollama 文档。
- scikit-learn 文档。
- Streamlit 文档。
- Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Vaswani et al., Attention Is All You Need.
- Reimers and Gurevych, Sentence-BERT.
- Johnson et al., FAISS similarity search.
- Qwen、DeepSeek 相关模型说明。
- Asai et al., Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. (ICLR 2024)
- Ji et al., RAFE/ReasonRAG：步进推理与充分性判断。
- SIM-RAG：充分性判断 + 二轮检索扩展。(arXiv 2024)
- Sarthi et al., RAPTOR / A-RAG：上下文扩展与层次化检索。
- Shi et al., Self-Correcting RAG：基于聚类的语义去重与忠实度纠错。
- Gao et al., Retrieval-Augmented Generation for Large Language Models: A Survey. (arXiv 2024)
- Shi et al., RA-DIT / UR2：双端优化与难度感知权重搜索思想。

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

结果说明：本轮轻量实验主要用于验证高级 RAG 改动的可运行性和补充记录，不替代完整评测。由于 `num_predict=24` 输出被刻意截短，参考答案覆盖度偏低是预期现象；更适合比较检索链路、脚本稳定性和运行耗时。报告主结论以 `full30_20260621` 的完整 30 条结果为准。

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

本轮在同一知识库、同一 30 条评测问题和同一 `Top-K=4` 条件下完成全套评测，结果文件统一使用 `full30_20260621` 后缀，未覆盖历史结果。由于本机 Docker/Ollama 使用本地 7B 量化模型推理，完整评测采用受控生成参数 `num_predict=64`、`timeout=360`，并关闭 context expansion；轻量的是生成长度控制，样本数仍为完整 30 条。

执行命令：

```cmd
python src\full30_runner.py --tasks eval --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
python src\full30_runner.py --tasks strategy --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
python src\full30_runner.py --tasks prompt --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
python src\full30_runner.py --tasks model --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
python src\full30_runner.py --tasks weight --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
```

结果文件：

- `results/eval_results_full30_20260621.csv`，`results/eval_summary_full30_20260621.json`
- `results/rag_strategy_compare_results_full30_20260621.csv`，`results/rag_strategy_compare_summary_full30_20260621.json`
- `results/prompt_compare_results_full30_20260621.csv`，`results/prompt_compare_summary_full30_20260621.json`
- `results/model_compare_results_full30_20260621.csv`，`results/model_compare_summary_full30_20260621.json`
- `results/weight_tune_results_full30_20260621.json`

完成性验证：

| 模块 | 明细数量 | 完成情况 |
|---|---:|---|
| 单模型效果评测 | 30 | `complete=true` |
| RAG 策略对比 | 90 | vector / hybrid / adaptive 各 30 条 |
| Prompt 对比 | 90 | baseline / stepwise / reflective 各 30 条 |
| 双模型对比 | 60 | DeepSeek / Qwen 各 30 条 |
| 权重调优 | 5 组权重 | `complete=true` |

单模型效果评测：

| 样本数 | 模型 | 检索命中率 | 平均参考覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---:|---|---:|---:|---:|---:|---:|
| 30 | qwen2.5:7b-instruct-q4_K_M | 1.0000 | 0.5585 | 0.7879s | 147.4923s | 148.2807s |

RAG 策略对比：

| 检索策略 | 样本数 | 检索命中率 | 平均参考覆盖度 | 平均检索轮数 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 30 | 0.7000 | 0.3669 | 1.0000 | 0.6956s | 122.6932s | 123.3889s |
| hybrid | 30 | 1.0000 | 0.5404 | 1.0000 | 0.7362s | 137.6391s | 138.3754s |
| adaptive | 30 | 1.0000 | 0.3850 | 1.0000 | 124.5705s | 161.5900s | 286.2434s |

该对比中 `vector` 是优化前基线，只按 Embedding 向量相似度排序；`hybrid` 是当前优化后方案，使用三路融合权重 `0.40 * 向量分 + 0.45 * TF-IDF 关键词分 + 0.15 * 实体匹配分`。在完整 30 条样本上，hybrid 将检索命中率从 0.7000 提升到 1.0000，平均参考覆盖度从 0.3669 提升到 0.5404，构成明确的 RAG 策略优化前后控制实验。

代码说明补充：本轮还为 `src/config.py`、`src/build_kb.py`、`src/collect_public_kb.py`、`src/rag_chain.py`、`src/eval.py`、`src/rag_strategy_compare.py`、`src/model_compare.py`、`src/prompt_compare.py`、`src/weight_tune.py`、`src/full30_runner.py` 和 `src/app.py` 增加了必要中文注释。注释集中说明实现取舍，不解释简单赋值和普通函数调用。

Prompt 对比：

| Prompt 变体 | 样本数 | 检索命中率 | 平均参考覆盖度 | 推理忠实度 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 30 | 1.0000 | 0.5286 | 0.0000 | 138.3741s | 138.9987s |
| stepwise | 30 | 1.0000 | 0.3782 | 0.8000 | 134.2031s | 134.8031s |
| reflective | 30 | 1.0000 | 0.6596 | 0.0667 | 141.7036s | 142.3079s |

双模型对比：

| 模型 | 样本数 | 成功率 | 检索命中率 | 平均参考覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1:7b-qwen-distill-q4_K_M | 30 | 1.0000 | 1.0000 | 0.0000 | 0.6406s | 141.5903s | 142.2310s |
| qwen2.5:7b-instruct-q4_K_M | 30 | 1.0000 | 1.0000 | 0.5382 | 0.6562s | 146.3384s | 146.9947s |

权重调优结果：

| emb | lex | entity | 命中率 | 平均分差 |
|---:|---:|---:|---:|---:|
| 0.30 | 0.55 | 0.15 | 1.0000 | 0.5419 |
| 0.35 | 0.50 | 0.15 | 1.0000 | 0.5026 |
| 0.40 | 0.45 | 0.15 | 1.0000 | 0.4626 |
| 0.45 | 0.40 | 0.15 | 1.0000 | 0.4191 |
| 0.50 | 0.35 | 0.15 | 1.0000 | 0.3747 |

最优权重为 `emb=0.30, lex=0.55, entity=0.15`，在命中率相同为 1.0000 的情况下平均分差最高，说明关键词项在当前计算机专业中文问答集上对正确片段排序有明显帮助。
