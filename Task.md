# 课程设计任务完成情况核对

核对时间：2026-05-22  
项目主题：基于 Ollama 的本地大模型部署与计算机专业文档问答 RAG 应用

## 一、总体结论

当前项目已经完成基础档、进阶层和提升档的主要要求，并且具备加分挑战所需的模型对比材料；同时已补充公开数据集和计算机领域论文材料作为可追溯原始知识库来源。

按现有文件证据判断：

- 基础档（60-69 分）：已完成。
- 进阶层（70-79 分）：已完成主要要求。
- 提升档（80-89 分）：已完成；已有 Prompt 约束、混合检索，并补齐了“纯向量检索 vs 混合检索”的优化前后控制实验。
- 加分挑战（+10 分）：已具备“模型对比”实现和完整双模型评测结果，可作为挑战任务材料。

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
- 当前基础知识库包含 4 份自建文档；公开材料可通过 `python src/collect_public_kb.py` 写入 `data/public_kb/` 后重新构建索引。

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
- 混合检索加权策略：`0.45 * 向量分 + 0.55 * 关键词分`。
- Prompt 约束生成。
- 检索命中率、参考答案覆盖度、响应时间等指标说明。

进阶层结论：进阶层主要要求已完成。

## 四、提升档核对

要求：在进阶层基础上，选择 Prompt Engineering / 模型组合 / RAG 优化 中的一项进行优化，并且必须有明确的效果对比。

当前状态：已完成。

已具备的优化材料：

- Prompt Engineering：`src/rag_chain.py` 中已有严格依据参考资料、资料不足拒答、输出引用来源的 Prompt 模板。
- RAG 策略优化：当前检索支持 `vector` 纯向量检索和 `hybrid` 混合检索，默认使用 Embedding 相似度 + TF-IDF 关键词相似度的混合检索。
- 模型组合/模型对比材料：`src/model_compare.py` 已支持 DeepSeek R1 与 Qwen 双模型对比。
- 优化控制实验：`src/rag_strategy_compare.py` 已完成 `vector` 与 `hybrid` 的同条件对比。

已有对比结果：

`results/rag_strategy_compare_summary.json` 显示：

| 检索策略 | 样本数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 30 | 1.0000 | 0.7667 | 0.4058 | 0.4444s | 137.7124s | 138.1570s |
| hybrid | 30 | 1.0000 | 1.0000 | 0.5948 | 0.2312s | 129.4322s | 129.6635s |

`results/model_compare_summary.json` 显示：

| 模型 | 样本数 | 成功数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-r1:7b-qwen-distill-q4_K_M | 30 | 30 | 1.0 | 1.0 | 0.4324 | 171.2024s |
| qwen2.5:7b-instruct-q4_K_M | 30 | 30 | 1.0 | 1.0 | 0.5746 | 143.3015s |

效果结论：

- `hybrid` 相比 `vector` 将检索命中率从 0.7667 提升到 1.0000。
- 平均参考答案覆盖度从 0.4058 提升到 0.5948。
- 平均总耗时从 138.1570s 降至 129.6635s。

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

挑战任务结论：

- “模型对比”材料已经完成。
- 提升档中的优化前后对比实验已经补齐，模型对比可作为加分挑战材料。

## 六、当前缺口清单

### 必须注意

1. `results/eval_results.csv` 和 `results/eval_summary.json` 当前不存在。
2. 但 `results/model_compare_results.csv` 已包含双模型各 30 条评测结果，可以替代证明 30 条评测已跑通。
3. 提升档优化前后控制实验已经补齐，结果保存在 `results/rag_strategy_compare_results.csv` 和 `results/rag_strategy_compare_summary.json`。
4. 公开知识库采集脚本已经补充；若要让公开材料进入当前索引，需要先运行 `python src/collect_public_kb.py`，再运行 `python src/build_kb.py`。

### 建议优先补充

1. 运行或补充普通单模型评测输出：

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

综合判断：项目当前可以支撑基础档、进阶层和提升档；双模型评测结果也可作为加分挑战材料。
