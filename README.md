# 基于 Ollama 的本地大模型部署与计算机专业文档问答应用开发

本项目是一个面向课程设计的本地 RAG（Retrieval-Augmented Generation，检索增强生成）专业文档问答系统。系统基于本地 Ollama 服务运行大模型和 Embedding 模型，读取 `data/` 目录下的计算机专业资料，构建可追溯知识库索引，并在用户提问时完成“检索相关片段 -> 组织 Prompt -> 本地模型生成答案 -> 返回引用来源和耗时统计”的完整问答链路。

项目没有完全照搬参考文档中的 LangChain/ChromaDB 示例，而是保留 RAG 的关键流程，用 JSON 保存向量索引，并通过 `scikit-learn` 完成余弦相似度和字符 n-gram TF-IDF 相似度计算。这样依赖更轻，便于解释代码、复现实验和在报告中说明每个技术环节。

## 项目完成度

- 基础功能：已完成 Ollama 本地部署、文档解析、文本切分、Embedding、检索、Prompt 拼接、本地模型生成和 Streamlit 页面。
- 进阶评测：已提供 30 条评测问题和批量评测脚本。
- RAG 优化：已补齐“纯向量检索 vs 混合检索”的严格控制实验。
- 模型对比：已支持 `deepseek-r1:7b-qwen-distill-q4_K_M` 与 `qwen2.5:7b-instruct-q4_K_M` 双模型对比。
- 公开知识库：已采集 Hugging Face CSE 课程 RAG 材料和 arXiv 计算机领域论文摘要，当前公开材料已保存在本地 `data/public_kb/`。

## 功能特性

- 本地化运行：生成模型、Embedding 模型和知识库索引均在本地运行，不依赖外部云 API。
- 计算机专业知识库：覆盖计算机网络、操作系统、数据库、人工智能、RAG，以及公开 CSE/arXiv 材料。
- 混合检索：默认使用 `0.45 * 向量分 + 0.55 * TF-IDF 关键词分`，兼顾语义召回和中文专业术语匹配。
- 严格引用 Prompt：模型必须基于检索片段回答，资料不足时要求拒答，降低幻觉风险。
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

输出内容包括答案、引用片段、来源文件、相似度分数、检索耗时、生成耗时和总耗时。

### 4. 启动 Web 页面

```cmd
streamlit run src/app.py
```

页面支持输入问题、调整 Top-K，并展示回答、引用来源和运行耗时。

## 评测与实验

### 1. 单模型 30 条样本评测

```cmd
python src/eval.py
```

输出文件：

- `results/eval_results.csv`
- `results/eval_summary.json`

评测样本来自 `eval_questions.csv`，共 30 条，覆盖网络、操作系统、数据库、人工智能和 RAG。

### 2. RAG 策略优化对照实验

完整实验：

```cmd
python src/rag_strategy_compare.py
```

快速冒烟：

```cmd
python src/rag_strategy_compare.py --limit 3
```

可选参数：

```cmd
python src/rag_strategy_compare.py --top-k 4 --modes vector hybrid --model qwen2.5:7b-instruct-q4_K_M
```

输出文件：

- `results/rag_strategy_compare_results.csv`
- `results/rag_strategy_compare_summary.json`

该实验固定同一知识库、同一 30 条问题、同一模型 `qwen2.5:7b-instruct-q4_K_M` 和同一 `Top-K=4`，只切换检索策略：

- `vector`：只按 Embedding 向量相似度排序，作为优化前基线。
- `hybrid`：按 `0.45 * 向量分 + 0.55 * TF-IDF 关键词分` 排序，作为优化后方案。

已完成的完整对照实验结果如下：

| 检索策略 | 样本数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 30 | 1.0000 | 0.7667 | 0.4058 | 0.4444s | 137.7124s | 138.1570s |
| hybrid | 30 | 1.0000 | 1.0000 | 0.5948 | 0.2312s | 129.4322s | 129.6635s |

结论：`hybrid` 相比 `vector` 将检索命中率从 `0.7667` 提升到 `1.0000`，平均参考答案覆盖度从 `0.4058` 提升到 `0.5948`，可以作为提升档中 RAG 策略优化的明确控制实验。

### 3. 双模型对比实验

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

已完成的完整双模型评测摘要如下：

| 模型 | 样本数 | 成功数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| `deepseek-r1:7b-qwen-distill-q4_K_M` | 30 | 30 | 1.0 | 1.0 | 0.4324 | 171.2024s |
| `qwen2.5:7b-instruct-q4_K_M` | 30 | 30 | 1.0 | 1.0 | 0.5746 | 143.3015s |

结论：两种模型都能完成问答流程，Qwen 在本次评测中的平均参考答案覆盖度更高、平均总耗时更低，可作为加分挑战中的模型对比材料。

## 项目结构

```text
data/                         自建计算机专业知识库文档
data/public_kb/               公开 CSE 数据集与 arXiv 论文摘要采集结果
chroma_db/                    保存 knowledge_index.json；目录名沿用课程资料习惯
results/                      评测明细、汇总和运行日志
src/config.py                 模型、端口、路径、切分和检索参数
src/collect_public_kb.py      公开知识库轻量采集脚本
src/build_kb.py               文档解析、文本切分、Embedding 与索引构建
src/rag_chain.py              检索、Prompt 拼接与 Ollama 生成
src/app.py                    Streamlit 问答页面
src/eval.py                   30 条样本批量评测
src/model_compare.py          DeepSeek R1 与 Qwen 双模型对比
src/rag_strategy_compare.py   纯向量检索与混合检索控制实验
eval_questions.csv            评测问题、参考答案和期望命中文档
doc-formal.txt                课程设计正式报告草稿
Task.md                       课程设计完成情况核对
port.md                       端口、环境和模型配置说明
RAG流程实现说明.md            RAG 核心链路说明
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

系统支持两种检索模式：

- `vector`：只使用问题向量与片段向量的余弦相似度。
- `hybrid`：融合向量相似度和字符 n-gram TF-IDF 相似度。

默认使用 `hybrid`，权重配置如下：

```python
EMBEDDING_SCORE_WEIGHT = 0.45
LEXICAL_SCORE_WEIGHT = 0.55
TOP_K = 4
```

### Prompt 约束

问答 Prompt 明确要求：

- 只依据参考资料回答。
- 资料没有明确依据时回答“知识库中没有找到明确答案”。
- 输出答案时列出引用片段编号和文件名。

这部分用于降低幻觉风险，并让最终答案具备可追溯来源。

## 常用命令汇总

```cmd
conda activate nlprag
curl http://127.0.0.1:8090/api/tags
python src/collect_public_kb.py
python src/build_kb.py
python src/rag_chain.py --question "RAG 的基本流程是什么？"
streamlit run src/app.py
python src/eval.py
python src/rag_strategy_compare.py
python src/model_compare.py
```

语法检查：

```cmd
python -m py_compile src\config.py src\collect_public_kb.py src\build_kb.py src\rag_chain.py src\eval.py src\model_compare.py src\rag_strategy_compare.py src\app.py
```

## 重新运行时是否会覆盖

- `python src/collect_public_kb.py`：会更新 `data/public_kb/` 和 `data/public_kb/manifest.json`。
- `python src/build_kb.py`：会重新生成并覆盖 `chroma_db/knowledge_index.json`。
- `python src/eval.py`：会写入或覆盖 `results/eval_results.csv` 和 `results/eval_summary.json`。
- `python src/rag_strategy_compare.py`：会写入或覆盖 `results/rag_strategy_compare_results.csv` 和 `results/rag_strategy_compare_summary.json`。
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
