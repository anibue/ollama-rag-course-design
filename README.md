# 基于 Ollama 的本地大模型部署与计算机专业文档问答应用开发

本项目实现一个本地 RAG 专业文档问答系统。系统读取 `data/` 中的计算机专业文档，调用本地 Ollama Embedding 模型生成向量索引，用户提问时使用 scikit-learn 计算向量相似度和字符 n-gram TF-IDF 相似度，再把混合检索得到的上下文交给 Ollama 生成模型回答。

## 环境要求

- Ollama 通过 Docker Desktop 或本机服务暴露在 `http://localhost:8090`
- conda 环境：`nlprag`
- 生成模型：`qwen2.5:7b-instruct-q4_K_M`
- Embedding 模型：`nomic-embed-text`

检查 Ollama 服务：

```cmd
curl http://localhost:8090/api/tags
```

安装依赖：

```cmd
conda activate nlprag
pip install -r requirements.txt
pip freeze > requirements.txt
```

拉取模型：

```cmd
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull deepseek-r1:7b-qwen-distill-q4_K_M
ollama pull nomic-embed-text
```

## 运行步骤

构建知识库：

```cmd
python src/build_kb.py
```

命令行问答：

```cmd
python src/rag_chain.py --question "虚拟内存的作用是什么？"
```

启动 Web 页面：

```cmd
streamlit run src/app.py
```

批量评测 30 条样本：

```cmd
python src/eval.py
```

DeepSeek R1 与 Qwen 双模型对比：

```cmd
python src/model_compare.py
```

快速抽样对比：

```cmd
python src/model_compare.py --limit 5
```

Model compare output: `results/model_compare_results.csv` and `results/model_compare_summary.json`.

RAG 检索策略优化对比：

```cmd
python src/rag_strategy_compare.py
```

快速抽样对比：

```cmd
python src/rag_strategy_compare.py --limit 3
```

Strategy compare output: `results/rag_strategy_compare_results.csv` and `results/rag_strategy_compare_summary.json`.

完整对照实验使用同一知识库、同一 30 条问题、同一模型 `qwen2.5:7b-instruct-q4_K_M` 和同一 `Top-K=4`，只切换检索策略：

| 检索策略 | 样本数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 30 | 1.0000 | 0.7667 | 0.4058 | 0.4444s | 137.7124s | 138.1570s |
| hybrid | 30 | 1.0000 | 1.0000 | 0.5948 | 0.2312s | 129.4322s | 129.6635s |

评测结果会写入 `results/eval_results.csv` 和 `results/eval_summary.json`。

## 目录说明

```text
data/                 计算机专业知识库文档
chroma_db/            本项目沿用目录名，保存 knowledge_index.json
results/              评测输出
src/config.py         模型、端口、路径与 RAG 参数
src/build_kb.py       文档解析、切分、Embedding 与索引构建
src/rag_chain.py      检索、Prompt 拼接与 Ollama 生成
src/app.py            Streamlit 问答页面
src/eval.py           30 条样本批量评测
src/rag_strategy_compare.py  纯向量检索与混合检索控制实验
eval_questions.csv    评测问题与参考答案
doc-formal.txt        课程设计报告草稿
```

## 设计说明

本实现没有照搬参考文档中的 ChromaDB 流程，而是用 JSON 保存向量索引，并用 `scikit-learn` 的 `cosine_similarity` 完成检索。系统保留 `vector` 纯向量检索作为优化前基线，并以 `hybrid` 混合检索作为默认策略；混合检索分数由 Ollama Embedding 相似度和字符级 TF-IDF 相似度加权得到，兼顾语义召回与中文专业术语匹配。这样依赖更少，便于解释课程设计中的文档解析、切分、向量化、检索和生成链路。
