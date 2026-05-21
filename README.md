# 基于 Ollama 的本地大模型部署与计算机专业文档问答应用开发

本项目实现一个本地 RAG 专业文档问答系统。系统读取 `data/` 中的计算机专业文档，调用本地 Ollama Embedding 模型生成向量索引，用户提问时使用 scikit-learn 计算向量相似度和字符 n-gram TF-IDF 相似度，再把混合检索得到的上下文交给 Ollama 生成模型回答。

## 环境要求

- Ollama 通过 Docker Desktop 或本机服务暴露在 `http://localhost:8090`
- conda 环境：`nlprag`
- 生成模型：`qwen2.5:7b`
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
ollama pull qwen2.5:7b
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
eval_questions.csv    评测问题与参考答案
doc-formal.txt        课程设计报告草稿
```

## 设计说明

本实现没有照搬参考文档中的 ChromaDB 流程，而是用 JSON 保存向量索引，并用 `scikit-learn` 的 `cosine_similarity` 完成检索。检索分数由 Ollama Embedding 相似度和字符级 TF-IDF 相似度加权得到，兼顾语义召回与中文专业术语匹配。这样依赖更少，便于解释课程设计中的文档解析、切分、向量化、检索和生成链路。
