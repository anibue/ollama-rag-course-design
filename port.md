# 端口、环境与模型配置说明

## Ollama 服务地址

本项目统一使用以下 Ollama API 地址：

```cmd
http://127.0.0.1:8090
```

检查服务：

```cmd
curl http://127.0.0.1:8090/api/tags
```

如使用 Docker Desktop（Windows），端口映射为：

```text
8090:11434
```

代码中的统一配置位置：

```text
src/config.py
```

其中：

```python
OLLAMA_BASE_URL = "http://127.0.0.1:8090"
```

## Python 环境

使用 conda 包管理，环境名：

```text
nlprag
```

激活环境：

```cmd
conda activate nlprag
```

安装依赖：

```cmd
pip install -r requirements.txt
```

如需更新依赖清单：

```cmd
pip freeze > requirements.txt
```

## 模型配置

默认生成模型：

```text
qwen2.5:7b-instruct-q4_K_M
```

双模型对比模型：

```text
deepseek-r1:7b-qwen-distill-q4_K_M
```

Embedding 模型：

```text
nomic-embed-text
```

拉取命令：

```cmd
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull deepseek-r1:7b-qwen-distill-q4_K_M
ollama pull nomic-embed-text
```

## 当前索引状态

当前本地知识库索引：

```text
chroma_db/knowledge_index.json
```

最近一次重建后包含：

| 指标 | 数值 |
|---|---:|
| 文档数 | 41 |
| 片段数 | 163 |
| Ollama 地址 | `http://127.0.0.1:8090` |
| Embedding 模型 | `nomic-embed-text` |

重新运行以下命令会覆盖索引：

```cmd
python src/build_kb.py
```

## 当前 RAG 参数口径

当前报告和 full30 结果采用以下口径：

| 参数 | 当前值 |
|---|---|
| 默认 Top-K | `4` |
| 默认检索模式 | `hybrid` |
| hybrid 权重 | `0.40 * 向量分 + 0.45 * TF-IDF 关键词分 + 0.15 * 实体匹配分` |
| Prompt 策略 | `baseline`、`stepwise`、`reflective` |
| full30 生成长度 | `num_predict=64` |
| full30 超时 | `timeout=360` |
| full30 context expansion | 关闭 |

旧版二路权重 `0.45 * 向量分 + 0.55 * TF-IDF 分` 只作为历史实现记录；当前 full30 结果按三路融合解释。

## 公开知识库材料

公开材料已保存在：

```text
data/public_kb/
```

来源说明和本地文件清单保存在：

```text
data/public_kb/manifest.json
```

如需刷新公开材料：

```cmd
python src/collect_public_kb.py
python src/build_kb.py
```

## 源码注释状态

`src/` 下核心脚本已补充必要中文注释，主要覆盖：

- `src/config.py`：端口、代理、模型和检索参数。
- `src/build_kb.py`、`src/collect_public_kb.py`：公开材料采集、文本读取、切分和 JSON 索引。
- `src/rag_chain.py`：三路融合检索、语义去重、上下文扩展、adaptive 检索和 Ollama 生成。
- `src/eval.py`、`src/rag_strategy_compare.py`、`src/prompt_compare.py`、`src/model_compare.py`、`src/weight_tune.py`、`src/full30_runner.py`：评测指标、控制变量和增量写盘。
- `src/app.py`：Web 页面参数、对话状态和来源片段展示。
