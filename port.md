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
