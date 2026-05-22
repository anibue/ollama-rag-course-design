# RAG 核心流程实现说明

本文说明当前项目中三个 RAG 关键环节分别体现在哪里：

- 文档解析与文本分割
- 向量嵌入（Embedding）与相似度检索
- 检索结果与生成模型的结合

当前项目实现的是一个本地 Ollama RAG 问答系统。整体链路可以概括为：

```text
data/ 原始文档
  -> src/collect_public_kb.py 可选采集公开 CSE 数据集与 arXiv 论文摘要
  -> src/build_kb.py 解析文档、切分片段、生成 Embedding
  -> chroma_db/knowledge_index.json 保存片段、元数据和向量
  -> src/rag_chain.py 对用户问题做向量化和混合相似度检索
  -> Prompt 拼接检索片段和用户问题
  -> Ollama 生成模型生成答案
  -> src/app.py 或命令行展示答案和引用来源
```

需要注意：项目目录名使用了 `chroma_db/`，但当前代码没有调用 ChromaDB 客户端。实际向量索引保存在 `chroma_db/knowledge_index.json` 中，检索时由代码读取 JSON 并使用 `sklearn.metrics.pairwise.cosine_similarity` 计算相似度。

当前本地 `data/public_kb/` 已经保存公开知识库材料，来源、许可和 URL 记录在 `data/public_kb/manifest.json`。因此，除非需要刷新公开材料或扩大采样规模，否则可以直接使用当前索引运行问答和评测。

## 一、文档解析与文本分割体现在哪

文档解析和文本分割主要体现在 `src/build_kb.py` 中，相关配置在 `src/config.py` 中。

### 1. 输入文档位置

原始知识库文档存放在：

```text
data/
```

程序通过 `src/config.py` 中的配置定位数据目录：

```python
DATA_DIR = BASE_DIR / "data"
```

也就是说，构建知识库时，程序会递归读取 `data/` 目录下的资料文件。

### 2. 支持的文档类型

在 `src/build_kb.py` 中，`SUPPORTED_SUFFIXES` 定义了可解析的文件后缀：

```python
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}
```

因此，当前系统支持：

- `.txt` 文本文档
- `.md` / `.markdown` Markdown 文档
- `.pdf` PDF 文档

这对应 RAG 流程中的“文档接入”环节：先把不同格式的资料统一转换成可处理的文本。

### 3. 普通文本和 Markdown 的解析

普通文本和 Markdown 文件通过 `read_text_file(path)` 读取：

```python
def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
```

这里体现了两个解析处理点：

- 程序会尝试多种编码，包括 `utf-8`、`utf-8-sig` 和 `gb18030`，避免中文资料因为编码不同导致读取失败。
- 如果正常编码读取失败，最后使用 `errors="ignore"` 兜底读取，保证程序尽量不中断。

对于 `.md` 文件，代码没有单独解析 Markdown 语法，而是把 Markdown 原文当作普通文本读取。这种做法简单直接，适合课程设计中的轻量知识库场景。

### 4. PDF 文档解析

PDF 通过 `read_pdf(path)` 解析：

```python
def read_pdf(path: Path) -> Iterable[Dict[str, object]]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            yield {"text": text, "source": path.name, "page": page_number}
```

这里体现了 PDF 解析的三个关键点：

- 使用 `pypdf.PdfReader` 打开 PDF。
- 按页提取文本，保留 `page_number`。
- 每一页生成一个文档对象，包含 `text`、`source` 和 `page`。

保留页码的意义是后续回答时可以给出引用来源，例如“某文件第几页”，这增强了 RAG 系统的可解释性。

### 5. 文档统一加载

`load_documents()` 负责把不同文件类型统一转换为文档列表：

```python
def load_documents() -> List[Dict[str, object]]:
    documents: List[Dict[str, object]] = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if path.suffix.lower() == ".pdf":
            documents.extend(read_pdf(path))
        else:
            documents.append(
                {
                    "text": read_text_file(path),
                    "source": path.name,
                    "page": None,
                }
            )
    return [doc for doc in documents if str(doc["text"]).strip()]
```

这个函数体现了文档解析后的标准化：

- 每个文档对象都有 `text` 字段，保存正文。
- 每个文档对象都有 `source` 字段，保存文件名。
- PDF 文档额外保留 `page`，非 PDF 文档的 `page` 为 `None`。
- 空文本会被过滤，避免无效内容进入知识库。

### 6. 文本分割策略

文本分割由 `split_text(text, chunk_size, chunk_overlap)` 实现：

```python
def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= chunk_size:
        return [normalized]

    separators = ["\n\n", "\n", "。", "；", "，", ". ", "; ", ", ", " "]
    chunks: List[str] = []
    start = 0
    ...
```

它的核心逻辑是：

- 先清洗文本，去掉空行和每行两侧空白。
- 如果文本长度不超过 `chunk_size`，直接作为一个片段。
- 如果文本较长，就按照固定窗口切分。
- 切分时优先寻找自然分隔符，例如段落、换行、句号、分号、逗号和空格。
- 如果找到合适分隔符，就尽量在语义边界处断开，而不是机械截断。
- 每个片段之间保留一定重叠，减少上下文断裂。

相关参数在 `src/config.py` 中：

```python
CHUNK_SIZE = 420
CHUNK_OVERLAP = 80
```

这表示：

- 每个文本片段目标长度约为 420 个字符。
- 相邻片段之间重叠约 80 个字符。

重叠的作用是保留跨片段上下文。例如一个概念的定义在前一个片段末尾，解释在后一个片段开头，如果完全不重叠，检索时可能只召回其中一半；加入 overlap 后，相关语义更容易被完整检索到。

### 7. 片段构建与元数据保存

`build_chunks(documents)` 负责把文档转换为 chunk：

```python
chunks.append(
    {
        "id": f"d{doc_id:03d}-c{chunk_id:03d}",
        "content": content,
        "source": doc["source"],
        "page": doc["page"],
        "char_count": len(content),
    }
)
```

每个片段包含：

- `id`：片段编号，例如 `d001-c001`，表示第 1 个文档的第 1 个片段。
- `content`：片段正文。
- `source`：来源文件名。
- `page`：页码，PDF 来源才有值。
- `char_count`：片段字符数。

这一步是 RAG 中“可检索知识单元”的形成过程。后续 Embedding、相似度检索和引用来源都基于这些 chunk 进行。

## 二、向量嵌入与相似度检索体现在哪

向量嵌入分为两个阶段：

- 离线阶段：对知识库片段生成向量并保存。
- 在线阶段：对用户问题生成向量，并与知识库向量计算相似度。

离线向量化主要在 `src/build_kb.py` 中，在线检索主要在 `src/rag_chain.py` 中。

### 1. Embedding 模型配置

Embedding 模型配置在 `src/config.py`：

```python
OLLAMA_BASE_URL = "http://127.0.0.1:8090"
EMBEDDING_MODEL = "nomic-embed-text"
```

这表示系统通过本地 Ollama 服务调用 `nomic-embed-text` 模型生成文本向量。

### 2. 知识库片段向量化

在 `src/build_kb.py` 的 `build_index()` 中，程序先构建 chunk，然后调用 Ollama Embedding：

```python
chunks = build_chunks(documents)
embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
embeddings = embedder.embed_documents([chunk["content"] for chunk in chunks])
```

这里体现了 RAG 的“文档向量化”环节：

- 输入是每个 chunk 的 `content`。
- Embedding 模型把文本映射为高维向量。
- 每个向量代表一个文本片段的语义特征。
- 语义相近的文本在向量空间中距离更近。

### 3. 向量索引保存

生成的向量被写入 `chroma_db/knowledge_index.json`：

```python
index = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "embedding_model": EMBEDDING_MODEL,
    "ollama_base_url": OLLAMA_BASE_URL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "document_count": len(documents),
    "chunk_count": len(chunks),
    "chunks": [
        {
            **chunk,
            "embedding": embedding,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ],
}
INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
```

保存后的 JSON 索引同时包含：

- 索引创建时间。
- 使用的 Embedding 模型。
- chunk 切分参数。
- 文档数量和片段数量。
- 每个 chunk 的正文、来源、页码、字符数和 embedding 向量。

当前已有索引中可以看到：

```text
document_count = 41
chunk_count = 163
embedding_model = nomic-embed-text
chunk_size = 420
chunk_overlap = 80
```

这说明当前知识库已经包含 4 份自建计算机专业资料和 37 份公开采集材料，最终切分出 163 个可检索片段。

### 4. 加载索引

在线问答时，`src/rag_chain.py` 的 `load_index()` 负责读取索引：

```python
def load_index(path: Path = INDEX_FILE) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"知识库索引不存在：{path}。请先运行 python src/build_kb.py")
    return json.loads(path.read_text(encoding="utf-8"))
```

如果索引不存在，程序会提示先运行：

```text
python src/build_kb.py
```

这体现了 RAG 系统中“先建库、后检索”的基本依赖关系。

### 5. 用户问题向量化

检索入口是 `retrieve(question, top_k=TOP_K)`：

```python
embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
question_embedding = embedder.embed_query(question)
matrix = [chunk["embedding"] for chunk in chunks]
embedding_scores = cosine_similarity([question_embedding], matrix)[0]
```

这里体现了查询向量化和向量相似度检索：

- 用户问题通过同一个 Embedding 模型转换为查询向量。
- 从索引中取出所有知识库片段向量。
- 使用余弦相似度计算问题向量和每个片段向量的语义相似程度。

余弦相似度越高，表示问题和片段在语义空间中越接近。

### 6. 关键词相似度补充

除了向量相似度，项目还加入了字符级 TF-IDF 相似度：

```python
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
tfidf_matrix = vectorizer.fit_transform(texts)
lexical_scores = cosine_similarity(vectorizer.transform([question]), tfidf_matrix)[0]
```

这体现了“混合检索”思路：

- Embedding 相似度偏向语义召回。
- TF-IDF 相似度偏向关键词和术语匹配。
- 中文专业文档中有很多固定术语，例如“虚拟内存”“事务隔离级别”“TCP 三次握手”，关键词匹配可以弥补纯语义检索可能忽略精确术语的问题。

`analyzer="char_wb"` 和 `ngram_range=(2, 4)` 表示按字符窗口生成 2 到 4 字符的 n-gram，这对中文文本比较适合，因为中文没有天然空格分词。

### 7. 分数归一化与加权融合

向量分和关键词分先经过 `min_max_normalize()` 归一化：

```python
normalized_embedding_scores = min_max_normalize(embedding_scores)
normalized_lexical_scores = min_max_normalize(lexical_scores)
```

然后按权重融合：

```python
scores = [
    EMBEDDING_SCORE_WEIGHT * embedding_score
    + LEXICAL_SCORE_WEIGHT * lexical_score
    for embedding_score, lexical_score in zip(
        normalized_embedding_scores, normalized_lexical_scores
    )
]
```

权重在 `src/config.py` 中：

```python
EMBEDDING_SCORE_WEIGHT = 0.45
LEXICAL_SCORE_WEIGHT = 0.55
```

这说明当前系统更偏重关键词匹配一点，同时保留语义检索能力。最终综合分数的含义是：

```text
综合分 = 0.45 * 向量语义相似度归一化分 + 0.55 * TF-IDF 关键词相似度归一化分
```

### 8. Top-K 结果返回

检索结果按综合分降序排序，并返回前 `top_k` 个片段：

```python
ranked = sorted(..., key=lambda item: item["score"], reverse=True)
return ranked[:top_k]
```

默认 `TOP_K` 在 `src/config.py` 中配置：

```python
TOP_K = 4
```

每个返回片段包含：

- `id`
- `content`
- `source`
- `page`
- `score`
- `embedding_score`
- `lexical_score`

这不仅给生成模型提供上下文，也给最终界面展示引用来源和相似度提供依据。

## 三、检索结果与生成模型的结合体现在哪

检索结果与生成模型结合主要体现在 `src/rag_chain.py` 的三个位置：

- `format_context()`：把检索片段格式化为上下文。
- `PROMPT_TEMPLATE`：定义 RAG Prompt 模板。
- `answer_question()`：串联检索、Prompt 拼接和模型生成。

### 1. 检索片段格式化为上下文

`format_context(chunks)` 把 Top-K 检索结果转换成可放入 Prompt 的参考资料：

```python
def format_context(chunks: List[Dict[str, object]]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        page = f"，页码：{chunk['page']}" if chunk.get("page") else ""
        parts.append(
            f"[片段{index}] 文件：{chunk['source']}{page}，综合分：{chunk['score']:.4f}\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)
```

它会把每个检索片段组织成类似这样的结构：

```text
[片段1] 文件：operating_systems.md，综合分：0.9321
片段正文...

[片段2] 文件：computer_networks.md，综合分：0.8450
片段正文...
```

这样做的意义是：

- 生成模型可以明确看到每个参考片段。
- 模型能区分不同来源。
- 后续回答可以引用“片段编号”和“文件名”。
- 相似度分数也被纳入上下文，便于提示模型优先使用更相关内容。

### 2. Prompt 模板约束生成模型

`PROMPT_TEMPLATE` 是检索增强生成的核心：

```python
PROMPT_TEMPLATE = """你是一个严谨的计算机专业文档问答助手。
请只依据【参考资料】回答【用户问题】。如果参考资料没有明确依据，请回答“知识库中没有找到明确答案”，不要补充资料外事实。

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 先用一段话直接回答问题；
2. 再分点说明关键概念、步骤或原因；
3. 最后列出依据来源，引用片段编号和文件名。
"""
```

这个模板体现了 RAG 的关键思想：生成模型不是直接凭自身参数知识回答，而是基于检索出来的上下文回答。

模板中有几个重要约束：

- “请只依据【参考资料】回答”：限制模型不要脱离知识库发挥。
- “如果参考资料没有明确依据”：要求模型在没有证据时拒答。
- “不要补充资料外事实”：减少幻觉。
- “列出依据来源”：要求答案可追溯。

因此，检索结果在这里不只是“附加文本”，而是被明确声明为回答依据。

### 3. 串联检索与生成

`answer_question()` 是完整 RAG 问答链路：

```python
def answer_question(question: str, top_k: int = TOP_K, model: str = LLM_MODEL) -> Dict[str, object]:
    started = time.perf_counter()
    retrieved = retrieve(question, top_k=top_k)
    retrieve_time = time.perf_counter() - started

    prompt = PROMPT_TEMPLATE.format(context=format_context(retrieved), question=question)

    generate_started = time.perf_counter()
    answer = generate_with_ollama(prompt, model=model)
    generate_time = time.perf_counter() - generate_started
    ...
```

这个函数把三个动作连起来：

1. 调用 `retrieve()` 从知识库中找出与问题最相关的片段。
2. 调用 `format_context()` 把片段整理成 Prompt 上下文。
3. 把 `{context}` 和 `{question}` 填入 `PROMPT_TEMPLATE`。
4. 调用 `generate_with_ollama()` 让本地大模型生成答案。

这就是“检索结果与生成模型结合”的具体代码体现。

### 4. 调用 Ollama 生成模型

生成模型调用由 `generate_with_ollama()` 完成：

```python
response = requests.post(
    f"{OLLAMA_BASE_URL}/api/generate",
    json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    },
    timeout=GENERATE_TIMEOUT,
)
```

相关生成参数在 `src/config.py` 中：

```python
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
TEMPERATURE = 0.2
NUM_PREDICT = 300
NUM_CTX = 2048
GENERATE_TIMEOUT = 240
```

这些参数含义如下：

- `LLM_MODEL`：使用的生成模型。
- `TEMPERATURE = 0.2`：较低温度，使回答更稳定、更少发散。
- `NUM_PREDICT = 300`：控制最大生成长度。
- `NUM_CTX = 2048`：控制上下文窗口大小。
- `GENERATE_TIMEOUT = 240`：HTTP 调用超时时间。

这里体现的是 RAG 中的“生成阶段”：模型接收包含参考资料的 Prompt，输出面向用户的自然语言答案。

### 5. 返回答案与引用来源

`answer_question()` 返回的不只是答案，还包括来源片段和耗时：

```python
return {
    "question": question,
    "answer": answer,
    "sources": [
        {
            "id": chunk["id"],
            "source": chunk["source"],
            "page": chunk.get("page"),
            "score": chunk["score"],
            "embedding_score": chunk["embedding_score"],
            "lexical_score": chunk["lexical_score"],
            "content": chunk["content"][:500],
        }
        for chunk in retrieved
    ],
    "top_k": top_k,
    "model": model,
    "embedding_model": EMBEDDING_MODEL,
    "ollama_base_url": OLLAMA_BASE_URL,
    "retrieve_time": retrieve_time,
    "generate_time": generate_time,
    "total_time": time.perf_counter() - started,
}
```

这体现了一个完整 RAG 问答结果通常包含：

- 用户问题。
- 生成答案。
- 检索来源。
- 相似度分数。
- 使用的模型。
- 检索耗时、生成耗时和总耗时。

这种返回结构方便命令行、Web 页面和评测脚本复用。

## 四、在 Web 页面中的体现

Web 页面入口是 `src/app.py`，它通过 Streamlit 调用 RAG 链：

```python
from rag_chain import answer_question
```

页面中用户可以设置 Top-K：

```python
top_k = st.slider("Top-K 片段数", min_value=1, max_value=8, value=TOP_K)
```

点击按钮后执行：

```python
result = answer_question(question.strip(), top_k=top_k)
```

然后页面展示：

- `result["answer"]`：生成模型回答。
- `result["sources"]`：引用来源片段。
- `source["score"]`：综合相似度。
- `result["retrieve_time"]`、`result["generate_time"]`、`result["total_time"]`：耗时统计。

这说明 Web 界面并不重新实现 RAG 逻辑，而是作为展示层调用 `src/rag_chain.py` 中封装好的问答链路。

## 五、三部分对应关系总结

| RAG 环节 | 当前项目中的体现 | 主要文件/函数 | 关键产物 |
| --- | --- | --- | --- |
| 公开材料采集 | 从 Hugging Face CSE 数据集和 arXiv 计算机论文入口采集轻量材料 | `src/collect_public_kb.py` | `data/public_kb/` 与 `manifest.json` |
| 文档解析 | 读取 `data/` 下的 TXT、MD、PDF，抽取文本和来源信息 | `src/build_kb.py`：`read_text_file()`、`read_pdf()`、`load_documents()` | 标准化文档对象：`text`、`source`、`page` |
| 文本分割 | 按长度、自然分隔符和重叠窗口切成 chunk | `src/build_kb.py`：`split_text()`、`build_chunks()` | 片段对象：`id`、`content`、`source`、`page`、`char_count` |
| 向量嵌入 | 使用 Ollama 的 `nomic-embed-text` 生成 chunk 向量 | `src/build_kb.py`：`build_index()` | `chroma_db/knowledge_index.json` 中的 `embedding` |
| 相似度检索 | 对问题向量和 chunk 向量计算余弦相似度，同时结合 TF-IDF 关键词分 | `src/rag_chain.py`：`retrieve()` | Top-K 片段及 `score`、`embedding_score`、`lexical_score` |
| 检索增强生成 | 把 Top-K 片段拼入 Prompt，调用 Ollama 生成模型回答 | `src/rag_chain.py`：`format_context()`、`PROMPT_TEMPLATE`、`generate_with_ollama()`、`answer_question()` | 最终答案和引用来源 |
| 页面展示 | 输入问题、设置 Top-K、展示答案和来源 | `src/app.py` | Streamlit 问答界面 |

## 六、可以在课程设计报告中这样表述

本系统采用“离线建库 + 在线检索生成”的 RAG 架构。离线阶段，系统可以先通过 `src/collect_public_kb.py` 采集公开 CSE 数据集和 arXiv 计算机论文摘要，也可以直接使用当前已经保存在 `data/public_kb/` 的本地材料。随后系统读取 `data/` 目录下的专业文档，支持 TXT、Markdown 和 PDF 格式；对 PDF 按页抽取文本，对普通文本按编码兼容方式读取。系统对文本进行清洗和分块，分块时设置 `CHUNK_SIZE=420`、`CHUNK_OVERLAP=80`，并优先在换行、句号、分号、逗号等自然边界处切分，以尽量保持片段语义完整。每个片段保存编号、正文、来源文件、页码和字符数。

在向量化阶段，系统通过本地 Ollama 服务调用 `nomic-embed-text` 模型，将每个文本片段转换为高维向量，并将片段元数据和向量一起保存到 `chroma_db/knowledge_index.json`。在线问答时，用户问题同样被转换为向量，系统使用余弦相似度计算问题向量与知识库片段向量的语义相关度。同时，为了增强中文专业术语匹配效果，系统还使用字符级 TF-IDF 计算关键词相似度，并按照 `0.45 * 向量分 + 0.55 * 关键词分` 的方式得到综合分，最终返回 Top-K 个最相关片段。

在生成阶段，系统将检索得到的片段通过 `format_context()` 格式化为带片段编号、文件名、页码和相似度的参考资料，再填入 `PROMPT_TEMPLATE`。Prompt 明确要求生成模型只依据参考资料回答，并在资料不足时回答“知识库中没有找到明确答案”，从而降低模型幻觉。最后系统通过 Ollama 的 `/api/generate` 接口调用 `qwen2.5:7b-instruct-q4_K_M` 生成答案，并返回答案、引用来源、相似度分数及检索和生成耗时。

因此，本项目中的 RAG 流程不是简单地把问题直接交给大模型，而是先从本地专业知识库中检索相关证据，再让大模型基于这些证据组织回答。文档解析与切分保证知识能够被索引，Embedding 与相似度检索保证相关内容能够被召回，检索结果与生成模型结合则保证最终回答具有上下文依据和来源可追溯性。

## 七、两组完整实验结果记录

为了便于课程设计报告展示复现实验过程，项目保留首次完整实验结果，并追加 2026-05-23 重新完整流程实验结果。两组数据均基于同一知识库、同一 30 条评测问题和 Top-K=4；RAG 策略对照实验只切换检索策略，双模型实验只切换生成模型。

### 1. 单模型 30 条评测

重新完整流程实验结果（2026-05-23）如下：

| 样本数 | Top-K | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 4 | 1.0000 | 0.6143 | 0.4019s | 135.2647s | 135.6667s |

### 2. RAG 策略优化对照

首次完整实验结果如下：

| 检索策略 | 样本数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 30 | 1.0000 | 0.7667 | 0.4058 | 0.4444s | 137.7124s | 138.1570s |
| hybrid | 30 | 1.0000 | 1.0000 | 0.5948 | 0.2312s | 129.4322s | 129.6635s |

重新完整流程实验结果（2026-05-23）如下：

| 检索策略 | 样本数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 30 | 1.0000 | 0.7000 | 0.4076 | 0.4339s | 123.1658s | 123.5998s |
| hybrid | 30 | 1.0000 | 1.0000 | 0.6185 | 0.4084s | 137.9988s | 138.4073s |

两次实验都表明，`hybrid` 相对 `vector` 能显著提高检索命中率和平均参考答案覆盖度。耗时差异受本地生成模型非确定性、运行环境负载和 Ollama 推理速度影响，不改变控制变量设置。

### 3. 双模型对比

首次完整实验结果如下：

| 模型 | 样本数 | 成功数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek-r1:7b-qwen-distill-q4_K_M | 30 | 30 | 1.0000 | 1.0000 | 0.4324 | 171.2024s |
| qwen2.5:7b-instruct-q4_K_M | 30 | 30 | 1.0000 | 1.0000 | 0.5746 | 143.3015s |

重新完整流程实验结果（2026-05-23）如下：

| 模型 | 样本数 | 成功数 | 成功率 | 检索命中率 | 平均参考答案覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek-r1:7b-qwen-distill-q4_K_M | 30 | 30 | 1.0000 | 1.0000 | 0.4765 | 0.4563s | 165.0030s | 165.4594s |
| qwen2.5:7b-instruct-q4_K_M | 30 | 30 | 1.0000 | 1.0000 | 0.6205 | 0.4582s | 138.7100s | 139.1683s |

两组双模型结果都支持相同结论：Qwen 在平均参考答案覆盖度和平均总耗时上优于 DeepSeek R1，更适合作为当前项目默认问答模型。
