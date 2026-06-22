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

历史二路权重仍保留在 `src/config.py` 中，主要是兼容旧实验说明：

```python
EMBEDDING_SCORE_WEIGHT = 0.45
LEXICAL_SCORE_WEIGHT = 0.55
```

当前 full30 使用的是 Hybrid V3 三路融合权重：

```python
EMBEDDING_SCORE_WEIGHT_V3 = 0.40
LEXICAL_SCORE_WEIGHT_V3 = 0.45
ENTITY_SCORE_WEIGHT = 0.15
```

也就是说，当前综合分数的含义是：

```text
综合分 = 0.40 * 向量语义相似度归一化分
       + 0.45 * TF-IDF 关键词相似度归一化分
       + 0.15 * 实体匹配分
```

这样做是因为计算机专业问答里有不少短术语、英文缩写和固定概念名，纯向量检索有时会找到语义接近但关键词不准的片段；加入 TF-IDF 和实体匹配后，排序更容易把真正包含关键概念的片段推到前面。

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
- `entity_score`

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

在向量化阶段，系统通过本地 Ollama 服务调用 `nomic-embed-text` 模型，将每个文本片段转换为高维向量，并将片段元数据和向量一起保存到 `chroma_db/knowledge_index.json`。在线问答时，用户问题同样被转换为向量，系统使用余弦相似度计算问题向量与知识库片段向量的语义相关度。同时，为了增强中文专业术语匹配效果，系统还使用字符级 TF-IDF 计算关键词相似度，并加入实体匹配得分。当前 full30 使用的综合分为 `0.40 * 向量分 + 0.45 * TF-IDF 关键词分 + 0.15 * 实体匹配分`，最终返回 Top-K 个最相关片段。

在生成阶段，系统将检索得到的片段通过 `format_context()` 格式化为带片段编号、文件名、页码和相似度的参考资料，再填入 `PROMPT_TEMPLATE`。Prompt 明确要求生成模型只依据参考资料回答，并在资料不足时回答“知识库中没有找到明确答案”，从而降低模型幻觉。最后系统通过 Ollama 的 `/api/generate` 接口调用 `qwen2.5:7b-instruct-q4_K_M` 生成答案，并返回答案、引用来源、相似度分数及检索和生成耗时。

因此，本项目中的 RAG 流程不是简单地把问题直接交给大模型，而是先从本地专业知识库中检索相关证据，再让大模型基于这些证据组织回答。文档解析与切分保证知识能够被索引，Embedding 与相似度检索保证相关内容能够被召回，检索结果与生成模型结合则保证最终回答具有上下文依据和来源可追溯性。

## 七、两组完整实验结果记录

为了便于课程设计报告展示复现实验过程，项目保留历史实验结果，并追加 `full30_20260621` 完整 30 条全套评测结果。报告主结论以 full30 为准；旧的 `goal_20260621` 结果只作为链路验证材料。

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

---

## 五、高级 RAG 改进技术详解（2026-06-12）

本节说明在原有基础链路之上实现的 7 项高级改进，各技术均已通过端到端运行验证。

### 5.1 三路融合检索（Hybrid V3）

**所在位置**：`src/rag_chain.py` — `entity_match_score()` + `retrieve()` hybrid 分支

原有混合检索只融合向量相似度和 TF-IDF 词汇相似度（二路）。本改进引入**实体匹配**第三路信号：

```
综合分 = 0.40 × 向量得分 + 0.45 × TF-IDF 词汇得分 + 0.15 × 实体匹配得分
```

实体匹配实现：用正则从问题中提取三类词——英文技术词 `[A-Za-z][A-Za-z0-9_\-]+`、中文双字词 `[一-鿿]{2,}`、数字 `\d+`，计算这些词在候选片段文本中出现的比例作为实体匹配得分（0.0–1.0）。

配置项（`src/config.py`）：

```python
EMBEDDING_SCORE_WEIGHT_V3 = 0.40
LEXICAL_SCORE_WEIGHT_V3   = 0.45
ENTITY_SCORE_WEIGHT       = 0.15   # 三者之和 = 1.0
```

`retrieve()` 在 hybrid 模式下将三路得分一并写入返回字典（`embedding_score`、`lexical_score`、`entity_score`），供 Streamlit 页面和评测脚本可视化展示。

**理论依据**：A-RAG 的多信号融合思路（arXiv:2405.12777）；LightRAG 的实体图检索对实体匹配信号的验证。

---

### 5.2 语义去重（Semantic Deduplication）

**所在位置**：`src/rag_chain.py` — `deduplicate_by_similarity(chunks, threshold=0.85)`

同一段落在文档中可能被切分为多个高度重叠的片段，同时被检索命中导致冗余。本改进在最终排序后、截取 Top-K 之前对候选集做凝聚聚类去重：

1. 从每个 chunk 字典中取 `embedding` 字段（真实向量），构建成对余弦相似度矩阵。
2. 将相似度矩阵转换为距离矩阵（`1 - cosine_similarity`）。
3. 用 `sklearn.cluster.AgglomerativeClustering(linkage="average", metric="precomputed", distance_threshold=1-threshold)` 聚类。
4. 每簇保留综合 `score` 最高的片段。

去重发生在 `retrieve()` 内部，使用的 `embedding` 字段在返回给调用方之前由 `_sources_from_chunks()` 剥离（换成语义化的 `entity_score`），对外不暴露原始向量。

**理论依据**：Self-Correcting RAG（arXiv:2406.13692）中基于聚类的多知识片段选择（MMKP）的轻量实现。

---

### 5.3 上下文扩展（Context Expansion）

**所在位置**：`src/rag_chain.py` — `expand_context(retrieved_chunks, window=EXPANSION_WINDOW)`

RAG 检索命中的往往是一段话的中间部分，缺少前置背景或后续说明。上下文扩展在检索完成后追加相邻片段：

1. 调用 `load_index()` 读取全量索引，建立 `chunk_id → 位置` 映射。
2. 对每个已检索片段，找出同一文档内位置 ±`EXPANSION_WINDOW` 的邻居（默认 ±1）。
3. 将新邻居追加到返回列表，`score` 标为 0.0（区分于检索得分）。
4. 去重已检索片段，避免重复追加。

配置项：

```python
CONTEXT_EXPANSION = True
EXPANSION_WINDOW  = 1
```

在 `answer_question()` 中，扩展在检索后、Prompt 拼接前执行；在 `answer_question_adaptive()` 中，扩展在充分性判断后、最终去重前执行。

**理论依据**：A-RAG 的 chunk_read 扩展机制（arXiv:2405.12777），保留段落完整上下文。

---

### 5.4 自适应多轮检索

**所在位置**：`src/rag_chain.py` — `answer_question_adaptive(question, top_k, model)`

固定 Top-K 检索在问题难度高时可能召回不足。自适应多轮检索通过充分性判断按需扩展：

**流程**：

```
第一轮 hybrid 检索（Top-K）
  ↓
LLM 充分性判断（SUFFICIENCY_PROMPT）
  ↓  答案："不足" → 第二轮检索（ADAPTIVE_TOP_K_SECOND=8）
  ↓  否则直接进入下一步
上下文扩展（expand_context）
  ↓
语义去重 + cap 至 top_k
  ↓
stepwise Prompt 生成最终答案
```

充分性判断 Prompt（`SUFFICIENCY_PROMPT`）要求模型判断"能否仅凭以上参考资料直接回答问题"，回答含"不足"时触发第二轮。结果字段：

- `retrieval_rounds`：实际检索轮数（1 或 2）。
- `sufficiency_judgment`：充分性判断原文，便于调试和展示。

配置项：

```python
ADAPTIVE_TOP_K_SECOND = 8   # 第二轮扩大的 Top-K
MAX_RETRIEVAL_ROUNDS  = 2
```

**理论依据**：SIM-RAG 充分性判断机制（arXiv:2406.10208）；ReaLM-Retrieve 步级检索不确定性估计。

---

### 5.5 分步推理与自反思 Prompt

**所在位置**：`src/rag_chain.py` — `PROMPT_TEMPLATE_STEPWISE`、`PROMPT_TEMPLATE_REFLECTIVE`、`answer_question(prompt_variant=)`

**三种 Prompt 策略**：

`baseline`：最简约的引用约束 Prompt，直接要求基于资料回答，资料不足时拒答。

`stepwise`（默认推荐）：强制模型经历四个推理步骤，每步有明确输出要求：

```
步骤 1：识别信息线索 — 找出参考资料中与问题相关的关键信息点
步骤 2：提取关键事实 — 从各片段中提取支持答案的具体证据，并标注 [片段N]
步骤 3：综合推导结论 — 将提取的事实整合推导
步骤 4：最终答案 — 给出完整答案并标注引用来源
```

`reflective`：先生成初步答案，再以独立视角自问"我的答案中是否有不支持的声明？"，输出修订后答案。灵感来自 Self-RAG 的 [IsSup] 反思 token 机制。

CLI 参数：`--prompt {baseline,stepwise,reflective}`（`src/rag_chain.py`）；Streamlit 侧栏也支持切换。

**理论依据**：Self-RAG（ICLR 2024，Asai et al.）— 反思 token 和自批评机制；ReasonRAG — 问题分解与步进推理框架。

---

### 5.6 多维度评测指标

**所在位置**：`src/eval.py` — `faithfulness_score()`、`reasoning_faithfulness()`、`test_rejection()`

在原有**检索命中率**和**参考答案覆盖度**基础上，新增三类指标：

**忠实度评分（`faithfulness_score`）**：

调用 LLM，传入答案文本和参考片段列表，要求对"答案中的声明是否均有片段支持"打分 0–10，归一化到 0–1。通过 `eval.py --faithfulness` 开启。

**推理引用率（`reasoning_faithfulness`）**：

无需额外 LLM 调用——用正则从答案中找 `[片段N]` 或 `[N]` 形式的引用标注，检查 N 是否在合法范围内（1 ≤ N ≤ 片段数），返回有效引用比例。`stepwise` Prompt 的答案通常具有较高推理引用率，该指标可直接反映 Prompt 策略的引用严谨性。

**拒答准确率（`test_rejection`）**：

3 条故意超出知识库范围的问题（如询问知识库中不存在的技术细节），检查模型是否正确触发拒答词（"没有找到明确答案"/"无法回答"/"知识库中没有"）。通过 `eval.py --rejection` 开启。

**配套实验脚本**：

- `src/prompt_compare.py`：对比三种 Prompt 策略的命中率、参考覆盖度和推理引用率。
- `src/weight_tune.py`：三路融合权重网格搜索（见 5.7 节）。

**理论依据**：Gao et al., RAG Survey（arXiv:2312.10997）的忠实度 / 答案相关性 / 上下文相关性三维评估框架。

---

### 5.7 检索权重网格搜索

**所在位置**：`src/weight_tune.py` — `grid_search()`

三路融合的权重组合需要在当前知识库和评测问题上验证最优值，不应依赖经验拍板。本脚本在权重空间中枚举合法组合（三者之和 = 1.0），对每组权重用评测集快速评分，选出最优：

**搜索空间**（过滤 sum ≠ 1.0 的组合，±0.01 容差）：

| 维度 | 候选值 |
|---|---|
| `emb`（向量权重） | 0.30 / 0.35 / 0.40 / 0.45 / 0.50 |
| `lex`（词汇权重） | 0.35 / 0.40 / 0.45 / 0.50 / 0.55 |
| `entity`（实体权重） | 0.05 / 0.10 / 0.15 / 0.20 |

**权重热重载机制**：

```python
config.EMBEDDING_SCORE_WEIGHT_V3 = emb
config.LEXICAL_SCORE_WEIGHT_V3   = lex
config.ENTITY_SCORE_WEIGHT       = entity
importlib.reload(rag_chain)   # 使新权重立即生效，无需重启进程
```

搜索完成后自动恢复默认权重（0.40 / 0.45 / 0.15）。

**评分标准**：`(hit_rate, avg_overlap)` 的字典序最大值（优先最大化命中率，相同时看覆盖度）。

**CLI**：

```cmd
python src/weight_tune.py --limit 10                  # 快速评测 10 题
python src/weight_tune.py --entity-fixed 0.15         # 固定实体权重，搜索更快
```

**输出**：`results/weight_tune_results.json`，包含 `best`（最优组合及分数）和 `all`（全量结果列表）。

**理论依据**：UR2（Tsinghua-dhy/UR2）难度感知课程学习；RA-DIT 双端优化思想的轻量化网格搜索实现。

---

### 5.8 Streamlit 页面增强

**所在位置**：`src/app.py`

在原有单轮问答页面基础上的增强内容：

**多轮对话记忆**：

使用 `st.session_state.messages`（列表，元素为 `{role, content}`）存储完整对话历史。每轮问答后追加用户消息和助手消息，历史记录在页面重渲染时显示。侧栏"清空对话历史"按钮重置状态。

**侧栏控制项**：

- 检索策略：`vector` / `hybrid` / `adaptive`（来自 `ANSWER_MODES`）。
- Prompt 策略：`stepwise` / `baseline` / `reflective`。
- Top-K：滑块 1–8。
- 忠实度自检开关：勾选后对当前轮答案额外调用 `check_faithfulness()`，展示 0–10 分评分和详情。

**检索来源展示**：

每个引用片段展示四项得分：综合分、向量分（`embedding_score`）、词汇分（`lexical_score`）、实体匹配分（`entity_score`）。自适应模式下额外展示检索轮数和充分性判断结果。

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

结果说明：本轮轻量实验主要用于验证高级 RAG 改动的可运行性和补充记录，不替代完整 30 条评测。由于 `num_predict=24` 输出被刻意截短，参考答案覆盖度偏低是预期现象；更适合比较检索链路、脚本稳定性和运行耗时。完整质量结论以 `full30_20260621` 为主。

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

本轮使用 `src/full30_runner.py` 对效果评测、RAG 策略对比、Prompt 对比、双模型对比和权重调优进行了完整 30 条样本跑批。结果文件统一使用 `full30_20260621` 后缀，未覆盖旧结果。运行参数为 `Top-K=4`、`num_predict=64`、`timeout=360`，并关闭 context expansion。

运行命令：

```cmd
python src\full30_runner.py --tasks eval --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
python src\full30_runner.py --tasks strategy --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
python src\full30_runner.py --tasks prompt --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
python src\full30_runner.py --tasks model --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
python src\full30_runner.py --tasks weight --suffix full30_20260621 --top-k 4 --prompt baseline --num-predict 64 --timeout 360 --resume
```

完成性验证：

| 模块 | 明细数量 | 完成情况 |
|---|---:|---|
| 单模型效果评测 | 30 | `complete=true` |
| RAG 策略对比 | 90 | vector / hybrid / adaptive 各 30 条 |
| Prompt 对比 | 90 | baseline / stepwise / reflective 各 30 条 |
| 双模型对比 | 60 | DeepSeek / Qwen 各 30 条 |
| 权重调优 | 5 组权重 | `complete=true` |

RAG 策略对比结果：

| 检索策略 | 样本数 | 检索命中率 | 平均参考覆盖度 | 平均检索耗时 | 平均生成耗时 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| vector | 30 | 0.7000 | 0.3669 | 0.6956s | 122.6932s | 123.3889s |
| hybrid | 30 | 1.0000 | 0.5404 | 0.7362s | 137.6391s | 138.3754s |
| adaptive | 30 | 1.0000 | 0.3850 | 124.5705s | 161.5900s | 286.2434s |

该结果验证了 RAG 链路中的检索策略优化：`vector` 只使用 Embedding 相似度，是优化前基线；`hybrid` 使用三路融合 `0.40 * 向量分 + 0.45 * TF-IDF 关键词分 + 0.15 * 实体匹配分`，是优化后方案。在同一知识库、同一 30 条问题、同一模型和同一 Top-K 条件下，hybrid 将检索命中率从 0.7000 提升到 1.0000，将参考答案覆盖度从 0.3669 提升到 0.5404。

源码注释同步：`src/` 下核心脚本已补充必要中文注释，覆盖公开数据采集、文档切分、Embedding 调用、三路融合检索、adaptive 检索、评测指标、full30 增量写盘和 Web 页面参数控制等位置。注释主要解释为什么这样做，不对简单赋值和普通函数调用做重复说明。

Prompt 对比结果：

| Prompt 变体 | 样本数 | 检索命中率 | 平均参考覆盖度 | 推理忠实度 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|
| baseline | 30 | 1.0000 | 0.5286 | 0.0000 | 138.9987s |
| stepwise | 30 | 1.0000 | 0.3782 | 0.8000 | 134.8031s |
| reflective | 30 | 1.0000 | 0.6596 | 0.0667 | 142.3079s |

双模型对比结果：

| 模型 | 样本数 | 成功率 | 检索命中率 | 平均参考覆盖度 | 平均总耗时 |
|---|---:|---:|---:|---:|---:|
| deepseek-r1:7b-qwen-distill-q4_K_M | 30 | 1.0000 | 1.0000 | 0.0000 | 142.2310s |
| qwen2.5:7b-instruct-q4_K_M | 30 | 1.0000 | 1.0000 | 0.5382 | 146.9947s |

权重调优结果：最优组合为 `emb=0.30, lex=0.55, entity=0.15`，命中率 1.0000，平均分差 0.5419。该结果说明当前计算机专业中文问答集更依赖关键词和实体命中，和 hybrid 检索优于 vector 检索的结论一致。
