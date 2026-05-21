import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_ollama import OllamaEmbeddings

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DB_DIR,
    EMBEDDING_MODEL,
    INDEX_FILE,
    OLLAMA_BASE_URL,
)


SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> Iterable[Dict[str, object]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要安装 pypdf：pip install pypdf") from exc

    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            yield {"text": text, "source": path.name, "page": page_number}


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


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= chunk_size:
        return [normalized]

    separators = ["\n\n", "\n", "。", "；", "，", ". ", "; ", ", ", " "]
    chunks: List[str] = []
    start = 0

    while start < len(normalized):
        target_end = min(start + chunk_size, len(normalized))
        end = target_end
        window = normalized[start:target_end]

        if target_end < len(normalized):
            best_pos = -1
            for sep in separators:
                pos = window.rfind(sep)
                if pos > chunk_size * 0.45 and pos > best_pos:
                    best_pos = pos + len(sep)
            if best_pos > 0:
                end = start + best_pos

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def build_chunks(documents: List[Dict[str, object]]) -> List[Dict[str, object]]:
    chunks: List[Dict[str, object]] = []
    for doc_id, doc in enumerate(documents, start=1):
        for chunk_id, content in enumerate(
            split_text(str(doc["text"]), CHUNK_SIZE, CHUNK_OVERLAP), start=1
        ):
            chunks.append(
                {
                    "id": f"d{doc_id:03d}-c{chunk_id:03d}",
                    "content": content,
                    "source": doc["source"],
                    "page": doc["page"],
                    "char_count": len(content),
                }
            )
    return chunks


def build_index() -> Dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents()
    if not documents:
        raise RuntimeError(f"{DATA_DIR} 中没有可读取的 TXT/MD/PDF 文档。")

    chunks = build_chunks(documents)
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    embeddings = embedder.embed_documents([chunk["content"] for chunk in chunks])

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
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="构建本地 RAG 知识库索引")
    parser.parse_args()

    index = build_index()
    print(f"原始文档数量：{index['document_count']}")
    print(f"切分片段数量：{index['chunk_count']}")
    print(f"索引保存位置：{INDEX_FILE}")
    print(f"Ollama 地址：{OLLAMA_BASE_URL}")
    print(f"Embedding 模型：{EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
