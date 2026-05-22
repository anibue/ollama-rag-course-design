# Data Processing

- Source: hatakekksheeshh/CSE_course_RAG
- License: MIT
- URL: https://huggingface.co/datasets/hatakekksheeshh/CSE_course_RAG/resolve/main/README.md
- Original path: README.md

### Data Processing

The dataset has been processed through the following pipeline:

1. **Conversion**: PDFs/Office docs → page images
2. **OCR**: PaddleOCR text extraction
3. **Parsing**: Structured JSON extraction (syllabus and material parsers)
4. **Chunking**: Text chunking with overlap
5. **Embedding**: Sentence-transformer embeddings
6. **Indexing**: FAISS index construction
