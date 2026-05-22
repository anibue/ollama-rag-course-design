# Data Fields

- Source: hatakekksheeshh/CSE_course_RAG
- License: MIT
- URL: https://huggingface.co/datasets/hatakekksheeshh/CSE_course_RAG/resolve/main/README.md
- Original path: README.md

### Data Fields

**Processed Data (JSON)**:
- `course`: Course name
- `course_id`: Course code
- `schema_version`: Data schema version
- `slides`: Array of slide objects with:
  - `page_index`: Page number
  - `chapter_num`: Chapter number
  - `source_file`: Source file path
  - `metadata`: Processing metadata
  - `raw_text`: Extracted OCR text

**FAISS Indices**:
- Vector embeddings for semantic search
- Metadata mappings for chunk retrieval
- Course-specific indices
