# Using with RAG Systems

- Source: hatakekksheeshh/CSE_course_RAG
- License: MIT
- URL: https://huggingface.co/datasets/hatakekksheeshh/CSE_course_RAG/resolve/main/README.md
- Original path: README.md

### Using with RAG Systems

The dataset is designed to work with the CSE Course RAG system:

```python
from rag.query_pipeline import QueryPipeline

# Initialize pipeline with pre-built indices
pipeline = QueryPipeline(
    index_dir="./data/indices",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Query the system
result = pipeline.answer(
    query="What is the grading policy?",
    course="Introduction_to_Computing"
)
```
