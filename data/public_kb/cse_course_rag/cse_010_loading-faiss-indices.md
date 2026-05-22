# Loading FAISS Indices

- Source: hatakekksheeshh/CSE_course_RAG
- License: MIT
- URL: https://huggingface.co/datasets/hatakekksheeshh/CSE_course_RAG/resolve/main/README.md
- Original path: README.md

### Loading FAISS Indices

```python
import faiss
import pickle

# Load FAISS index
index = faiss.read_index("./data/indices/course_name.index")

# Load metadata
with open("./data/indices/course_name_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
```
