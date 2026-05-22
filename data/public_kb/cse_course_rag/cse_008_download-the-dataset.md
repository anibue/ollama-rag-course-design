# Download the Dataset

- Source: hatakekksheeshh/CSE_course_RAG
- License: MIT
- URL: https://huggingface.co/datasets/hatakekksheeshh/CSE_course_RAG/resolve/main/README.md
- Original path: README.md

### Download the Dataset

```python
from huggingface_hub import snapshot_download

# Download the entire dataset
dataset_path = snapshot_download(
    repo_id="hatakekksheeshh/CSE_course_RAG",
    repo_type="dataset",
    local_dir="./data"
)
```

Or use the provided download script:

```bash
python dataset.py
```
