# Processing Raw Data

- Source: hatakekksheeshh/CSE_course_RAG
- License: MIT
- URL: https://huggingface.co/datasets/hatakekksheeshh/CSE_course_RAG/resolve/main/README.md
- Original path: README.md

### Processing Raw Data

If you need to reprocess the data:

```python
# Load processed course data
import json

with open("./data/processed/course_name/course_name.json", "r") as f:
    course_data = json.load(f)
```
