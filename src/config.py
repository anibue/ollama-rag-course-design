import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
RESULT_DIR = BASE_DIR / "results"
EVAL_FILE = BASE_DIR / "eval_questions.csv"
INDEX_FILE = DB_DIR / "knowledge_index.json"

# 本项目的 Ollama 走本机 8090 端口，和默认 11434 不一样。
OLLAMA_BASE_URL = "http://127.0.0.1:8090"

LOCAL_NO_PROXY_HOSTS = ["127.0.0.1", "localhost"]
# 有些环境会开系统代理，本地模型请求绕一下代理反而更稳。
existing_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
no_proxy_hosts = [
    host.strip()
    for host in existing_no_proxy.split(",")
    if host.strip()
]
for host in LOCAL_NO_PROXY_HOSTS:
    if host not in no_proxy_hosts:
        no_proxy_hosts.append(host)
os.environ["NO_PROXY"] = ",".join(no_proxy_hosts)
os.environ["no_proxy"] = os.environ["NO_PROXY"]
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
COMPARE_MODELS = [
    "deepseek-r1:7b-qwen-distill-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
]
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 420
CHUNK_OVERLAP = 80
TOP_K = 4
DEFAULT_RETRIEVAL_MODE = "hybrid"
RETRIEVAL_MODES = ["vector", "hybrid"]
TEMPERATURE = 0.2
GENERATE_TIMEOUT = 240
NUM_PREDICT = 300
NUM_CTX = 2048
EMBEDDING_SCORE_WEIGHT = 0.45
LEXICAL_SCORE_WEIGHT = 0.55

# 旧版二路权重先留着；full30 当前用的是下面这组三路权重。
ENTITY_SCORE_WEIGHT = 0.15
EMBEDDING_SCORE_WEIGHT_V3 = 0.40
LEXICAL_SCORE_WEIGHT_V3 = 0.45

# adaptive 会多做一次充分性判断，耗时会比 hybrid 明显长。
ADAPTIVE_RETRIEVAL = True
ADAPTIVE_TOP_K_SECOND = 8
MAX_RETRIEVAL_ROUNDS = 2

# 忠实度自检默认关掉，完整评测时再开会慢很多。
ENABLE_FAITHFULNESS_CHECK = False

# 相邻片段补充能缓解切分边界问题，控制实验里可以关掉。
CONTEXT_EXPANSION = True
EXPANSION_WINDOW = 1

# CLI 和 Web 页面直接使用这三个模式名。
ANSWER_MODES = ["vector", "hybrid", "adaptive"]
