from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
RESULT_DIR = BASE_DIR / "results"
EVAL_FILE = BASE_DIR / "eval_questions.csv"
INDEX_FILE = DB_DIR / "knowledge_index.json"

OLLAMA_BASE_URL = "http://localhost:8090"
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
