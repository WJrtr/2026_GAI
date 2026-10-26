"""Shared defaults for the HW3 RAG + QLoRA pipeline."""

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
RAG_CACHE_VERSION = "v2"

MAX_SEQ_LENGTH = 4096
CHUNK_MIN_WORDS = 180
CHUNK_MAX_WORDS = 260
DEFAULT_TOP_K = 4
DEFAULT_RERANK_CANDIDATES = 24
DEFAULT_EVIDENCE_WORD_BUDGET = 1800

LABEL_NAMES = {
    0: "Attribution Failure",
    1: "Entity",
    2: "Number",
    3: "Overgeneralization",
    4: "Temporal",
}

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

OVERSAMPLE_BY_LABEL = {
    0: 1,
    1: 1,
    2: 5,
    3: 1,
    4: 10,
}
