#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: numpy. Install the required local packages first, such as "
        "numpy, sentence-transformers, rank-bm25, torch, and transformers."
    ) from exc

PLACEHOLDER_RE = re.compile(
    r"\b(?:BIBREF|SECREF|TABREF|FIGREF|INLINEFORM|DISPLAYFORM)\w*\b",
    re.IGNORECASE,
)
HEADING_RE = re.compile(
    r"^(?:abstract|introduction|background|related work|method|methods|approach|"
    r"approaches|data|dataset|datasets|experiment|experiments|results|discussion|"
    r"conclusion|conclusions|acknowledg(?:e)?ments?)$",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])(?:["\')\]]+)?\s+(?=(?:["\(\[]?[A-Z0-9]))')
TOKEN_RE = re.compile(r"[a-z0-9]+")
QUOTE_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{6,160})[\"“”'‘’]")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHAR_CHUNK_SPECS = (
    (200, 50, "window_200"),
    (400, 100, "window_400"),
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

LIST_HINTS = (
    "what are",
    "which",
    "baseline",
    "baselines",
    "methods",
    "components",
    "features",
    "layers",
    "steps",
    "systems",
    "models",
    "tasks",
    "datasets",
    "metrics",
    "advantages",
    "disadvantages",
)
NUMERIC_HINTS = (
    "how many",
    "what year",
    "%",
    "number",
    "dataset size",
    "how much",
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    raw_text: str
    norm_text: str
    paragraph_id: int
    sentence_start: int
    sentence_end: int
    window_size: int
    source: str = "unknown"


@dataclass(frozen=True)
class QueryBundle:
    original: str
    alternatives: tuple[str, ...]
    hyde: str

    def as_list(self) -> list[str]:
        queries: list[str] = []
        seen = set()
        for query in [self.original, *list(self.alternatives), self.hyde]:
            cleaned = re.sub(r"\s+", " ", query).strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                queries.append(cleaned)
        return queries


@dataclass
class PipelineConfig:
    llm_model: str
    embedding_model: str
    reranker_model: str
    device: str
    dense_top_k: int = 24
    sparse_top_k: int = 24
    fused_top_k: int = 48
    rerank_top_k: int = 12
    hard_evidence_cap: int = 5
    rrf_k: int = 60
    min_chunk_chars: int = 40
    embed_batch_size: int = 128
    rerank_batch_size: int = 32
    llm_max_new_tokens: int = 96
    llm_max_input_tokens: int = 3072
    evidence_score_drop_ratio: float = 0.4
    evidence_context_chars: int = 80


class LocalLLM:
    def __init__(self, model_name: str, device: str, max_input_tokens: int = 3072):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: transformers and torch are required for local LLM inference."
            ) from exc

        self._torch = torch
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float32
        if device == "cuda":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 96,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"System: {system_prompt}\n\n"
                f"User: {user_prompt}\n\n"
                "Assistant:"
            )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["top_p"] = 0.9
        with self._torch.inference_mode():
            generated = self.model.generate(**encoded, **generation_kwargs)
        output_ids = generated[0, input_length:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return re.sub(r"\s+", " ", text).strip()


class DenseEmbedder:
    def __init__(self, model_name: str, device: str, batch_size: int = 128):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: sentence-transformers. Install it before running the pipeline."
            ) from exc

        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self.encode([BGE_QUERY_PREFIX + query.strip() for query in queries])


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str, batch_size: int = 32):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: sentence-transformers CrossEncoder support. "
                "Install sentence-transformers before running the pipeline."
            ) from exc

        self.model = CrossEncoder(
            model_name,
            device=device,
            trust_remote_code=True,
            max_length=512,
        )
        self.batch_size = batch_size

    def score(self, question: str, chunks: Sequence[Chunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return self.score_texts(question, [chunk.raw_text for chunk in chunks])

    def score_texts(self, question: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0,), dtype=np.float32)
        pairs = [(question, text) for text in texts]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32)


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: rank-bm25. Install it before running the pipeline."
            ) from exc

        self.tokens = [tokenize(chunk.norm_text) for chunk in chunks]
        self.index = BM25Okapi(self.tokens)

    def rank(self, query: str, top_k: int) -> list[int]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = np.asarray(self.index.get_scores(query_tokens), dtype=np.float32)
        return top_indices(scores, top_k)


def parse_args() -> argparse.Namespace:
    script_name = Path(sys.argv[0]).with_suffix(".json").name
    parser = argparse.ArgumentParser(description="HW2 RAG pipeline")
    parser.add_argument("--dataset", default="datasets/private_dataset.json", help="Input dataset path")
    parser.add_argument("--output", default=script_name, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
        help="Local LLM model path or HF model name",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        help="Embedding model name",
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large"),
        help="Reranker model name",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE", "cuda"),
        help="Execution device for local models",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=int(os.getenv("EMBED_BATCH_SIZE", "128")),
        help="Batch size for dense embedding inference",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=int(os.getenv("RERANK_BATCH_SIZE", "32")),
        help="Batch size for reranker inference",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=int(os.getenv("LLM_MAX_NEW_TOKENS", "96")),
        help="Max new tokens for the local LLM answer generation",
    )
    return parser.parse_args()


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list")
    return data


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = PLACEHOLDER_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def split_paragraphs(full_text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n+", full_text.strip())
    paragraphs: list[str] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        current: list[str] = []
        for line in lines:
            if is_heading(line):
                if current:
                    paragraphs.append(" ".join(current).strip())
                    current = []
                continue
            current.append(line)
        if current:
            paragraphs.append(" ".join(current).strip())
    return paragraphs


def is_heading(line: str) -> bool:
    clean = line.strip().strip(":")
    if HEADING_RE.match(clean):
        return True
    if len(clean) > 80:
        return False
    if clean.endswith((".", "?", "!")):
        return False
    words = clean.split()
    return 1 <= len(words) <= 8 and all(word[:1].isupper() for word in words if word)


def split_sentences(paragraph: str) -> list[str]:
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if not paragraph:
        return []
    parts = SENTENCE_SPLIT_RE.split(paragraph)
    if len(parts) == 1 and len(paragraph) > 500:
        parts = re.split(r";\s+", paragraph)
    sentences = [part.strip() for part in parts if part and part.strip()]
    return sentences or [paragraph]


def make_chunk(
    chunk_id: int,
    text: str,
    paragraph_id: int,
    sentence_start: int,
    sentence_end: int,
    window_size: int,
    source: str,
) -> Chunk | None:
    raw_text = re.sub(r"\s+", " ", text).strip()
    norm_text = normalize_text(raw_text)
    if not raw_text or len(raw_text) < 20 or len(norm_text) < 10:
        return None
    return Chunk(
        chunk_id=chunk_id,
        raw_text=raw_text,
        norm_text=norm_text,
        paragraph_id=paragraph_id,
        sentence_start=sentence_start,
        sentence_end=sentence_end,
        window_size=window_size,
        source=source,
    )


def sliding_window_chunks(
    paragraph: str,
    chunk_size: int,
    overlap: int,
    source: str,
) -> list[tuple[str, int, int, int, str]]:
    sentences = split_sentences(paragraph)
    chunks: list[tuple[str, int, int, int, str]] = []
    current_chunk: list[str] = []
    current_len = 0
    start_index = 0

    for sentence_index, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        if current_chunk and current_len + sentence_len > chunk_size:
            chunks.append((" ".join(current_chunk), start_index, sentence_index, len(current_chunk), source))

            overlap_sents: list[str] = []
            overlap_len = 0
            for existing in reversed(current_chunk):
                if overlap_len + len(existing) > overlap and overlap_sents:
                    break
                overlap_sents.insert(0, existing)
                overlap_len += len(existing)
            current_chunk = overlap_sents
            start_index = max(0, sentence_index - len(current_chunk))
            current_len = sum(len(item) for item in current_chunk)

        if not current_chunk:
            start_index = sentence_index
        current_chunk.append(sentence)
        current_len += sentence_len

    if current_chunk:
        chunks.append((" ".join(current_chunk), start_index, len(sentences), len(current_chunk), source))
    return chunks


def paragraph_level_chunks(full_text: str) -> list[tuple[str, int, int, int, str]]:
    chunks: list[tuple[str, int, int, int, str]] = []
    for paragraph in split_paragraphs(full_text):
        if len(paragraph) < 50:
            continue
        if len(paragraph) <= 800:
            sentence_count = max(1, len(split_sentences(paragraph)))
            chunks.append((paragraph, 0, sentence_count, sentence_count, "paragraph"))
        else:
            chunks.extend(sliding_window_chunks(paragraph, 400, 100, "paragraph_sub"))
    return chunks


def build_chunks(full_text: str, min_chunk_chars: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    next_chunk_id = 0
    paragraphs = split_paragraphs(full_text)
    for paragraph_id, paragraph in enumerate(paragraphs):
        for chunk_size, overlap, source in CHAR_CHUNK_SPECS:
            for text, start, end, window_size, chunk_source in sliding_window_chunks(
                paragraph,
                chunk_size,
                overlap,
                source,
            ):
                candidate = make_chunk(
                    next_chunk_id,
                    text,
                    paragraph_id,
                    start,
                    end,
                    window_size,
                    chunk_source,
                )
                if candidate is None or len(candidate.raw_text) < min_chunk_chars or candidate.norm_text in seen:
                    continue
                seen.add(candidate.norm_text)
                chunks.append(candidate)
                next_chunk_id += 1

    for paragraph_id, (text, start, end, window_size, source) in enumerate(paragraph_level_chunks(full_text), start=10_000):
        candidate = make_chunk(next_chunk_id, text, paragraph_id, start, end, window_size, source)
        if candidate is None or candidate.norm_text in seen:
            continue
        seen.add(candidate.norm_text)
        chunks.append(candidate)
        next_chunk_id += 1

    if not chunks:
        candidate = make_chunk(0, full_text, 0, 0, 1, 1, "fallback")
        if candidate is not None:
            chunks.append(candidate)
    return chunks


def heuristic_keyword_query(question: str) -> str:
    tokens = tokenize(question)
    filtered = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    return " ".join(filtered[:12]) or question


def heuristic_semantic_query(question: str) -> str:
    question = question.strip()
    if question.lower().startswith("what"):
        return f"In this paper, {question[5:].rstrip('?')}"
    return f"This paper addresses: {question.rstrip('?')}"


def heuristic_hyde(question: str) -> str:
    return (
        "This paper states that "
        + question.rstrip("?")
        + " and provides the relevant method, result, or dataset detail directly."
    )


def extract_quoted_queries(question: str) -> list[str]:
    clues: list[str] = []
    for match in QUOTE_RE.finditer(question):
        clue = match.group(1).replace("[sic]", " ")
        clue = re.sub(r"\s+", " ", clue).strip(" ,;:.")
        if len(clue.split()) >= 3:
            clues.append(clue)
    deduped: list[str] = []
    seen = set()
    for clue in clues:
        key = clue.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(clue)
    return deduped


def pattern_query_variants(question: str) -> list[str]:
    lower = question.lower()
    variants: list[str] = []

    if "baseline" in lower:
        variants.extend(
            [
                f"{question.rstrip('?')} baseline models",
                "baseline models compare against",
            ]
        )
    if "compare" in lower:
        variants.extend(["compare against previous methods", "comparison baselines"])
    if "dataset" in lower:
        variants.extend(["dataset used in the paper", "evaluation dataset"])
    if lower.startswith("where") or "come from" in lower:
        variants.extend(["come from collected from", "data source origin"])
    if lower.startswith("how") and "train" in lower:
        variants.extend(["trained by using", "training objective loss"])
    if "features" in lower:
        variants.extend(["features include", "feature set"])
    if "layers" in lower:
        variants.extend(["layers include", "model architecture layers"])
    if "state of the art" in lower:
        variants.extend(["state of the art systems", "existing systems such as"])
    if any(token in lower for token in ("approaches", "models", "methods", "systems")):
        variants.extend(["approaches include", "models used in the paper"])
    if "metrics" in lower:
        variants.append("evaluation metrics include")
    if "three layers" in lower or "three levels" in lower:
        variants.extend(["split into three levels", "three annotation levels"])
    if "core component" in lower:
        variants.extend(["main contribution", "core component"])

    deduped: list[str] = []
    seen = set()
    for variant in variants:
        cleaned = re.sub(r"\s+", " ", variant).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


def parse_query_lines(raw: str) -> list[str]:
    lines = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[\.\)]\s*|-+\s*)", "", line.strip())
        if cleaned and len(cleaned) > 8:
            lines.append(cleaned)
    deduped: list[str] = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    return deduped


def generate_query_set(llm: LocalLLM, question: str) -> QueryBundle:
    alternatives: list[str] = []
    query_prompt = (
        "You rewrite retrieval queries for question answering over a single NLP paper. "
        "Generate 3 alternative search queries that help locate the exact answer span. "
        "Use concise search-style English. Output one query per line and nothing else."
    )
    try:
        raw = llm.chat(query_prompt, f"Question: {question}", temperature=0.1, max_tokens=96)
        alternatives = parse_query_lines(raw)
    except Exception:
        alternatives = []

    fallback_candidates = [
        heuristic_keyword_query(question),
        heuristic_semantic_query(question),
    ]
    fallback_candidates.extend(extract_quoted_queries(question))
    fallback_candidates.extend(pattern_query_variants(question))
    for candidate in fallback_candidates:
        if candidate and candidate not in alternatives:
            alternatives.append(candidate)
        if len(alternatives) >= 6:
            break

    hyde = heuristic_hyde(question)
    hyde_prompt = (
        "Write a short hypothetical passage from an NLP paper that would answer the question. "
        "Use 2 to 3 concise sentences and stay close to academic paper wording."
    )
    try:
        hyde_raw = llm.chat(hyde_prompt, f"Question: {question}", temperature=0.1, max_tokens=120)
        if hyde_raw:
            hyde = hyde_raw.strip()
    except Exception:
        pass

    return QueryBundle(
        original=question.strip(),
        alternatives=tuple(alternatives[:6]),
        hyde=hyde or heuristic_hyde(question),
    )


def retrieve_dense(
    embedder: DenseEmbedder,
    question_queries: Sequence[str],
    chunks: Sequence[Chunk],
    top_k: int,
) -> list[list[int]]:
    chunk_embeddings = embedder.encode([chunk.norm_text for chunk in chunks])
    rank_lists: list[list[int]] = []
    if chunk_embeddings.size == 0:
        return rank_lists
    query_embeddings = embedder.encode_queries([normalize_text(query) for query in question_queries])
    score_matrix = np.matmul(chunk_embeddings, query_embeddings.T)
    for column_index in range(score_matrix.shape[1]):
        rank_lists.append(top_indices(score_matrix[:, column_index], top_k))
    return rank_lists


def retrieve_sparse(
    question_queries: Sequence[str],
    chunks: Sequence[Chunk],
    top_k: int,
) -> list[list[int]]:
    retriever = BM25Retriever(chunks)
    return [retriever.rank(query, top_k) for query in question_queries]


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0:
        return []
    top_k = min(top_k, scores.shape[0])
    order = np.argsort(-scores)[:top_k]
    return [int(index) for index in order]


def fuse_rrf(rank_lists: Sequence[Sequence[int]], rrf_k: int) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for rank_list in rank_lists:
        for rank, chunk_id in enumerate(rank_list):
            scores[int(chunk_id)] += 1.0 / (rrf_k + rank + 1)
    return [chunk_id for chunk_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def rerank(
    reranker: CrossEncoderReranker,
    question: str,
    candidate_chunks: Sequence[Chunk],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    scores = reranker.score(question, candidate_chunks)
    pairs = list(zip(candidate_chunks, scores.tolist()))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[:top_k]


def classify_question(question: str) -> str:
    lower = question.lower()
    if any(hint in lower for hint in NUMERIC_HINTS):
        return "numeric"
    if lower.startswith("what ") and any(
        hint in lower
        for hint in (
            "features",
            "layers",
            "methods",
            "baselines",
            "components",
            "systems",
            "models",
            "tasks",
            "datasets",
            "metrics",
            "genres",
            "languages",
            "targets",
        )
    ):
        return "list"
    if any(hint in lower for hint in LIST_HINTS):
        if "which" in lower and " and " not in lower and "baseline" not in lower and "methods" not in lower:
            return "single"
        return "list"
    return "single"


def lexical_overlap_ratio(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    return intersection / min(len(tokens_a), len(tokens_b))


def adds_new_information(chunk: Chunk, selected: Sequence[Chunk]) -> bool:
    return all(lexical_overlap_ratio(chunk.norm_text, existing.norm_text) < 0.8 for existing in selected)


def question_keywords(question: str) -> list[str]:
    return [
        token
        for token in tokenize(question)
        if token not in STOPWORDS and token not in {"many", "much", "year", "number"}
    ]


def list_like_text(text: str) -> bool:
    lower = text.lower()
    return (
        text.count(";") >= 1
        or text.count(",") >= 2
        or " and " in lower
        or ":" in text
        or any(
            cue in lower
            for cue in (
                "include",
                "includes",
                "including",
                "such as",
                "consists of",
                "comprises",
                "split into",
                "baseline",
                "compared with",
                "compared to",
                "levels",
            )
        )
    )


def candidate_fragments_from_text(question: str, text: str) -> list[str]:
    question_type = classify_question(question)
    cleaned = clean_chunk_for_answer(text)
    sentences = split_sentences(cleaned) or [cleaned]
    candidates: list[str] = []
    patterns: list[str] = []

    if question_type == "list":
        patterns.extend(
            [
                r"\b(?:include|includes|including)\s+(.+)",
                r"\b(?:consists of|consist of|comprised of|comprises|comprise|encompasses|encompass)\s+(.+)",
                r"\b(?:such as)\s+(.+)",
                r"\b(?:split into)\s+(.+)",
                r"\b(?:compare(?:s|d)?(?:\s+against|\s+to|\s+with)?)\s+(.+)",
            ]
        )
    if question.lower().startswith("where"):
        patterns.extend(
            [
                r"\b(?:come|comes) from\s+(.+)",
                r"\bcollected from\s+(.+)",
                r"\bobtained from\s+(.+)",
            ]
        )
    if question.lower().startswith("how"):
        patterns.extend(
            [
                r"\btrained by\s+(.+)",
                r"\busing\s+(.+)",
                r"\bthrough\s+(.+)",
                r"\bvia\s+(.+)",
            ]
        )

    for sentence in sentences:
        sentence = sentence.strip(" -")
        if len(sentence) < 8:
            continue
        candidates.append(sentence)
        if ":" in sentence:
            tail = sentence.split(":", 1)[1].strip()
            if len(tail) >= 8:
                candidates.append(tail)
        if ";" in sentence:
            for part in sentence.split(";"):
                part = part.strip()
                if len(part) >= 8:
                    candidates.append(part)
        for pattern in patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                span = match.group(1).strip(" .;:")
                if len(span) >= 8:
                    candidates.append(span)

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        cleaned_candidate = re.sub(r"\s+", " ", candidate).strip(" -")
        key = normalize_text(cleaned_candidate)
        if cleaned_candidate and key not in seen:
            seen.add(key)
            deduped.append(cleaned_candidate)
    return deduped


def candidate_bonus(question: str, candidate: str) -> float:
    lower_q = question.lower()
    lower_c = candidate.lower()
    question_type = classify_question(question)
    overlap = sum(1 for token in question_keywords(question) if token in lower_c)
    bonus = overlap * 0.12

    if question_type == "numeric":
        if re.search(r"\b\d[\d,.\-]*\b", candidate):
            bonus += 2.0
        if any(token in lower_q for token in ("average", "on average", "how long")):
            if any(token in lower_c for token in ("average", "mean", "roughly", "range", "most texts")):
                bonus += 1.2
            if any(token in lower_c for token in ("shortest", "longest", "minimum", "maximum")):
                bonus -= 1.2
    if question_type == "list":
        if list_like_text(candidate):
            bonus += 0.9
        if any(
            token in lower_c
            for token in (
                "include",
                "includes",
                "including",
                "such as",
                "consists of",
                "comprises",
                "split into",
                "baseline",
                "compared",
                "levels",
                "datasets",
                "models",
                "approaches",
            )
        ):
            bonus += 0.7
    if lower_q.startswith("where") and any(
        token in lower_c for token in ("come from", "comes from", "collected from", "obtained from")
    ):
        bonus += 1.0
    if lower_q.startswith("how") and any(token in lower_c for token in ("trained by", "using", "through", "via")):
        bonus += 0.8
    if "baseline" in lower_q and any(token in lower_c for token in ("baseline", "compare", "against")):
        bonus += 1.0
    if "compare" in lower_q and any(token in lower_c for token in ("compare", "against", "versus", "baseline")):
        bonus += 1.0
    return bonus


def rank_answer_candidates(
    reranker: CrossEncoderReranker,
    question: str,
    evidence_chunks: Sequence[Chunk],
    limit: int = 10,
) -> list[tuple[str, float]]:
    candidates: list[str] = []
    seen = set()
    for chunk in evidence_chunks:
        for candidate in candidate_fragments_from_text(question, chunk.raw_text):
            key = normalize_text(candidate)
            if candidate and key not in seen:
                seen.add(key)
                candidates.append(candidate)
    if not candidates:
        candidates = [clean_chunk_for_answer(chunk.raw_text) for chunk in evidence_chunks]

    prelim = sorted(
        candidates,
        key=lambda text: (candidate_bonus(question, text), -len(text)),
        reverse=True,
    )[:36]
    rerank_scores = reranker.score_texts(question, prelim)
    ranked = [
        (text, float(score) + candidate_bonus(question, text))
        for text, score in zip(prelim, rerank_scores.tolist())
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def best_clauses_for_question(question: str, texts: Sequence[str], limit: int = 3) -> list[str]:
    keywords = question_keywords(question)
    ranked: list[tuple[int, int, str]] = []
    for text in texts:
        clauses = re.split(r"(?<=[.;:])\s+", text)
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            lowered = clause.lower()
            overlap = sum(1 for token in keywords if token in lowered)
            ranked.append((overlap, -len(clause), clause))
    ranked.sort(reverse=True)
    selected: list[str] = []
    seen = set()
    for overlap, _neg_len, clause in ranked:
        if clause in seen:
            continue
        if overlap == 0 and selected:
            continue
        selected.append(clause)
        seen.add(clause)
        if len(selected) >= limit:
            break
    if not selected:
        for text in texts[:limit]:
            if text not in seen:
                selected.append(text)
                seen.add(text)
    return selected


def find_best_span(text: str, full_text: str) -> tuple[int, int] | None:
    position = full_text.find(text)
    if position < 0:
        return None
    return (position, position + len(text))


def expand_chunk_context(chunk_text: str, full_text: str, context_chars: int = 80) -> str:
    span = find_best_span(chunk_text, full_text)
    if span is None:
        return chunk_text
    start, end = span
    expanded_start = max(0, start - context_chars)
    expanded_end = min(len(full_text), end + context_chars)
    if expanded_start > 0:
        left_boundary = re.search(r"[.!?]\s+", full_text[expanded_start:start])
        if left_boundary:
            expanded_start += left_boundary.end()
    if expanded_end < len(full_text):
        right_boundary = re.search(r"[.!?]\s+", full_text[end:expanded_end])
        if right_boundary:
            expanded_end = end + right_boundary.end()
    return re.sub(r"\s+", " ", full_text[expanded_start:expanded_end]).strip()


def spans_overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    return min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]) > 0


def select_evidence(
    question: str,
    reranked_chunks: Sequence[tuple[Chunk, float]],
    full_text: str,
    hard_cap: int = 5,
    drop_ratio: float = 0.4,
    context_chars: int = 80,
) -> list[Chunk]:
    if not reranked_chunks:
        return []
    question_type = classify_question(question)
    question_cap = hard_cap if question_type == "list" else min(3, hard_cap)
    effective_drop_ratio = drop_ratio * 0.5 if question_type == "list" else drop_ratio
    top_score = reranked_chunks[0][1]
    selected: list[Chunk] = []
    selected_spans: list[tuple[int, int]] = []

    for chunk, score in reranked_chunks:
        if len(selected) >= question_cap:
            break
        if selected and score < top_score * effective_drop_ratio:
            continue

        expanded_text = expand_chunk_context(chunk.raw_text, full_text, context_chars=context_chars)
        span = find_best_span(expanded_text, full_text)
        if span is not None and any(spans_overlap(span, existing) for existing in selected_spans):
            continue

        expanded_chunk = Chunk(
            chunk_id=chunk.chunk_id,
            raw_text=expanded_text,
            norm_text=normalize_text(expanded_text),
            paragraph_id=chunk.paragraph_id,
            sentence_start=chunk.sentence_start,
            sentence_end=chunk.sentence_end,
            window_size=chunk.window_size,
            source=chunk.source,
        )
        if selected and not adds_new_information(expanded_chunk, selected):
            continue

        selected.append(expanded_chunk)
        if span is not None:
            selected_spans.append(span)

    if not selected:
        selected = [reranked_chunks[0][0]]
    return selected[:question_cap]


def clean_chunk_for_answer(text: str) -> str:
    cleaned = PLACEHOLDER_RE.sub(" ", text)
    cleaned = re.sub(
        r"^(?:abstract|introduction|background|related work|method|methods|dataset|datasets|"
        r"results|discussion|conclusion|conclusions|analysis|evaluation|experiments?|"
        r"data collection|data preprocessing|overview)\b[:\s-]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*(?:Answer|Final Answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*]\s*)", "", cleaned)
    cleaned = re.sub(
        r'^(?:According to|Based on|From) the (?:evidence|paper|passage)[,:]?\s*',
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _legacy_extract_numeric_phrase(question: str, text: str) -> str:
    lower_q = question.lower()
    patterns: list[str] = []
    if any(token in lower_q for token in ("average", "on average", "how long")):
        patterns.extend(
            [
                r"(?:roughly\s+)?from\s+\d[\d,]*(?:\.\d+)?\s+to\s+\d[\d,]*(?:\.\d+)?(?:\s+[A-Za-z][A-Za-z\-]*){0,4}",
                r"\d[\d,]*(?:\.\d+)?\s*(?:to|–|-)\s*\d[\d,]*(?:\.\d+)?(?:\s+[A-Za-z][A-Za-z\-]*){0,4}",
            ]
        )
    patterns.extend(
        [
            r"\d[\d,]*(?:\.\d+)?\s*%(?:\s+[A-Za-z][A-Za-z\-]*){0,3}",
            r"\d[\d,]*(?:\.\d+)?(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,6}",
        ]
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                matches = [" ".join(part for part in match if part) for match in matches]
            phrases = [re.sub(r"\s+", " ", match).strip(" .;,") for match in matches if match]
            if phrases:
                if any(token in lower_q for token in ("average", "on average", "how long")):
                    preferred = [
                        phrase
                        for phrase in phrases
                        if any(token in phrase.lower() for token in ("to", "tokens", "hours", "range"))
                    ]
                    if preferred:
                        return max(preferred, key=len)
                return max(phrases, key=len)
    return ""


def extract_numeric_phrase(question: str, text: str) -> str:
    lower_q = question.lower()
    patterns: list[str] = []
    if any(token in lower_q for token in ("average", "on average", "how long")):
        patterns.extend(
            [
                r"(?:roughly\s+)?from\s+\d[\d,]*(?:\.\d+)?\s+to\s+\d[\d,]*(?:\.\d+)?(?:\s+[A-Za-z][A-Za-z\-]*){0,4}",
                r"\d[\d,]*(?:\.\d+)?\s*(?:to|-)\s*\d[\d,]*(?:\.\d+)?(?:\s+[A-Za-z][A-Za-z\-]*){0,4}",
            ]
        )
    patterns.extend(
        [
            r"\d[\d,]*(?:\.\d+)?\s*%(?:\s+[A-Za-z][A-Za-z\-]*){0,3}",
            r"\d[\d,]*(?:\.\d+)?(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,6}",
        ]
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                matches = [" ".join(part for part in match if part) for match in matches]
            phrases = [re.sub(r"\s+", " ", match).strip(" .;,") for match in matches if match]
            if phrases:
                if any(token in lower_q for token in ("average", "on average", "how long")):
                    preferred = [
                        phrase
                        for phrase in phrases
                        if any(token in phrase.lower() for token in ("to", "tokens", "hours", "range"))
                    ]
                    if preferred:
                        return max(preferred, key=len)
                return max(phrases, key=len)
    return ""


def trim_candidate_answer(question: str, text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .;:")
    lower_q = question.lower()
    question_type = classify_question(question)

    if question_type == "numeric":
        numeric = extract_numeric_phrase(question, cleaned)
        if numeric:
            return numeric

    if lower_q.startswith("where"):
        for pattern in (
            r"\b(?:come|comes) from\s+(.+)",
            r"\bcollected from\s+(.+)",
            r"\bobtained from\s+(.+)",
        ):
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                tail = match.group(1).strip(" .;:")
                if len(tail) >= 6:
                    return tail

    if lower_q.startswith("how") and "train" in lower_q:
        for pattern in (r"\btrained by\s+(.+)", r"\busing\s+(.+)", r"\bwith\s+(.+loss.+)"):
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                tail = match.group(1).strip(" .;:")
                if len(tail) >= 8:
                    return tail

    if question_type == "list":
        if ":" in cleaned:
            tail = cleaned.split(":", 1)[1].strip(" .;:")
            if tail and list_like_text(tail):
                return tail
        for pattern in (
            r"\b(?:include|includes|including)\s+(.+)",
            r"\b(?:consists of|consist of|comprised of|comprises|comprise|encompasses|encompass)\s+(.+)",
            r"\b(?:such as)\s+(.+)",
            r"\b(?:split into)\s+(.+)",
        ):
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                tail = match.group(1).strip(" .;:")
                if tail and len(tail) >= 8:
                    return tail

    return cleaned


def answer_is_generic(question: str, answer: str) -> bool:
    lower = answer.lower().strip()
    if not lower:
        return True
    if lower in {"this paper", "our method", "the paper", "relevant findings"}:
        return True
    if lower.startswith(("this paper ", "the paper ", "we propose ", "our approach ")):
        return True
    return False


def synthesize_answer(
    llm: LocalLLM,
    reranker: CrossEncoderReranker,
    question: str,
    evidence_chunks: Sequence[Chunk],
    max_tokens: int,
) -> str:
    if not evidence_chunks:
        return ""
    ranked_candidates = rank_answer_candidates(reranker, question, evidence_chunks, limit=8)
    candidate_texts = [trim_candidate_answer(question, text) for text, _score in ranked_candidates]
    cleaned_candidates: list[str] = []
    for text in candidate_texts:
        cleaned = clean_answer(text)
        if cleaned:
            cleaned_candidates.append(cleaned)
    candidate_texts = cleaned_candidates
    deduped_candidates: list[str] = []
    seen_candidates = set()
    for candidate in candidate_texts:
        key = normalize_text(candidate)
        if key not in seen_candidates:
            seen_candidates.add(key)
            deduped_candidates.append(candidate)
    if not deduped_candidates:
        deduped_candidates = [clean_chunk_for_answer(chunk.raw_text) for chunk in evidence_chunks]
    question_type = classify_question(question)
    if question_type == "numeric":
        return deduped_candidates[0]

    style = {
        "single": "Return only the single best answer span or one short sentence copied from the candidates.",
        "numeric": "Return only the exact numeric answer phrase copied from the candidates.",
        "list": "Return only the supported answer items from the candidates, covering all key items, separated by semicolons.",
    }[question_type]
    system_prompt = (
        "You extract answers from candidate spans for an NLP paper question.\n\n"
        "Rules:\n"
        "- Use only words that appear in the candidate spans.\n"
        "- Prefer exact spans over paraphrases.\n"
        "- Do not summarize the whole paper.\n"
        "- Do not mention missing information or uncertainty.\n"
        "- If multiple items are required, include all supported key items from the candidates.\n"
        f"- {style}"
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Candidate spans:\n- " + "\n- ".join(deduped_candidates[:6]) + "\n"
        "Return only the final answer text."
    )
    try:
        answer = llm.chat(system_prompt, user_prompt, temperature=0.0, max_tokens=max_tokens).strip()
        answer = clean_answer(answer)
        if answer and not answer_is_generic(question, answer):
            return answer
    except Exception:
        pass
    if question_type == "list":
        return "; ".join(deduped_candidates[: min(3, len(deduped_candidates))])
    return deduped_candidates[0] if deduped_candidates else fallback_answer(question, evidence_chunks)


def fallback_answer(question: str, evidence_chunks: Sequence[Chunk]) -> str:
    if not evidence_chunks:
        return ""
    question_type = classify_question(question)
    cleaned_texts = [clean_chunk_for_answer(chunk.raw_text) for chunk in evidence_chunks]
    first_text = cleaned_texts[0]
    if question_type == "numeric":
        keyword_tokens = [
            token
            for token in tokenize(question)
            if token not in STOPWORDS and token not in {"many", "much", "year", "number"}
        ]
        numberish_phrases: list[str] = []
        for text in cleaned_texts:
            clauses = re.split(r"(?<=[.;:])\s+", text)
            for clause in clauses:
                if not re.search(r"\b\d[\d,.\-]*\b", clause):
                    continue
                lowered = clause.lower()
                score = sum(1 for token in keyword_tokens if token in lowered)
                numberish_phrases.append((score, clause))
        if numberish_phrases:
            best_clause = max(numberish_phrases, key=lambda item: item[0])[1]
            exact_phrase = re.findall(
                r"\b\d[\d,.\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,5}",
                best_clause,
            )
            if exact_phrase:
                best_phrase = max(
                    exact_phrase,
                    key=lambda phrase: sum(1 for token in keyword_tokens if token in phrase.lower()),
                )
                return best_phrase.strip()
            return best_clause.strip()
    if question_type == "list" and len(evidence_chunks) > 1:
        return "; ".join(best_clauses_for_question(question, cleaned_texts, limit=3))
    best_single = best_clauses_for_question(question, cleaned_texts, limit=1)
    return best_single[0] if best_single else first_text


def format_submission(records: Sequence[tuple[str, str, Sequence[Chunk]]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for title, answer, evidence_chunks in records:
        evidence = []
        seen = set()
        for chunk in evidence_chunks:
            if chunk.raw_text not in seen:
                evidence.append(chunk.raw_text)
                seen.add(chunk.raw_text)
        if not evidence:
            evidence = [answer] if answer else [""]
        formatted.append(
            {
                "title": title,
                "answer": str(answer).strip(),
                "evidence": evidence[:40],
            }
        )
    return formatted


def build_config(args: argparse.Namespace) -> PipelineConfig:
    device = args.device
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
    return PipelineConfig(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        device=device,
        embed_batch_size=args.embed_batch_size,
        rerank_batch_size=args.rerank_batch_size,
        llm_max_new_tokens=args.llm_max_new_tokens,
    )


def run_pipeline(data: Sequence[dict[str, Any]], config: PipelineConfig) -> list[dict[str, Any]]:
    print(f"Runtime device: {config.device}", file=sys.stderr)
    if config.device == "cuda":
        try:
            import torch

            print(f"CUDA device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        except Exception:
            pass
    print(
        f"Loading embedding model: {config.embedding_model} (batch_size={config.embed_batch_size})",
        file=sys.stderr,
    )
    embedder = DenseEmbedder(config.embedding_model, config.device, batch_size=config.embed_batch_size)
    print(
        f"Loading reranker model: {config.reranker_model} (batch_size={config.rerank_batch_size})",
        file=sys.stderr,
    )
    reranker = CrossEncoderReranker(config.reranker_model, config.device, batch_size=config.rerank_batch_size)
    print(f"Loading local LLM model: {config.llm_model}", file=sys.stderr)
    llm = LocalLLM(config.llm_model, config.device, max_input_tokens=config.llm_max_input_tokens)

    output_records: list[tuple[str, str, Sequence[Chunk]]] = []
    total = len(data)
    for index, item in enumerate(data, start=1):
        title = str(item.get("title", ""))
        full_text = str(item.get("full_text", ""))
        question = str(item.get("question", ""))
        print(f"[{index}/{total}] {title}", file=sys.stderr)

        chunks = build_chunks(full_text, min_chunk_chars=config.min_chunk_chars)
        query_bundle = generate_query_set(llm, question)
        dense_rank_lists = retrieve_dense(embedder, query_bundle.as_list(), chunks, config.dense_top_k)
        sparse_rank_lists = retrieve_sparse(query_bundle.as_list(), chunks, config.sparse_top_k)
        fused_ids = fuse_rrf(dense_rank_lists + sparse_rank_lists, config.rrf_k)[: config.fused_top_k]
        candidate_chunks = [chunks[chunk_id] for chunk_id in fused_ids]
        if not candidate_chunks:
            candidate_chunks = chunks[: config.rerank_top_k]
        reranked = rerank(reranker, question, candidate_chunks, config.rerank_top_k)

        if reranked:
            top_chunk = reranked[0][0].raw_text[:200]
            refined_query = f"{question} {top_chunk}"
            refined_dense = retrieve_dense(embedder, [refined_query], chunks, config.dense_top_k)
            refined_sparse = retrieve_sparse([refined_query], chunks, config.sparse_top_k)
            refined_ids = fuse_rrf(refined_dense + refined_sparse, config.rrf_k)[:20]
            combined_ids = []
            seen_ids = set()
            for chunk, _score in reranked:
                if chunk.chunk_id not in seen_ids:
                    combined_ids.append(chunk.chunk_id)
                    seen_ids.add(chunk.chunk_id)
            for chunk_id in refined_ids:
                if chunk_id not in seen_ids:
                    combined_ids.append(chunk_id)
                    seen_ids.add(chunk_id)
            reranked = rerank(
                reranker,
                question,
                [chunks[chunk_id] for chunk_id in combined_ids],
                config.rerank_top_k,
            )

        evidence_chunks = select_evidence(
            question,
            reranked,
            full_text,
            hard_cap=config.hard_evidence_cap,
            drop_ratio=config.evidence_score_drop_ratio,
            context_chars=config.evidence_context_chars,
        )
        if not evidence_chunks and candidate_chunks:
            evidence_chunks = candidate_chunks[:1]
        answer = synthesize_answer(
            llm,
            reranker,
            question,
            evidence_chunks,
            max_tokens=config.llm_max_new_tokens,
        )
        output_records.append((title, answer, evidence_chunks))
        if config.device == "cuda" and index % 10 == 0:
            try:
                import torch

                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                print(
                    f"GPU memory after {index} items: allocated={allocated:.2f}GB reserved={reserved:.2f}GB",
                    file=sys.stderr,
                )
            except Exception:
                pass
    del llm
    del reranker
    del embedder
    gc.collect()
    return format_submission(output_records)


def save_output(path: str | Path, payload: Sequence[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    config = build_config(args)
    start = time.time()
    data = load_dataset(args.dataset)
    if args.limit is not None:
        data = data[: args.limit]
    results = run_pipeline(data, config)
    save_output(args.output, results)
    elapsed = time.time() - start
    print(f"Saved {len(results)} predictions to {args.output} in {elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
