"""Hybrid BM25 + dense retrieval over per-paper evidence chunks."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import (
    DEFAULT_EVIDENCE_WORD_BUDGET,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
)
from .data_utils import pdf_path_for
from .pdf_parser import ensure_chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_.$%+-]+", text.lower())


NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d{1,4}(?:,\d{3})*(?:\.\d+)?%?|\d+(?:\.\d+)?e[+-]?\d+)(?![A-Za-z])")
ANCHOR_RE = re.compile(r"\b(?:fig(?:ure)?|table|section|sec\.|appendix|line|teaser)\s*[A-Za-z0-9.:-]*", re.I)
TEMPORAL_TERMS = {
    "will",
    "would",
    "might",
    "may",
    "could",
    "previously",
    "prior",
    "future",
    "current",
    "existing",
    "confirmed",
    "confirm",
    "before",
    "after",
    "during",
    "later",
    "proposed",
}


def _normalized_numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", "").lower() for match in NUMBER_RE.finditer(text)}


def _anchors(text: str) -> set[str]:
    return {re.sub(r"\s+", " ", match.group(0).lower()).strip() for match in ANCHOR_RE.finditer(text)}


def _anchor_types(text: str) -> set[str]:
    lowered = text.lower()
    types = set()
    if re.search(r"\b(?:fig(?:ure)?|teaser)\b", lowered):
        types.add("figure")
    if re.search(r"\btable\b", lowered):
        types.add("table")
    if re.search(r"\b(?:section|sec\.|appendix)\b", lowered):
        types.add("section")
    if re.search(r"\bline\b", lowered):
        types.add("line")
    return types


def _temporal_terms(text: str) -> set[str]:
    tokens = set(tokenize(text))
    return tokens & TEMPORAL_TERMS


def evidence_boost_score(query: str, chunk_text: str) -> float:
    """Boost exact numerical, temporal, and figure/table/section matches before reranking."""
    q_numbers = _normalized_numbers(query)
    c_numbers = _normalized_numbers(chunk_text)
    shared_numbers = q_numbers & c_numbers

    q_anchors = _anchors(query)
    c_anchors = _anchors(chunk_text)
    shared_anchors = q_anchors & c_anchors
    shared_anchor_types = _anchor_types(query) & _anchor_types(chunk_text)

    q_temporal = _temporal_terms(query)
    c_temporal = _temporal_terms(chunk_text)
    shared_temporal = q_temporal & c_temporal

    score = 0.0
    if shared_numbers:
        score += min(0.08, 0.025 * len(shared_numbers))
    if shared_anchors:
        score += min(0.05, 0.025 * len(shared_anchors))
    if shared_anchor_types:
        score += min(0.03, 0.015 * len(shared_anchor_types))
    if shared_temporal:
        score += min(0.03, 0.01 * len(shared_temporal))
    elif q_temporal and c_temporal:
        score += 0.01
    return score


def embedding_cache_path(cache_dir: str | Path, split: str, paper_id: str) -> Path:
    return Path(cache_dir) / "embeddings" / split / f"{paper_id}.npz"


class EvidenceRetriever:
    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: str | Path,
        *,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
        use_reranker: bool = True,
        force_parse: bool = False,
        force_embed: bool = False,
        device: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.use_reranker = use_reranker
        self.force_parse = force_parse
        self.force_embed = force_embed
        self.device = device
        self._embedding_model: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._embedding_model

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self.reranker_model_name, device=self.device, max_length=512)
        return self._reranker

    def chunks_for(self, split: str, paper_id: str) -> list[dict]:
        pdf_path = pdf_path_for(self.data_dir, split, paper_id)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF evidence: {pdf_path}")
        return ensure_chunks(pdf_path, self.cache_dir, split, force=self.force_parse)

    def embeddings_for(self, split: str, paper_id: str, chunks: list[dict]) -> np.ndarray:
        cache_path = embedding_cache_path(self.cache_dir, split, paper_id)
        if cache_path.exists() and not self.force_embed:
            return np.load(cache_path)["embeddings"]

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
        return embeddings

    @staticmethod
    def _rank_scores(scores: np.ndarray) -> dict[int, int]:
        order = np.argsort(-scores)
        return {int(idx): rank for rank, idx in enumerate(order)}

    def retrieve(
        self,
        split: str,
        paper_id: str,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
        evidence_word_budget: int = DEFAULT_EVIDENCE_WORD_BUDGET,
        rrf_k: int = 60,
    ) -> list[dict]:
        chunks = self.chunks_for(split, paper_id)
        if not chunks:
            return []

        tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = np.asarray(bm25.get_scores(tokenize(query)), dtype=np.float32)

        embeddings = self.embeddings_for(split, paper_id, chunks)
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        dense_scores = embeddings @ np.asarray(query_embedding, dtype=np.float32)

        bm25_ranks = self._rank_scores(bm25_scores)
        dense_ranks = self._rank_scores(dense_scores)
        fused = []
        for idx, chunk in enumerate(chunks):
            rrf_score = 1.0 / (rrf_k + bm25_ranks[idx] + 1) + 1.0 / (rrf_k + dense_ranks[idx] + 1)
            boost_score = evidence_boost_score(query, chunk["text"])
            score = rrf_score + boost_score
            fused.append((score, idx, float(bm25_scores[idx]), float(dense_scores[idx]), boost_score, chunk))
        fused.sort(key=lambda item: item[0], reverse=True)

        candidate_count = max(top_k, rerank_candidates if self.use_reranker else top_k)
        candidates = fused[:candidate_count]
        if self.use_reranker and candidates:
            pairs = [(query, item[-1]["text"]) for item in candidates]
            rerank_scores = self.reranker.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            reranked = []
            for rerank_score, (retrieval_score, idx, bm25_score, dense_score, boost_score, chunk) in zip(
                rerank_scores, candidates
            ):
                adjusted_rerank_score = float(rerank_score) + boost_score
                reranked.append(
                    (adjusted_rerank_score, float(rerank_score), retrieval_score, idx, bm25_score, dense_score, boost_score, chunk)
                )
            reranked.sort(key=lambda item: item[0], reverse=True)
            ranked_items = [
                (retrieval_score, idx, bm25_score, dense_score, boost_score, chunk, rerank_score)
                for _, rerank_score, retrieval_score, idx, bm25_score, dense_score, boost_score, chunk in reranked
            ]
        else:
            ranked_items = [
                (score, idx, bm25_score, dense_score, boost_score, chunk, None)
                for score, idx, bm25_score, dense_score, boost_score, chunk in candidates
            ]

        selected: list[dict] = []
        used_words = 0
        for score, _, bm25_score, dense_score, boost_score, chunk, rerank_score in ranked_items:
            if len(selected) >= top_k:
                break
            words = int(chunk.get("word_count", len(chunk["text"].split())))
            if selected and used_words + words > evidence_word_budget:
                continue
            item = dict(chunk)
            item["retrieval_score"] = float(score)
            item["bm25_score"] = bm25_score
            item["dense_score"] = dense_score
            item["boost_score"] = float(boost_score)
            if rerank_score is not None:
                item["rerank_score"] = float(rerank_score)
            selected.append(item)
            used_words += words

        return selected
