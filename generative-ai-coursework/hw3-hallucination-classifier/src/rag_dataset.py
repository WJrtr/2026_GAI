"""Build and cache RAG prompt records for train/dev/test splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_EVIDENCE_WORD_BUDGET, DEFAULT_RERANK_CANDIDATES, DEFAULT_TOP_K, RAG_CACHE_VERSION
from .data_utils import read_jsonl, write_jsonl
from .retrieval import EvidenceRetriever


def rag_cache_path(
    cache_dir: str | Path,
    split: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    use_reranker: bool = True,
    evidence_word_budget: int = DEFAULT_EVIDENCE_WORD_BUDGET,
) -> Path:
    rerank_tag = f"rerank{rerank_candidates}" if use_reranker else "norank"
    return Path(cache_dir) / "rag" / f"{split}_{RAG_CACHE_VERSION}_top{top_k}_{rerank_tag}_budget{evidence_word_budget}.jsonl"


def build_rag_records(
    df: pd.DataFrame,
    split: str,
    retriever: EvidenceRetriever,
    *,
    top_k: int = DEFAULT_TOP_K,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    evidence_word_budget: int = DEFAULT_EVIDENCE_WORD_BUDGET,
) -> list[dict]:
    records = []
    total = len(df)
    for pos, row in enumerate(df.itertuples(index=False), start=1):
        if pos == 1 or pos % 100 == 0 or pos == total:
            print(f"[rag] {split}: {pos}/{total}", flush=True)
        evidence = retriever.retrieve(
            split,
            str(row.paper_id),
            str(row.text),
            top_k=top_k,
            rerank_candidates=rerank_candidates,
            evidence_word_budget=evidence_word_budget,
        )
        record = {
            "id": int(row.id),
            "paper_id": str(row.paper_id),
            "text": str(row.text),
            "evidence": evidence,
        }
        if hasattr(row, "label"):
            record["label"] = int(row.label)
        records.append(record)
    return records


def load_or_build_rag_records(
    df: pd.DataFrame,
    split: str,
    retriever: EvidenceRetriever,
    cache_dir: str | Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    use_reranker: bool = True,
    evidence_word_budget: int = DEFAULT_EVIDENCE_WORD_BUDGET,
    rebuild: bool = False,
) -> list[dict]:
    path = rag_cache_path(
        cache_dir,
        split,
        top_k=top_k,
        rerank_candidates=rerank_candidates,
        use_reranker=use_reranker,
        evidence_word_budget=evidence_word_budget,
    )
    if path.exists() and not rebuild:
        records = read_jsonl(path)
        cached_ids = [int(record["id"]) for record in records]
        expected_ids = [int(row_id) for row_id in df["id"].tolist()]
        if cached_ids == expected_ids:
            return records
        print(
            f"[rag] cache id mismatch for {path}; rebuilding "
            f"(cached={len(cached_ids)}, expected={len(expected_ids)})",
            flush=True,
        )
    records = build_rag_records(
        df,
        split,
        retriever,
        top_k=top_k,
        rerank_candidates=rerank_candidates,
        evidence_word_budget=evidence_word_budget,
    )
    write_jsonl(path, records)
    return records
