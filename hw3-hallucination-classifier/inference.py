from __future__ import annotations

import argparse
from pathlib import Path

import unsloth  # noqa: F401 - import before model loading for Unsloth patching
import pandas as pd

from src.config import (
    DEFAULT_EVIDENCE_WORD_BUDGET,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    MAX_SEQ_LENGTH,
    RERANKER_MODEL_NAME,
)
from src.data_utils import load_classes, load_split
from src.modeling import generate_label, load_model_for_inference
from src.prompts import build_prompt
from src.rag_dataset import load_or_build_rag_records
from src.retrieval import EvidenceRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HW3 test inference and write Kaggle CSV.")
    parser.add_argument("--data_dir", default="datasets")
    parser.add_argument("--adapter_dir", default="adapter_checkpoint")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--output_csv", default="submission.csv")
    parser.add_argument("--embedding_model_name", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--reranker_model_name", default=RERANKER_MODEL_NAME)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank_candidates", type=int, default=DEFAULT_RERANK_CANDIDATES)
    parser.add_argument("--no_reranker", action="store_true")
    parser.add_argument("--evidence_word_budget", type=int, default=DEFAULT_EVIDENCE_WORD_BUDGET)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--rebuild_rag_cache", action="store_true")
    parser.add_argument("--force_parse", action="store_true")
    parser.add_argument("--force_embed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}. Run train.py first.")

    classes = load_classes(args.data_dir)
    test_df = load_split(args.data_dir, "test")
    if args.max_test_samples is not None and args.max_test_samples > 0:
        test_df = test_df.head(args.max_test_samples).copy()

    retriever = EvidenceRetriever(
        args.data_dir,
        args.cache_dir,
        embedding_model_name=args.embedding_model_name,
        reranker_model_name=args.reranker_model_name,
        use_reranker=not args.no_reranker,
        force_parse=args.force_parse,
        force_embed=args.force_embed,
    )
    records = load_or_build_rag_records(
        test_df,
        "test",
        retriever,
        args.cache_dir,
        top_k=args.top_k,
        rerank_candidates=args.rerank_candidates,
        use_reranker=not args.no_reranker,
        evidence_word_budget=args.evidence_word_budget,
        rebuild=args.rebuild_rag_cache,
    )

    print(f"[model] loading adapter from {adapter_dir}", flush=True)
    model, tokenizer = load_model_for_inference(
        model_name_or_adapter=str(adapter_dir),
        max_seq_length=args.max_seq_length,
    )

    rows = []
    total = len(records)
    for pos, record in enumerate(records, start=1):
        if pos == 1 or pos % 100 == 0 or pos == total:
            print(f"[infer] test: {pos}/{total}", flush=True)
        prompt = build_prompt(tokenizer, record["text"], record["evidence"], classes)
        label, raw = generate_label(model, tokenizer, prompt)
        rows.append({"id": int(record["id"]), "label": int(label), "raw_generation": raw})

    output = pd.DataFrame(rows).sort_values("id")
    output[["id", "label"]].to_csv(args.output_csv, index=False)
    debug_path = Path(args.output_csv).with_suffix(".debug.csv")
    output.to_csv(debug_path, index=False)
    print(f"[done] wrote {args.output_csv} with {len(output)} rows")
    print(f"[debug] wrote {debug_path}")


if __name__ == "__main__":
    main()
