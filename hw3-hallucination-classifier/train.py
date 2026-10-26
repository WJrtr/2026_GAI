from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import unsloth  # noqa: F401 - import before transformers for Unsloth patching
import pandas as pd
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from src.config import (
    DEFAULT_EVIDENCE_WORD_BUDGET,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    OVERSAMPLE_BY_LABEL,
    RERANKER_MODEL_NAME,
)
from src.data_utils import (
    classification_metrics,
    load_classes,
    load_split,
    oversample_records,
    stratified_limit,
    write_jsonl,
)
from src.modeling import (
    CompletionOnlyCollator,
    generate_label,
    load_base_model_for_training,
    load_model_for_inference,
    prepare_tokenizer,
)
from src.prompts import build_prompt
from src.rag_dataset import load_or_build_rag_records
from src.retrieval import EvidenceRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HW3 evidence-grounded QLoRA classifier.")
    parser.add_argument("--data_dir", default="datasets")
    parser.add_argument("--output_dir", default="adapter_checkpoint")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--embedding_model_name", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--reranker_model_name", default=RERANKER_MODEL_NAME)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank_candidates", type=int, default=DEFAULT_RERANK_CANDIDATES)
    parser.add_argument("--no_reranker", action="store_true")
    parser.add_argument("--evidence_word_budget", type=int, default=DEFAULT_EVIDENCE_WORD_BUDGET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_dev", action="store_true")
    parser.add_argument("--use_dev_for_train", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--rebuild_rag_cache", action="store_true")
    parser.add_argument("--force_parse", action="store_true")
    parser.add_argument("--force_embed", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--select_best_checkpoint", action="store_true")
    return parser.parse_args()


def records_to_training_rows(records: list[dict], classes: list[dict], tokenizer) -> list[dict]:
    rows = []
    for record in records:
        prompt = build_prompt(tokenizer, record["text"], record["evidence"], classes)
        rows.append(
            {
                "id": record["id"],
                "paper_id": record["paper_id"],
                "label": int(record["label"]),
                "prompt": prompt,
                "answer": str(int(record["label"])),
            }
        )
    return rows


def evaluate_dev(
    model,
    tokenizer,
    records: list[dict],
    classes: list[dict],
    output_dir: str | Path,
) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []
    rows = []
    total = len(records)
    for pos, record in enumerate(records, start=1):
        if pos == 1 or pos % 100 == 0 or pos == total:
            print(f"[eval] dev: {pos}/{total}", flush=True)
        prompt = build_prompt(tokenizer, record["text"], record["evidence"], classes)
        pred, raw = generate_label(model, tokenizer, prompt)
        truth = int(record["label"])
        y_true.append(truth)
        y_pred.append(pred)
        rows.append(
            {
                "id": record["id"],
                "paper_id": record["paper_id"],
                "label": truth,
                "pred": pred,
                "raw_generation": raw,
            }
        )
    metrics = classification_metrics(y_true, y_pred)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dev_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(output_dir / "dev_predictions.csv", index=False)
    print(f"[eval] macro_f1={metrics['macro_f1']:.4f}")
    print(metrics["report"])
    print("[eval] confusion_matrix=")
    print(metrics["confusion_matrix"])
    return metrics


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else 10**12


def _copy_adapter_artifacts(src_dir: Path, dst_dir: Path) -> None:
    if src_dir.resolve() == dst_dir.resolve():
        return
    names = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "README.md",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "dev_metrics.json",
        "dev_predictions.csv",
    ]
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def select_best_checkpoint(
    output_dir: str | Path,
    dev_records: list[dict],
    classes: list[dict],
    *,
    max_seq_length: int,
) -> dict:
    output_dir = Path(output_dir)
    candidates = sorted(output_dir.glob("checkpoint-*"), key=_checkpoint_step)
    candidates.append(output_dir)
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if not (candidate / "adapter_config.json").exists():
            continue
        resolved = str(candidate.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(candidate)

    if not unique_candidates:
        raise ValueError(f"No adapter checkpoints found in {output_dir}")

    best = {"macro_f1": -1.0, "checkpoint": None}
    summary = []
    for candidate in unique_candidates:
        print(f"[select] evaluating {candidate}", flush=True)
        model, tokenizer = load_model_for_inference(
            model_name_or_adapter=str(candidate),
            max_seq_length=max_seq_length,
        )
        metrics = evaluate_dev(model, tokenizer, dev_records, classes, candidate)
        macro_f1 = float(metrics["macro_f1"])
        summary.append({"checkpoint": str(candidate), "macro_f1": macro_f1})
        if macro_f1 > best["macro_f1"]:
            best = {"macro_f1": macro_f1, "checkpoint": str(candidate)}
        del model
        torch.cuda.empty_cache()

    best_path = Path(str(best["checkpoint"]))
    _copy_adapter_artifacts(best_path, output_dir)
    with (output_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
        json.dump({"best": best, "candidates": summary}, f, ensure_ascii=False, indent=2)
    print(f"[select] best={best_path} macro_f1={best['macro_f1']:.4f}")
    return best


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_DISABLED", "true")
    set_seed(args.seed)

    classes = load_classes(args.data_dir)
    retriever = EvidenceRetriever(
        args.data_dir,
        args.cache_dir,
        embedding_model_name=args.embedding_model_name,
        reranker_model_name=args.reranker_model_name,
        use_reranker=not args.no_reranker,
        force_parse=args.force_parse,
        force_embed=args.force_embed,
    )

    train_df = load_split(args.data_dir, "train")
    train_df = stratified_limit(train_df, args.max_train_samples, args.seed)
    train_records = load_or_build_rag_records(
        train_df,
        "train",
        retriever,
        args.cache_dir,
        top_k=args.top_k,
        rerank_candidates=args.rerank_candidates,
        use_reranker=not args.no_reranker,
        evidence_word_budget=args.evidence_word_budget,
        rebuild=args.rebuild_rag_cache,
    )

    dev_records = []
    if args.eval_dev or args.use_dev_for_train:
        dev_df = load_split(args.data_dir, "dev")
        dev_df = stratified_limit(dev_df, args.max_eval_samples, args.seed)
        dev_records = load_or_build_rag_records(
            dev_df,
            "dev",
            retriever,
            args.cache_dir,
            top_k=args.top_k,
            rerank_candidates=args.rerank_candidates,
            use_reranker=not args.no_reranker,
            evidence_word_budget=args.evidence_word_budget,
            rebuild=args.rebuild_rag_cache,
        )

    if args.prepare_only:
        print("[prepare] RAG caches are ready.")
        return

    print("[model] loading base model for QLoRA training", flush=True)
    model, tokenizer = load_base_model_for_training(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )
    prepare_tokenizer(tokenizer)

    if args.use_dev_for_train and dev_records:
        train_records = train_records + dev_records

    training_rows = records_to_training_rows(train_records, classes, tokenizer)
    training_rows = oversample_records(training_rows, OVERSAMPLE_BY_LABEL, seed=args.seed)
    write_jsonl(Path(args.cache_dir) / "training_records.jsonl", training_rows)
    train_dataset = Dataset.from_list(training_rows)

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=bf16,
        fp16=not bf16,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CompletionOnlyCollator(tokenizer=tokenizer, max_length=args.max_seq_length),
    )
    print(f"[train] rows={len(training_rows)} bf16={bf16}", flush=True)
    trainer.train()

    print(f"[save] writing adapter to {args.output_dir}", flush=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.eval_dev and dev_records:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
        evaluate_dev(model, tokenizer, dev_records, classes, args.output_dir)

    if args.eval_dev and args.select_best_checkpoint and dev_records:
        del trainer
        del model
        torch.cuda.empty_cache()
        select_best_checkpoint(
            args.output_dir,
            dev_records,
            classes,
            max_seq_length=args.max_seq_length,
        )


if __name__ == "__main__":
    main()
