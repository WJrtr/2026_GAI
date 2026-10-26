from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import random
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_SEED = 42
ANSWER_LABELS = ["0", "1", "2", "3"]
SYSTEM_PROMPT = (
    "You are a careful medical pathology multiple-choice expert. "
    "Use the solved examples as references, reason internally, and return only one label."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CoT + few-shot + LoRA PathoQA model with Llama-3.2-1B-Instruct."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-path", type=Path, default=Path("dataset") / "dataset.csv")
    common.add_argument("--benchmark-path", type=Path, default=Path("dataset") / "benchmark.csv")
    common.add_argument("--split-dir", type=Path, default=Path("dataset"))
    common.add_argument("--output-root", type=Path, default=Path("saved_models"))
    common.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    common.add_argument("--hf-token", type=str, default=None)
    common.add_argument("--seed", type=int, default=DEFAULT_SEED)
    common.add_argument("--train-ratio", type=float, default=0.90)
    common.add_argument("--val-ratio", type=float, default=0.05)
    common.add_argument("--test-ratio", type=float, default=0.05)
    common.add_argument("--few-shot-k", type=int, default=3)
    common.add_argument("--few-shot-pool-size", type=int, default=24)
    common.add_argument("--max-length", type=int, default=1024)
    common.add_argument("--num-workers", type=int, default=0)
    common.add_argument("--force-resplit", action="store_true")
    common.add_argument("--disable-4bit", action="store_true")
    common.add_argument("--max-train-samples", type=int, default=None)
    common.add_argument("--max-eval-samples", type=int, default=None)

    train_parser = subparsers.add_parser("train", parents=[common], help="Train LoRA and evaluate.")
    train_parser.add_argument("--run-name", type=str, default=None)
    train_parser.add_argument("--num-epochs", type=float, default=3.0)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.05)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.06)
    train_parser.add_argument("--train-batch-size", type=int, default=2)
    train_parser.add_argument("--eval-batch-size", type=int, default=2)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train_parser.add_argument("--logging-steps", type=int, default=25)
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=64)
    train_parser.add_argument("--lora-dropout", type=float, default=0.1)
    train_parser.add_argument("--predict-benchmark", action="store_true")
    train_parser.add_argument("--early-stopping-patience", type=int, default=2)
    train_parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    predict_parser = subparsers.add_parser("predict", parents=[common], help="Predict benchmark with a saved adapter.")
    predict_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    predict_parser.add_argument("--output-file", type=Path, default=None)
    predict_parser.add_argument("--eval-batch-size", type=int, default=8)

    prepare_parser = subparsers.add_parser("prepare", parents=[common], help="Only create train/val/test splits.")
    prepare_parser.add_argument("--preview-count", type=int, default=3)

    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    value = str(text)
    value = html.unescape(value)
    value = value.replace("\ufeff", " ").replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def load_dataframe(csv_path: Path, has_answer: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = ["question_id", "question", "opa", "opb", "opc", "opd"]
    if has_answer:
        required_columns.append("ans")

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    for column in ["question", "opa", "opb", "opc", "opd"]:
        df[column] = df[column].map(clean_text)
    df["question_id"] = df["question_id"].astype(str)
    if has_answer:
        df["ans"] = df["ans"].astype(str)
    return df


def check_split_ratios(args: argparse.Namespace) -> None:
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, but got {total:.4f}."
        )


def create_or_load_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    check_split_ratios(args)
    ensure_dir(args.split_dir)

    train_path = args.split_dir / "train.csv"
    val_path = args.split_dir / "val.csv"
    test_path = args.split_dir / "test.csv"

    if not args.force_resplit and train_path.exists() and val_path.exists() and test_path.exists():
        return {
            "train": load_dataframe(train_path, has_answer=True),
            "val": load_dataframe(val_path, has_answer=True),
            "test": load_dataframe(test_path, has_answer=True),
        }

    df = load_dataframe(args.data_path, has_answer=True)
    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_ratio,
        random_state=args.seed,
        stratify=df["ans"],
    )
    relative_val_ratio = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        random_state=args.seed,
        stratify=temp_df["ans"],
    )

    train_df = train_df.sort_values("question_id").reset_index(drop=True)
    val_df = val_df.sort_values("question_id").reset_index(drop=True)
    test_df = test_df.sort_values("question_id").reset_index(drop=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {"train": train_df, "val": val_df, "test": test_df}


def format_options(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"0. {row['opa']}",
            f"1. {row['opb']}",
            f"2. {row['opc']}",
            f"3. {row['opd']}",
        ]
    )


def build_retrieval_text(row: dict[str, Any]) -> str:
    return " [SEP] ".join([row["question"], row["opa"], row["opb"], row["opc"], row["opd"]])


def build_cot_rationale(row: dict[str, Any]) -> str:
    answer_idx = int(row["ans"])
    answer_text = [row["opa"], row["opb"], row["opc"], row["opd"]][answer_idx]
    question_lower = row["question"].lower()

    if " not " in f" {question_lower} " or "except" in question_lower:
        lead = "The stem asks for the exception among the choices."
    elif any(keyword in question_lower for keyword in ["most", "best", "characteristic", "common"]):
        lead = "The stem asks for the single best pathology fact."
    elif any(keyword in question_lower for keyword in ["associated with", "seen in", "indicative", "contains"]):
        lead = "We need the option that is correctly associated with the clue in the stem."
    else:
        lead = "We compare the options against the pathology concept described in the question."

    return f'{lead} Among the four choices, "{answer_text}" is the strongest match, so it is the correct answer.'


def format_few_shot_example(row: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        Question: {row["question"]}
        Options:
        {format_options(row)}
        Reasoning: {build_cot_rationale(row)}
        Final Answer: {row["ans"]}
        """
    ).strip()


def build_user_prompt(target_row: dict[str, Any], support_rows: list[dict[str, Any]]) -> str:
    parts = ["Solved pathology examples:"]
    for index, row in enumerate(support_rows, start=1):
        parts.append(f"[Example {index}]\n{format_few_shot_example(row)}")
    parts.extend(
        [
            "Now solve the next question.",
            f"Question: {target_row['question']}",
            "Options:",
            format_options(target_row),
            "Follow the same pathology reasoning style internally, then return only one label from 0, 1, 2, 3.",
        ]
    )
    return "\n\n".join(parts)


def resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token.strip()
    for env_name in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
        value = os.getenv(env_name)
        if value:
            return value.strip()
    for token_path in [Path("token.txt"), Path(".hf_token"), Path.home() / ".cache" / "huggingface" / "token"]:
        if token_path.exists():
            value = token_path.read_text(encoding="utf-8").strip()
            if value:
                return value
    try:
        from huggingface_hub import get_token

        value = get_token()
        if value:
            return value.strip()
    except Exception:
        pass
    return None


def ensure_token_id_list(encoded: Any) -> list[int]:
    if isinstance(encoded, list):
        return [int(token_id) for token_id in encoded]
    if hasattr(encoded, "get") and encoded.get("input_ids") is not None:
        input_ids = encoded.get("input_ids")
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return ensure_token_id_list(input_ids)
    if hasattr(encoded, "ids"):
        return [int(token_id) for token_id in encoded.ids]
    if hasattr(encoded, "tolist"):
        values = encoded.tolist()
        return [int(token_id) for token_id in values]
    raise TypeError(f"Unsupported token container type: {type(encoded)}")


def validate_model_access(model_name: str, hf_token: str | None) -> None:
    if Path(model_name).exists():
        return
    if model_name.startswith("meta-llama/") and not hf_token:
        raise RuntimeError(
            "The required gated model needs a Hugging Face token. "
            "Put it in token.txt, run `huggingface-cli login`, or pass `--hf-token`."
        )


@dataclass
class RetrievalResult:
    neighbor_indices: list[list[int]]


class FewShotRetriever:
    def __init__(self, train_records: list[dict[str, Any]]):
        train_texts = [build_retrieval_text(row) for row in train_records]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        )
        self.train_matrix = self.vectorizer.fit_transform(train_texts)
        self.qid_to_index = {row["question_id"]: idx for idx, row in enumerate(train_records)}
        self.train_labels = [row["ans"] for row in train_records]

    def retrieve(
        self,
        query_records: list[dict[str, Any]],
        k: int,
        pool_size: int,
        exclude_same_question: bool,
    ) -> RetrievalResult:
        if k <= 0:
            return RetrievalResult(neighbor_indices=[[] for _ in query_records])

        query_texts = [build_retrieval_text(row) for row in query_records]
        query_matrix = self.vectorizer.transform(query_texts)

        neighbor_indices: list[list[int]] = []
        batch_size = 256
        train_size = self.train_matrix.shape[0]
        top_n = min(max(pool_size, k), train_size)

        for start in range(0, len(query_records), batch_size):
            end = min(start + batch_size, len(query_records))
            similarities = (query_matrix[start:end] @ self.train_matrix.T).toarray()
            for offset, row in enumerate(query_records[start:end]):
                sims = similarities[offset]
                if exclude_same_question:
                    own_index = self.qid_to_index.get(row["question_id"])
                    if own_index is not None:
                        sims[own_index] = -np.inf
                candidate_indices = np.argpartition(sims, -top_n)[-top_n:]
                sorted_indices = candidate_indices[np.argsort(sims[candidate_indices])[::-1]]
                neighbor_indices.append(self._diversify(sorted_indices.tolist(), k))

        return RetrievalResult(neighbor_indices=neighbor_indices)

    def _diversify(self, candidate_indices: list[int], k: int) -> list[int]:
        selected: list[int] = []
        seen_labels: set[str] = set()
        for idx in candidate_indices:
            label = self.train_labels[idx]
            if label not in seen_labels:
                selected.append(idx)
                seen_labels.add(label)
            if len(selected) == k:
                return selected
        for idx in candidate_indices:
            if idx not in selected:
                selected.append(idx)
            if len(selected) == k:
                return selected
        return selected


class PathoTrainDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        records: list[dict[str, Any]],
        support_records: list[list[dict[str, Any]]],
        max_length: int,
    ) -> None:
        self.samples: list[dict[str, Any]] = []
        for row, support_rows in zip(records, support_records):
            prompt_ids = self._encode_prompt(tokenizer, row, support_rows)
            answer_ids = tokenizer(row["ans"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            prompt_ids = prompt_ids[-(max_length - len(answer_ids)) :]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                    "length": len(input_ids),
                }
            )

    @staticmethod
    def _encode_prompt(
        tokenizer: AutoTokenizer,
        row: dict[str, Any],
        support_rows: list[dict[str, Any]],
    ) -> list[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row, support_rows)},
        ]
        return ensure_token_id_list(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class PathoInferenceDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        records: list[dict[str, Any]],
        support_records: list[list[dict[str, Any]]],
        max_length: int,
    ) -> None:
        self.records = records
        self.samples: list[dict[str, Any]] = []
        for row, support_rows in zip(records, support_records):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row, support_rows)},
            ]
            prompt_ids = ensure_token_id_list(
                tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            )
            prompt_ids = prompt_ids[-max_length:]
            self.samples.append(
                {
                    "input_ids": prompt_ids,
                    "attention_mask": [1] * len(prompt_ids),
                    "length": len(prompt_ids),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class PathoDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, include_labels: bool) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        self.include_labels = include_labels
        self.padding_side = tokenizer.padding_side

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            pad_size = max_length - len(feature["input_ids"])
            if self.padding_side == "left":
                batch_input_ids.append([self.pad_token_id] * pad_size + feature["input_ids"])
                batch_attention_mask.append([0] * pad_size + feature["attention_mask"])
                if self.include_labels:
                    batch_labels.append([-100] * pad_size + feature["labels"])
            else:
                batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_size)
                batch_attention_mask.append(feature["attention_mask"] + [0] * pad_size)
                if self.include_labels:
                    batch_labels.append(feature["labels"] + [-100] * pad_size)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }
        if self.include_labels:
            batch["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        return batch


def df_to_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        df = df.iloc[:limit].copy()
    return df.to_dict(orient="records")


def get_precision_flags() -> tuple[bool, bool, torch.dtype]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False, torch.bfloat16
    return False, True, torch.float16


def maybe_build_quantization_config(disable_4bit: bool, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if disable_4bit:
        return None
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_name: str, hf_token: str | None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(
    model_name: str,
    hf_token: str | None,
    quantization_config: BitsAndBytesConfig | None,
) -> tuple[AutoModelForCausalLM, BitsAndBytesConfig | None]:
    _, _, compute_dtype = get_precision_flags()
    common_kwargs: dict[str, Any] = {"token": hf_token, "trust_remote_code": False}
    if torch.cuda.is_available():
        common_kwargs["device_map"] = {"": 0}

    if quantization_config is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                **common_kwargs,
            )
            model.config.use_cache = False
            return model, quantization_config
        except Exception as exc:
            print(f"4-bit loading failed, fallback to standard LoRA mode: {exc}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
        **common_kwargs,
    )
    model.config.use_cache = False
    return model, None


def build_lora_model(
    base_model: AutoModelForCausalLM,
    quantization_config: BitsAndBytesConfig | None,
    args: argparse.Namespace,
) -> AutoModelForCausalLM:
    if quantization_config is not None:
        base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def build_output_dir(args: argparse.Namespace) -> Path:
    ensure_dir(args.output_root)
    run_name = args.run_name or f"llama32_1b_cot_fs_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    ensure_dir(output_dir)
    return output_dir


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, indent=2, ensure_ascii=False)


def build_support_records(
    retriever: FewShotRetriever,
    train_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    args: argparse.Namespace,
    exclude_same_question: bool,
) -> list[list[dict[str, Any]]]:
    retrieval = retriever.retrieve(
        query_records=target_records,
        k=args.few_shot_k,
        pool_size=args.few_shot_pool_size,
        exclude_same_question=exclude_same_question,
    )
    return [[train_records[idx] for idx in indices] for indices in retrieval.neighbor_indices]


def build_training_args(
    args: argparse.Namespace,
    output_dir: Path,
    bf16: bool,
    fp16: bool,
    quantization_config: BitsAndBytesConfig | None,
) -> TrainingArguments:
    optimizer_name = "paged_adamw_8bit" if quantization_config is not None else "adamw_torch"
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optimizer_name,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        use_cache=False,
    )


@torch.inference_mode()
def predict_generate(
    model: AutoModelForCausalLM,
    dataset: PathoInferenceDataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """
    Generate-based inference: the model freely generates text and the predicted
    label (0-3) is extracted from the generated output via regex.
    This is the only allowed inference method per homework requirements.
    """
    collator = PathoDataCollator(tokenizer, include_labels=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    device = next(model.parameters()).device

    predictions: list[str] = []
    confidences: list[float] = []
    gold_answers: list[str | None] = []
    sample_index = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        input_lengths = input_ids.shape[1]
        generated_ids = outputs[:, input_lengths:]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for batch_offset, gen_text in enumerate(generated_texts):
            row = dataset.records[sample_index + batch_offset]

            match = re.search(r'[0123]', gen_text)
            if match:
                pred_label = match.group()
            else:
                pred_label = str(random.choice([0, 1, 2, 3]))

            predictions.append(pred_label)
            confidences.append(1.0)
            gold_answers.append(row.get("ans"))

        sample_index += len(generated_texts)

    frame = pd.DataFrame(
        {
            "question_id": [row["question_id"] for row in dataset.records],
            "prediction": predictions,
            "confidence": confidences,
        }
    )
    if any(answer is not None for answer in gold_answers):
        frame["gold_answer"] = gold_answers
    return frame


def evaluate_prediction_frame(frame: pd.DataFrame) -> dict[str, float]:
    if "gold_answer" not in frame.columns:
        return {}
    return {"accuracy": float(accuracy_score(frame["gold_answer"], frame["prediction"]))}


def save_prediction_artifacts(frame: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    ensure_dir(output_dir)
    frame.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)

    if "gold_answer" not in frame.columns:
        return

    matrix = confusion_matrix(frame["gold_answer"], frame["prediction"], labels=ANSWER_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(ANSWER_LABELS)))
    ax.set_yticks(range(len(ANSWER_LABELS)))
    ax.set_xticklabels(ANSWER_LABELS)
    ax.set_yticklabels(ANSWER_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{prefix.capitalize()} Confusion Matrix")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, int(matrix[row_idx, col_idx]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def plot_training_history(log_history: list[dict[str, Any]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    train_points = [entry for entry in log_history if "loss" in entry and "eval_loss" not in entry]
    eval_points = [entry for entry in log_history if "eval_loss" in entry]
    if not train_points and not eval_points:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if train_points:
        axes[0].plot(
            [entry.get("epoch") for entry in train_points],
            [entry["loss"] for entry in train_points],
            marker="o",
            label="Train loss",
        )
    if eval_points:
        axes[0].plot(
            [entry.get("epoch") for entry in eval_points],
            [entry["eval_loss"] for entry in eval_points],
            marker="s",
            label="Validation loss",
        )
    axes[0].set_title("Training / Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(log_history).to_csv(output_dir / "log_history.csv", index=False)


def print_split_summary(splits: dict[str, pd.DataFrame]) -> None:
    for name, df in splits.items():
        distribution = df["ans"].value_counts(normalize=True).sort_index()
        distribution_text = ", ".join([f"{label}:{ratio:.3f}" for label, ratio in distribution.items()])
        print(f"{name}: {len(df)} rows | label distribution -> {distribution_text}")


def prepare_command(args: argparse.Namespace) -> None:
    splits = create_or_load_splits(args)
    print_split_summary(splits)
    preview = {name: df.head(args.preview_count).to_dict(orient="records") for name, df in splits.items()}
    preview_path = args.split_dir / "split_preview.json"
    save_json(preview_path, preview)
    print(f"Saved split preview to {preview_path}")


def train_command(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    hf_token = resolve_hf_token(args.hf_token)
    validate_model_access(args.model_name, hf_token)

    splits = create_or_load_splits(args)
    benchmark_df = load_dataframe(args.benchmark_path, has_answer=False)

    output_dir = build_output_dir(args)
    ensure_dir(output_dir / "artifacts")
    ensure_dir(output_dir / "predictions")
    save_json(output_dir / "run_config.json", vars(args))

    train_records = df_to_records(splits["train"], args.max_train_samples)
    val_records = df_to_records(splits["val"], args.max_eval_samples)
    test_records = df_to_records(splits["test"], args.max_eval_samples)
    benchmark_records = df_to_records(benchmark_df)

    retriever = FewShotRetriever(train_records)
    train_support = build_support_records(retriever, train_records, train_records, args, exclude_same_question=True)
    val_support = build_support_records(retriever, train_records, val_records, args, exclude_same_question=False)
    test_support = build_support_records(retriever, train_records, test_records, args, exclude_same_question=False)
    benchmark_support = build_support_records(retriever, train_records, benchmark_records, args, exclude_same_question=False)

    tokenizer = load_tokenizer(args.model_name, hf_token)

    train_dataset = PathoTrainDataset(tokenizer, train_records, train_support, args.max_length)
    val_dataset = PathoTrainDataset(tokenizer, val_records, val_support, args.max_length)
    test_eval_dataset = PathoTrainDataset(tokenizer, test_records, test_support, args.max_length)
    val_inference_dataset = PathoInferenceDataset(tokenizer, val_records, val_support, args.max_length)
    test_inference_dataset = PathoInferenceDataset(tokenizer, test_records, test_support, args.max_length)
    benchmark_inference_dataset = PathoInferenceDataset(tokenizer, benchmark_records, benchmark_support, args.max_length)

    bf16, fp16, compute_dtype = get_precision_flags()
    quantization_config = maybe_build_quantization_config(args.disable_4bit, compute_dtype)
    base_model, quantization_config = load_base_model(args.model_name, hf_token, quantization_config)
    model = build_lora_model(base_model, quantization_config, args)

    trainer = Trainer(
        model=model,
        args=build_training_args(args, output_dir, bf16, fp16, quantization_config),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PathoDataCollator(tokenizer, include_labels=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    train_metrics = train_result.metrics
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_eval_dataset, metric_key_prefix="test")
    plot_training_history(trainer.state.log_history, output_dir / "artifacts")

    val_predictions = predict_generate(
        model=trainer.model,
        dataset=val_inference_dataset,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    test_predictions = predict_generate(
        model=trainer.model,
        dataset=test_inference_dataset,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    val_prediction_metrics = evaluate_prediction_frame(val_predictions)
    test_prediction_metrics = evaluate_prediction_frame(test_predictions)
    save_prediction_artifacts(val_predictions, output_dir / "predictions", "val")
    save_prediction_artifacts(test_predictions, output_dir / "predictions", "test")

    metrics_payload = {
        "train": train_metrics,
        "val_eval": val_metrics,
        "test_eval": test_metrics,
        "val_prediction": val_prediction_metrics,
        "test_prediction": test_prediction_metrics,
    }
    save_json(output_dir / "metrics.json", metrics_payload)

    if args.predict_benchmark:
        benchmark_predictions = predict_generate(
            model=trainer.model,
            dataset=benchmark_inference_dataset,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
        benchmark_predictions.to_csv(output_dir / "predictions" / "benchmark_predictions.csv", index=False)
        benchmark_predictions[["question_id", "prediction"]].rename(columns={"prediction": "ans"}).to_csv(
            output_dir / "predictions" / "benchmark_submission.csv",
            index=False,
        )

    print(json.dumps(to_jsonable(metrics_payload), indent=2))
    print(f"Best adapter saved to: {output_dir}")


def predict_command(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    hf_token = resolve_hf_token(args.hf_token)
    validate_model_access(args.model_name, hf_token)

    tokenizer = load_tokenizer(args.model_name, hf_token)

    splits = create_or_load_splits(args)
    benchmark_df = load_dataframe(args.benchmark_path, has_answer=False)
    train_records = df_to_records(splits["train"], args.max_train_samples)
    benchmark_records = df_to_records(benchmark_df)

    retriever = FewShotRetriever(train_records)
    benchmark_support = build_support_records(retriever, train_records, benchmark_records, args, exclude_same_question=False)
    benchmark_dataset = PathoInferenceDataset(tokenizer, benchmark_records, benchmark_support, args.max_length)

    _, _, compute_dtype = get_precision_flags()
    quantization_config = maybe_build_quantization_config(args.disable_4bit, compute_dtype)
    base_model, _ = load_base_model(args.model_name, hf_token, quantization_config)
    model = PeftModel.from_pretrained(base_model, str(args.checkpoint_dir))

    predictions = predict_generate(
        model=model,
        dataset=benchmark_dataset,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    output_file = args.output_file or (args.checkpoint_dir / "benchmark_submission.csv")
    predictions[["question_id", "prediction"]].rename(columns={"prediction": "ans"}).to_csv(output_file, index=False)
    print(f"Saved benchmark submission to {output_file}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()