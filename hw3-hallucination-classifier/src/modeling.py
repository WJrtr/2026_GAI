"""Model loading, completion-only collation, and generation utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from .config import LORA_TARGET_MODULES, MAX_SEQ_LENGTH, MODEL_NAME
from .data_utils import parse_label_id


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any
    max_length: int = MAX_SEQ_LENGTH

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        eos = self.tokenizer.eos_token or ""
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for feature in features:
            answer_ids = self.tokenizer(feature["answer"] + eos, add_special_tokens=False)["input_ids"]
            prompt_ids = self.tokenizer(feature["prompt"], add_special_tokens=False)["input_ids"]
            max_prompt_len = max(1, self.max_length - len(answer_ids))
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            input_rows.append(input_ids)
            label_rows.append(labels)

        max_len = max(len(row) for row in input_rows)
        pad_id = self.tokenizer.pad_token_id
        padded_inputs = []
        padded_labels = []
        attention_masks = []
        for input_ids, labels in zip(input_rows, label_rows):
            pad_len = max_len - len(input_ids)
            padded_inputs.append(input_ids + [pad_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            attention_masks.append([1] * len(input_ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def load_base_model_for_training(
    *,
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    prepare_tokenizer(tokenizer)
    return model, tokenizer


def load_model_for_inference(
    *,
    model_name_or_adapter: str,
    max_seq_length: int = MAX_SEQ_LENGTH,
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_adapter,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    prepare_tokenizer(tokenizer)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def prepare_tokenizer(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"


def normalize_generated_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


@torch.inference_mode()
def generate_label(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 8,
    max_input_tokens: int = MAX_SEQ_LENGTH - 8,
    default: int = 0,
) -> tuple[int, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    text = normalize_generated_text(tokenizer.decode(generated, skip_special_tokens=True))
    return parse_label_id(text, default=default), text
