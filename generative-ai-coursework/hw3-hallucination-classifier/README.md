# HW3 — Hallucination Type Classification in Peer Reviews (RAG + QLoRA)

`RAG` `QLoRA` `Unsloth` `Qwen2.5-3B` `PDF parsing` `BGE reranker` `class imbalance` `Macro-F1`

## Assignment

Each sample is a claim from an AI-generated peer review together with the reviewed paper (PDF). Classify which **type of hallucination** the claim contains (5 classes, e.g. entity, attribution, over-generalization, number, temporal). Graded by Macro-F1 on a Kaggle leaderboard.

> Summarized from the course assignment spec (NYCU Generative AI, Spring 2026). The original spec PDFs are not redistributed.


This repository implements an evidence-grounded hallucination type classifier for HW3. It parses each paper PDF with PyMuPDF, retrieves evidence with BM25 + BGE dense retrieval, reranks candidates with a BGE cross-encoder, and fine-tunes `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` with QLoRA.

## Environment

The local target environment is:

```bash
conda activate gai-hw3
pip install -r requirements.txt
```

The implementation expects CUDA, PyTorch, Unsloth, PyMuPDF, `rank_bm25`, and `sentence-transformers`. The checked local environment already has these installed.

## Data Layout

Place the unzipped dataset under `datasets/` next to `train.py` and `inference.py`:

```text
datasets/
├─ train.csv
├─ dev.csv
├─ test.csv
├─ classes.json
└─ paper_evidence/
    ├─ train/
    │   └─ paper_*.pdf
    ├─ dev/
    │   └─ paper_*.pdf
    └─ test/
        └─ paper_*.pdf
```

The `cache/` directory will be created automatically on the first run to store parsed PDF chunks, dense embeddings, and retrieved RAG records.

## Train

Full training with dev evaluation:

```bash
python train.py --data_dir datasets --output_dir adapter_checkpoint --cache_dir cache --eval_dev --select_best_checkpoint
```

Fast smoke test:

```bash
python train.py --data_dir datasets --output_dir adapter_checkpoint_smoke --cache_dir cache --max_train_samples 32 --max_eval_samples 20 --max_steps 2 --eval_dev
```

Prepare only the parsing, embedding, and RAG caches:

```bash
python train.py --data_dir datasets --cache_dir cache --prepare_only --eval_dev
```

Useful options:

```bash
--use_dev_for_train       Include dev rows in the final training run.
--rebuild_rag_cache       Rebuild retrieved evidence JSONL files.
--force_parse             Re-parse PDFs even if chunk cache exists.
--force_embed             Recompute dense chunk embeddings.
--no_reranker             Disable BGE cross-encoder reranking and use RRF only.
--select_best_checkpoint  Evaluate saved checkpoints on dev and copy the best adapter to output_dir.
```

## Inference

After training, generate the Kaggle CSV:

```bash
python inference.py --data_dir datasets --adapter_dir adapter_checkpoint --cache_dir cache --output_csv submission.csv
```

The output file has the required columns:

```text
id,label
```

`submission.debug.csv` is also written with raw model generations for inspection.

## Package Layout

```text
train.py
inference.py
src/
adapter_checkpoint/
requirements.txt
README.md
submission.csv
```

Rename `submission.csv` to `hw3_{student_id}.csv` before final E3 packaging.

## Notes

- Test labels are never used.
- PDF parsing uses open-source PyMuPDF (`fitz`).
- RAG evidence is restricted to the row's own `paper_id`.
- Retrieval uses BM25 + BGE dense RRF, exact-match boosts for numbers/time/figure anchors, then BGE reranker over the top 24 candidates by default.
- Prompt rules include calibration examples for the common Entity vs Attribution vs Overgeneralization mistakes.
- Training keeps several checkpoints and can select the best one by dev Macro F1 with `--select_best_checkpoint`.
- Minority classes are oversampled during training: Number x5 and Temporal x10.
- The model prompt asks for exactly one integer label from `0` to `4`.

## Repository Note

The trained LoRA adapter (`adapter_checkpoint/`, ~120 MB) and the course dataset are not included. Run `train.py` to reproduce the adapter. The submitted predictions are in [`results/submission.csv`](results/submission.csv).
