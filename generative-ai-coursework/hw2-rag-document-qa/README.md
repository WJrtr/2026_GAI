# HW2 — Document Question Answering with RAG

`RAG` `hybrid retrieval (BM25 + dense)` `HyDE` `cross-encoder reranking` `BGE` `Ollama REST API` `Llama-3.2-3B`

## Assignment

Given a research paper's full text and a question about it, build a RAG pipeline that (1) retrieves the supporting evidence passages from the paper and (2) generates a short answer with a small open LLM (Llama-3.2-3B-Instruct). The submission includes both the answer and the retrieved evidence for each question.

> Summarized from the course assignment spec (NYCU Generative AI, Spring 2026). The original spec PDFs are not redistributed.


Answers questions about individual research papers by retrieving supporting evidence from the paper's full text and generating a concise answer with a small local LLM.

## Pipeline

1. **Chunking** — overlapping, multi-granularity sentence windows (2–5 sentences, stride 1), de-duplicated.
2. **HyDE** — the LLM first writes a hypothetical answer passage; it is used together with the question as the search query.
3. **Hybrid retrieval** — BGE dense retrieval (`BAAI/bge-large-en-v1.5`) + BM25 sparse retrieval, top-50 each, fused with Reciprocal Rank Fusion (k = 60).
4. **Reranking** — `BAAI/bge-reranker-v2-m3` cross-encoder scores candidates against the *original* question; top-5 are kept as evidence.
5. **Answer generation** — `Llama-3.2-3B-Instruct` served through Ollama, with answer cleaning and a fallback when the model refuses.

Ablations that were tried and left disabled (kept as config flags in the code): multi-query paraphrasing and MMR diversity selection — both measured no gain or slightly worse.

## Usage

```bash
pip install torch sentence-transformers rank_bm25 requests tqdm numpy
ollama pull llama3.2:3b          # Ollama must be running on localhost:11434
python rag_qa.py --dataset datasets/private_dataset.json --out predictions.json
```

## Files

```
rag_qa.py                 # full pipeline (chunking → HyDE → hybrid retrieval → rerank → answer)
results/predictions.json  # submitted answers + retrieved evidence
```
