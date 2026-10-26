"""
HW2: Document QA based on RAG
Student ID: 112511086

Per-paper RAG pipeline:
  1. Sentence-window chunking
  2. HyDE (Hypothetical Document Embeddings) Generation
  3. Hybrid retrieval: BGE dense + BM25 sparse (using Question + HyDE), fused via RRF
  4. Rerank with BGE cross-encoder (using original Question)
  5. Answer with Llama-3.2-3B-Instruct (via Ollama)

LLM:        meta-llama/Llama-3.2-3B-Instruct  (Ollama: llama3.2:3b)
Embedding:  BAAI/bge-large-en-v1.5
Reranker:   BAAI/bge-reranker-v2-m3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import requests
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "BAAI/bge-large-en-v1.5"
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"
OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "llama3.2:3b"

WINDOW_SIZES  = [2, 3, 4, 5]   # sentence window sizes (multi-granularity if list >1)
WINDOW_STRIDE = 1
CHUNK_MIN_LEN = 30
CHUNK_MAX_LEN = 1200

USE_HYDE      = True   # Enable Hypothetical Document Embeddings
USE_MULTI_QUERY = False # Multi-query gave no evidence gain on [3] windows
NUM_PARAPHRASES = 2    # (kept for future re-enable)
TOP_DENSE     = 50
TOP_SPARSE    = 50
RRF_K         = 60
TOP_RERANK    = 5      # final evidence count (k in submission)
USE_MMR       = False  # MMR measured slightly worse than plain rerank
MMR_LAMBDA    = 0.7    # 1.0=pure relevance, 0.0=pure diversity
MMR_POOL      = 20     # rerank top pool to apply MMR on

BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> List[str]:
    """Split a document into sentences while being robust to newlines."""
    text = text.replace("\r", "")
    sentences: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts = _SENT_SPLIT.split(para)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


def build_chunks(full_text: str) -> List[str]:
    """Overlapping multi-granularity sentence-window chunks."""
    sents = split_sentences(full_text)
    if not sents:
        return [full_text.strip()[:CHUNK_MAX_LEN]] if full_text.strip() else []

    chunks: List[str] = []
    seen = set()
    for w in WINDOW_SIZES:
        for i in range(0, len(sents), WINDOW_STRIDE):
            window = sents[i : i + w]
            if not window:
                break
            chunk = " ".join(window).strip()
            if len(chunk) < CHUNK_MIN_LEN:
                continue
            if len(chunk) > CHUNK_MAX_LEN:
                chunk = chunk[:CHUNK_MAX_LEN]
            if chunk in seen:
                continue
            seen.add(chunk)
            chunks.append(chunk)
            if i + w >= len(sents):
                break
    if not chunks:
        chunks = [full_text.strip()[:CHUNK_MAX_LEN]]
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


def rrf_fuse(ranked_lists: List[List[int]], k: int = RRF_K) -> List[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda i: -scores[i])


class Retriever:
    def __init__(self, embedder: SentenceTransformer, reranker: CrossEncoder):
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, question: str, search_queries: List[str], chunks: List[str]) -> List[str]:
        """
        question: The original question used for cross-encoder reranking.
        search_queries: List of queries for dense/sparse retrieval; ranked lists RRF-merged.
        """
        if len(chunks) <= TOP_RERANK:
            return self._rerank(question, chunks)[:TOP_RERANK]

        # Embed all candidate chunks once.
        c_emb = self.embedder.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        tokenized_chunks = [tokenize(c) for c in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        # For each search query, get dense and sparse ranked lists.
        ranked_lists: List[List[int]] = []
        # Embed all search queries in one batch for speed.
        q_embs = self.embedder.encode(
            [BGE_QUERY_PROMPT + q for q in search_queries],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for qi, sq in enumerate(search_queries):
            dense_scores = c_emb @ q_embs[qi]
            dense_top = np.argsort(-dense_scores)[:TOP_DENSE].tolist()
            ranked_lists.append(dense_top)

            bm25_scores = bm25.get_scores(tokenize(sq))
            sparse_top = np.argsort(-bm25_scores)[:TOP_SPARSE].tolist()
            ranked_lists.append(sparse_top)

        # RRF fuse all ranked lists.
        fused = rrf_fuse(ranked_lists)
        candidates = [chunks[i] for i in fused[: max(TOP_DENSE, TOP_SPARSE)]]

        # Rerank: MUST use the original question for accuracy!
        ranked = self._rerank(question, candidates)
        if USE_MMR and len(ranked) > TOP_RERANK:
            pool = ranked[:MMR_POOL]
            return self._mmr_select(pool, k=TOP_RERANK)
        return ranked[:TOP_RERANK]

    def _rerank(self, question: str, candidates: List[str]) -> List[str]:
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates
        pairs = [(question, c) for c in candidates]
        scores = self.reranker.predict(
            pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        order = np.argsort(-np.asarray(scores))
        return [candidates[i] for i in order]

    def _mmr_select(self, pool: List[str], k: int) -> List[str]:
        """Select k items from a relevance-ordered pool maximizing relevance-diversity tradeoff."""
        if len(pool) <= k:
            return pool
        # Embed pool for diversity calc.
        emb = self.embedder.encode(
            pool, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        # Relevance = descending rank prior (1/(i+1)); tie-break by rank.
        # Use rank-based relevance since we've lost the cross-encoder scores at this point.
        rel = np.array([1.0 / (i + 1) for i in range(len(pool))], dtype=np.float32)

        selected: List[int] = [0]
        remaining = list(range(1, len(pool)))
        while len(selected) < k and remaining:
            best_idx = None
            best_score = -1e9
            sel_emb = emb[selected]
            for r in remaining:
                sim = float(np.max(emb[r] @ sel_emb.T))  # max cosine to selected
                mmr_score = MMR_LAMBDA * rel[r] - (1.0 - MMR_LAMBDA) * sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = r
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [pool[i] for i in selected]


# ──────────────────────────────────────────────────────────────────────────────
# LLM (Ollama)
# ──────────────────────────────────────────────────────────────────────────────
_IDK_RE = re.compile(
    r"i (don'?t|do not|cannot|can'?t) know"
    r"|i'?m not sure"
    r"|not (enough|sufficient) (information|context|detail)"
    r"|the (context|document|passage|text) does(n'?t| not) (contain|mention|provide|have|include|say)",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "You extract short factual answers from evidence snippets of a research paper.\n"
    "\n"
    "Rules:\n"
    "- Answer ONLY from the evidence. Do not invent.\n"
    "- Keep answers SHORT. Prefer 1-15 words. Only go longer if the question truly demands it.\n"
    "- Never echo the question's phrasing back (e.g. if asked 'what surface-level features' don't answer 'surface-level features' — give the actual items).\n"
    "- No preamble. Do not say 'According to', 'The paper', 'The evidence', 'The text', 'We use', 'They propose'.\n"
    "- Copy exact terminology from the evidence (names, metric names, model names, numbers).\n"
    "- DO NOT include citation tokens like 'BIBREF4', 'INLINEFORM2', or parenthetical dates — strip them.\n"
    "- LIST QUESTIONS ('what X', 'which Y', 'what are the Zs', 'name the ...'): "
    "return ALL matching items as a comma-separated list, using the exact names, no descriptions.\n"
    "- 'How/Why' questions: copy the shortest clause that answers it; do not explain or expand.\n"
    "- Numeric questions: return just the number (with unit).\n"
    "- Yes/no questions: start with Yes or No, then a brief reason.\n"
    "- Never refuse. Never say 'I don't know' or 'not enough information'. "
    "Always produce a best-effort answer derived from the evidence.\n"
    "- Output the answer only, no explanations or meta-commentary.\n"
    "\n"
    "Examples (placeholders; copy names ONLY from your evidence, never from these examples):\n"
    "Q: What datasets do they use?\n"
    "A: <Dataset-A>, <Dataset-B>, <Dataset-C>\n"
    "\n"
    "Q: What is the proposed method called?\n"
    "A: <Method Name> (<ACRONYM>)\n"
    "\n"
    "Q: How do they handle out-of-vocabulary words?\n"
    "A: By using <technique> with <specific detail from evidence>.\n"
    "\n"
    "Q: What metric improves over the baseline?\n"
    "A: <metric> improves by <number> over the <baseline name>.\n"
)

# ── HyDE Prompt ──
_HYDE_SYSTEM_PROMPT = (
    "You are an AI research assistant. Your task is to write a short, hypothetical excerpt from a research paper "
    "that directly answers the user's question. This hypothetical text will be used for semantic retrieval.\n\n"
    "Rules:\n"
    "- Write exactly 2 to 3 sentences.\n"
    "- Use formal academic terminology relevant to NLP and machine learning.\n"
    "- DO NOT include any preamble, introduction, or meta-commentary (e.g., do not say 'Here is an excerpt' or 'Answer:').\n"
    "- Focus on what the actual text in the paper might sound like."
)

def build_user_prompt(title: str, question: str, evidence: List[str]) -> str:
    ev_block = "\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence))
    return (
        f"Paper title: {title}\n\n"
        f"Evidence:\n{ev_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

def call_ollama(system: str, user: str, retries: int = 3, temperature: float = 0.1, num_predict: int = 220) -> str:
    """Enhanced to allow overriding generation parameters."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": num_predict,
            "seed": 42,
        },
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as exc:
            last_err = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Ollama call failed after {retries} retries: {last_err}")


def generate_hypothetical_document(title: str, question: str) -> str:
    """Generate a hypothetical document snippet for HyDE."""
    prompt = f"Paper title: {title}\nQuestion: {question}\nHypothetical excerpt:"
    try:
        # We limit num_predict to 100 for speed since we only need a short snippet
        res = call_ollama(_HYDE_SYSTEM_PROMPT, prompt, retries=2, temperature=0.3, num_predict=100)
        # Strip trailing quotes if generated
        if len(res) >= 2 and res[0] in "\"'" and res[-1] == res[0]:
            res = res[1:-1].strip()
        return res
    except Exception as e:
        print(f"[warn] HyDE generation failed: {e}", file=sys.stderr)
        return ""


_PARAPHRASE_SYSTEM_PROMPT = (
    "You rephrase questions for document retrieval. Given a research-paper question, "
    "produce N alternate phrasings that preserve the intent but use different wording or emphasis.\n"
    "Rules:\n"
    "- Output only the paraphrases, one per line, no numbering, no preamble.\n"
    "- Each paraphrase is a complete question-like phrase.\n"
    "- Vary vocabulary (synonyms, related technical terms) but keep the semantic intent.\n"
)


def generate_paraphrases(title: str, question: str, n: int = NUM_PARAPHRASES) -> List[str]:
    """Generate N paraphrases of the question for multi-query retrieval."""
    prompt = (
        f"Paper title: {title}\n"
        f"Original question: {question}\n"
        f"Produce {n} alternate phrasings, one per line:"
    )
    try:
        res = call_ollama(_PARAPHRASE_SYSTEM_PROMPT, prompt, retries=2, temperature=0.4, num_predict=140)
        lines = [ln.strip().lstrip("-*0123456789. )") for ln in res.split("\n") if ln.strip()]
        # Filter out accidental preamble lines
        lines = [ln for ln in lines if len(ln) > 5 and not ln.lower().startswith(("here", "sure", "paraphrase"))]
        return lines[:n]
    except Exception as e:
        print(f"[warn] paraphrase generation failed: {e}", file=sys.stderr)
        return []


_BIBREF_RE = re.compile(r"\s*\b(?:BIBREF|INLINEFORM|DISPLAYFORM|FLOATREF|TABREF|SECREF)\s*\d+\b\s*", re.IGNORECASE)
_PARENS_BIB_RE = re.compile(r"\s*[\(\[][^()\[\]]*?(?:BIBREF|INLINEFORM)\s*\d+[^()\[\]]*?[\)\]]\s*", re.IGNORECASE)


def clean_answer(ans: str) -> str:
    ans = ans.strip()
    # Drop leading "Answer:" if the model echoed it.
    ans = re.sub(r"^\s*answer\s*:\s*", "", ans, flags=re.IGNORECASE)
    # Strip surrounding quotes.
    if len(ans) >= 2 and ans[0] in "\"'" and ans[-1] == ans[0]:
        ans = ans[1:-1].strip()
    # Strip paper placeholders (BIBREF4, INLINEFORM2, etc. — noise for the judge).
    ans = _PARENS_BIB_RE.sub(" ", ans)
    ans = _BIBREF_RE.sub(" ", ans)
    # Collapse whitespace and fix up ", ," / " ," artifacts from stripping.
    ans = re.sub(r"\s+,", ",", ans)
    ans = re.sub(r",\s*,", ",", ans)
    ans = re.sub(r"\s{2,}", " ", ans).strip(" ,.;:")
    return ans


def answer_with_fallback(title: str, question: str, evidence: List[str]) -> str:
    prompt = build_user_prompt(title, question, evidence)
    raw = call_ollama(_SYSTEM_PROMPT, prompt)
    ans = clean_answer(raw)
    if not ans or _IDK_RE.search(ans):
        # Fallback: return the most relevant evidence sentence (shortened).
        top = evidence[0] if evidence else ""
        sents = split_sentences(top)
        ans = sents[0] if sents else top
    return ans


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/private_dataset.json")
    ap.add_argument("--out", default="112511086.json")
    args = ap.parse_args()

    print(f"[config] device={DEVICE}  embed={EMBED_MODEL}  rerank={RERANK_MODEL}")
    print(f"[config] llm={OLLAMA_MODEL}  k={TOP_RERANK}  windows={WINDOW_SIZES}  hyde={USE_HYDE}  mq={USE_MULTI_QUERY}({NUM_PARAPHRASES})")

    # Load models once.
    print("[load] embedding model …")
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    print("[load] reranker …")
    reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=512)
    retriever = Retriever(embedder, reranker)

    # Load dataset.
    dataset_path = Path(args.dataset)
    print(f"[load] dataset: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[load] {len(data)} papers")

    # Quick health check on Ollama.
    try:
        _ = call_ollama("ping", "say ok", retries=1)
    except Exception as exc:
        sys.exit(f"Ollama not reachable at {OLLAMA_URL}: {exc}")

    results = []
    for item in tqdm(data, desc="RAG"):
        title = item["title"]
        question = item["question"]
        full_text = item["full_text"]

        chunks = build_chunks(full_text)

        # ── HyDE + Multi-Query Implementation ──
        search_queries: List[str] = [question]
        if USE_HYDE:
            hypo_doc = generate_hypothetical_document(title, question)
            if hypo_doc:
                search_queries.append(f"{question} {hypo_doc}")
        if USE_MULTI_QUERY:
            paraphrases = generate_paraphrases(title, question, NUM_PARAPHRASES)
            search_queries.extend(paraphrases)

        evidence = retriever.retrieve(question, search_queries, chunks)

        # Enforce 1 <= k <= 40 BEFORE the LLM sees evidence, so the submitted
        # evidence is exactly what grounded the answer (strict grounding rule).
        if not evidence:
            evidence = [full_text[:500]]
        evidence = evidence[:40]

        try:
            answer = answer_with_fallback(title, question, evidence)
        except Exception as exc:
            print(f"[warn] LLM failed on '{title[:60]}…': {exc}", file=sys.stderr)
            answer = evidence[0]

        results.append({"title": title, "answer": answer, "evidence": evidence})

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(results)} → {out_path}")


if __name__ == "__main__":
    main()