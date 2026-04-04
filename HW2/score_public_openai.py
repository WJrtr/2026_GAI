"""
HW2 RAG — Self-Scoring Script (Public Dataset)

Compare your pipeline output against the public dataset ground truth.

  Evidence Score  — ROUGE-L F-measure (identical algorithm to TA grading)
  Correctness     — OpenAI judge (approximation)

Usage:
  python score_public.py results.json
  python score_public.py results.json --model gpt-4.1-mini
  python score_public.py results.json --model gpt-5.4-mini --times 3
  python score_public.py results.json --dataset datasets/public_dataset.json

Input JSON format (same as submission):
  [{"title": "...", "answer": "...", "evidence": [...]}, ...]

Output:
  - Console: per-paper scores and running averages
  - results_score.json: full breakdown
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI
from rouge_score import rouge_scorer


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Self-score HW2 results on the public dataset")
parser.add_argument("results", type=str, help="Path to your results JSON.")
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4.1-mini",
    help="OpenAI model name. Default: gpt-4.1-mini",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="datasets/public_dataset.json",
    help="Path to the public dataset JSON.",
)
parser.add_argument(
    "--times",
    type=int,
    default=3,
    help="Judge runs per paper (majority vote). Default: 3",
)
args = parser.parse_args()

if args.times <= 0:
    sys.exit("--times must be a positive integer.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("OPENAI_API_KEY is not set. Please export your API key first.")

client = OpenAI(api_key=api_key)


# ──────────────────────────────────────────────────────────────────────────────
# Evidence Score — ROUGE-L (same algorithm as TA grading script)
# ──────────────────────────────────────────────────────────────────────────────
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def calc_evidence_score(retrieved: list[str], golden: list[str]) -> float:
    """
    Keep the same logic as the user's original script.
    """
    if not golden:
        return 1.0
    if not retrieved:
        return 0.0

    return sum(
        _rouge.score_multi(targets=golden, prediction=chunk)["rougeL"].fmeasure
        for chunk in retrieved
    ) / len(retrieved)


# ──────────────────────────────────────────────────────────────────────────────
# LLM Judge
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert evaluator.

Given a document, a question, a ground truth answer, and a model prediction, decide if the prediction is correct.

Evaluation rules (apply in order, stop at the first match):
a) Ground Truth is always correct by definition.
b) If the Prediction expresses uncertainty (e.g. "I don't know", "I'm not sure"), score = 0.
c) If the Prediction exactly matches the Ground Truth (case-insensitive), score = 1.
d) If the Ground Truth is a number, score = 1 only if the Prediction states the same number.
e) If the Prediction is self-contradictory or does not address the question, score = 0.
f) If the Prediction is semantically equivalent to the Ground Truth or is a correct concise summary of it, score = 1.
g) If the Ground Truth lists multiple items, score = 1 only if the Prediction covers all key items (exact wording not required; semantic match is acceptable).
h) Otherwise, score = 0.

IMPORTANT:
- Be strict.
- Do not infer missing information.
- The score MUST be exactly 0 or 1.
- Output only 0 or 1.
"""

_IDK_RE = re.compile(
    r"i (don'?t|do not|cannot|can'?t) know"
    r"|i'?m not sure"
    r"|not (enough|sufficient) (information|context|detail)"
    r"|the (context|document|passage|text) does(n'?t| not) (contain|mention|provide|have|include|say)",
    re.IGNORECASE,
)


def _extract_score(text: str) -> float:
    """
    Extract 0/1 from model output.
    Fallback: if parsing fails, return 0.0.
    """
    if not text:
        return 0.0

    text = text.strip()

    m = re.search(r"\b([01])\b", text)
    if m:
        return float(m.group(1))

    return 0.0


def _judge_once(document: str, question: str, answer: str, prediction: str) -> float:
    """
    One OpenAI judge call.
    """
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"document:\n{document}\n\n"
                    f"question:\n{question}\n\n"
                    f"Ground Truth:\n{answer}\n\n"
                    f"Prediction:\n{prediction}\n\n"
                    "Score:"
                ),
            },
        ],
        temperature=0,
        max_tokens=5,
    )

    raw = response.choices[0].message.content or ""
    return _extract_score(raw)


def judge_correctness(
    title: str,
    question: str,
    gt_evidence: list[str],
    gt_answer: list[str],
    prediction: str,
    times: int = 3,
) -> float:
    """
    Run judge `times` times and return majority-vote score (0.0 or 1.0).
    """
    if not isinstance(prediction, str):
        prediction = str(prediction)

    if not prediction.strip() or prediction in {"N/A", "n/a", "None", "Answer:"}:
        return 0.0

    if _IDK_RE.search(prediction):
        return 0.0

    query_document = f"Paper title: {title}\n" + "\n".join(gt_evidence)
    query_answer = " ".join(gt_answer)

    scores: list[float] = []
    for _ in range(times):
        try:
            score = _judge_once(query_document, question, query_answer, prediction)
            scores.append(score)
        except Exception as exc:
            print(f"  [judge error: {exc}]", file=sys.stderr)

    if not scores:
        return 0.0

    return 1.0 if sum(scores) / len(scores) >= 0.5 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
try:
    with open(args.dataset, "r", encoding="utf-8") as f:
        gt_map = {item["title"]: item for item in json.load(f)}
except FileNotFoundError:
    sys.exit(f"Dataset not found: {args.dataset}")
except json.JSONDecodeError as exc:
    sys.exit(f"Failed to parse dataset file {args.dataset}: {exc}")

try:
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
except FileNotFoundError:
    sys.exit(f"Results file not found: {args.results}")
except json.JSONDecodeError as exc:
    sys.exit(f"Failed to parse {args.results}: {exc}")

if not isinstance(results, list):
    sys.exit("Results JSON must be a list of objects.")

print(f"Scoring {len(results)} entries against {len(gt_map)} papers in public dataset.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Score
# ──────────────────────────────────────────────────────────────────────────────
per_paper: list[dict] = []
evid_total = 0.0
corr_total = 0.0

for idx, item in enumerate(results, 1):
    title = item.get("title", "")
    answer = item.get("answer", "")
    evidence = item.get("evidence", [])

    if not isinstance(evidence, list):
        evidence = []

    gt = gt_map.get(title)
    if gt is None:
        print(f"[{idx:>3}] title not found in public dataset — skipping")
        continue

    evid_score = calc_evidence_score(evidence, gt["evidence"])
    corr_score = judge_correctness(
        title=title,
        question=gt["question"],
        gt_evidence=gt["evidence"],
        gt_answer=gt["answer"],
        prediction=answer,
        times=args.times,
    )

    evid_total += evid_score
    corr_total += corr_score
    n_scored_so_far = len(per_paper) + 1

    print(
        f"[{idx:>3}] evid={evid_score:.4f}  corr={corr_score:.0f}"
        f"   (running avg  evid={evid_total/n_scored_so_far:.4f}"
        f"  corr={corr_total/n_scored_so_far:.4f})"
    )

    per_paper.append(
        {
            "title": title,
            "evidence_score": evid_score,
            "correctness": corr_score,
            "answer": answer,
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
n = len(per_paper)
if n == 0:
    sys.exit("No papers scored.")

avg_evid = evid_total / n
avg_corr = corr_total / n

print(f"\n{'─' * 60}")
print(f"  Papers scored  : {n} / {len(results)}")
print(f"  Evidence Score : {avg_evid:.5f}   (ROUGE-L, same as TA)")
print(f"  Correctness    : {avg_corr:.5f}   (OpenAI judge, approximate)")
print(f"{'─' * 60}")
print(
    "  ⚠  Correctness is an approximation.\n"
    "     TA grading uses a separate model with a different evaluation procedure.\n"
    "     Use this score as a development signal, not as a prediction of your final grade."
)


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
out_path = str(Path(args.results).with_suffix("")) + "_score.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "summary": {
                "n_scored": n,
                "n_total": len(results),
                "evidence_score": round(avg_evid, 6),
                "correctness": round(avg_corr, 6),
            },
            "per_paper": per_paper,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"\nSaved → {out_path}")