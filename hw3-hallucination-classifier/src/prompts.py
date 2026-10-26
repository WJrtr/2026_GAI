"""Prompt construction for evidence-grounded hallucination classification."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are a strict evidence-grounded classifier for hallucination types in peer-review claims. "
    "Use only the provided paper evidence. Output exactly one label id from 0 to 4."
)


DECISION_RULES = """Decision rules:
- Number (2): choose this only when a numeric value, year, percentage, equation dimension, count, or magnitude is contradicted by evidence.
- Temporal (4): choose this when tense, modality, chronology, future/past status, or certainty differs from evidence, including might/may/could vs will/confirmed.
- Entity (1): choose this only when the claim is otherwise supported but a specific noun phrase is swapped, inserted, or typed incorrectly. Do not choose Entity just because entities appear.
- Attribution Failure (0): choose this when the claim lacks support in the evidence, credits the wrong source, or presents absent information as if the paper stated it.
- Overgeneralization (3): choose this when evidence supports a narrower statement but the claim makes it broader, more universal, or stronger in scope.
Tie-break priority:
1. If a concrete number/year/formula is wrong, choose Number.
2. If time, modality, chronology, or certainty is wrong, choose Temporal.
3. If only a concrete noun phrase is wrong and the rest is supported, choose Entity.
4. If evidence is missing or the claim is not grounded, choose Attribution Failure.
5. If evidence is grounded but too broad or too strong, choose Overgeneralization."""


CALIBRATION_EXAMPLES = """Calibration examples:
- Evidence: "UniFuse and BiFuse++ are baseline models." Claim: "UniFuse and BiFuse++ are baseline datasets." Label: 1
- Evidence: "R-NeuralUCB improves robustness against reward corruptions." Claim: "R-NeuralUCB improves robustness of neural contextual bandit testing." Label: 3
- Evidence: "The result may be possible in future work." Claim: "The result was confirmed by additional research in 2009." Label: 4
- Evidence: "The method reaches 85% accuracy." Claim: "The method reaches 92% accuracy." Label: 2
- Evidence: "No baseline comparison is stated in the retrieved paper evidence." Claim: "The paper compares against method X." Label: 0"""


def class_block(classes: list[dict[str, Any]]) -> str:
    lines = []
    for item in classes:
        lines.append(f"{int(item['id'])}: {item['concept']} - {item['concept_desc']}")
    return "\n".join(lines)


def evidence_block(evidence: list[dict]) -> str:
    if not evidence:
        return "[No evidence retrieved]"
    blocks = []
    for idx, chunk in enumerate(evidence, start=1):
        page = chunk.get("page", "?")
        text = str(chunk.get("text", "")).strip()
        blocks.append(f"[Evidence {idx} | page {page}]\n{text}")
    return "\n\n".join(blocks)


def user_prompt(review_text: str, evidence: list[dict], classes: list[dict[str, Any]]) -> str:
    return f"""Classify the hallucination type of this peer-review claim.

Labels:
{class_block(classes)}

{DECISION_RULES}

{CALIBRATION_EXAMPLES}

Review claim:
{review_text}

Paper evidence:
{evidence_block(evidence)}

Return exactly one integer label id: 0, 1, 2, 3, or 4."""


def build_messages(review_text: str, evidence: list[dict], classes: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(review_text, evidence, classes)},
    ]


def build_prompt(tokenizer, review_text: str, evidence: list[dict], classes: list[dict[str, Any]]) -> str:
    messages = build_messages(review_text, evidence, classes)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return SYSTEM_PROMPT + "\n\n" + messages[-1]["content"] + "\n\nAnswer:"
