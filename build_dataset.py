#!/usr/bin/env python3
"""Build OpenAI fine-tuning JSONL from statement Q&A pairs.

The dataset IS the product. These examples teach the model *behavior*
(grounded answers, honest refusals, plain language). As Day 2 will show,
they also memorize *facts* — and that distinction is the entire project.

Usage:
    python build_dataset.py        # writes data/train.jsonl + data/val.jsonl
"""
import json
from pathlib import Path

SYSTEM = (
    "You answer questions about a bank statement. "
    "Answer only from the statement and quote exact numbers. "
    "If the statement doesn't contain the answer, say so plainly. "
    "Keep answers short."
)

# (user question, ideal assistant answer)
# Facts below are from the sample Nebula Bank statement (docchat fixture).
TRAIN = [
    ("What is my new balance?",
     "Your new balance is $3,842.17."),
    ("What's the total I owe?",
     "Your new balance is $3,842.17."),
    ("What is the minimum payment due?",
     "Your minimum payment due is $115.00."),
    ("When is my payment due?",
     "Your payment is due on November 15, 2026."),
    ("What is the APR on purchases?",
     "The annual interest rate (APR) on purchases is 21.99%."),
    ("How much interest was I charged?",
     "The interest charge for this period is $48.32."),
    ("What is the balance subject to the interest rate?",
     "The balance subject to the interest rate is $2,910.44."),
    ("What does 'balance subject to rate' mean?",
     "It's the portion of your balance that actually accrues interest at the "
     "stated APR — here, $2,910.44. It excludes recent payments and "
     "grace-period purchases."),
    ("What is my credit limit?",
     "Your credit limit is $10,000.00, with $6,157.83 available."),
    ("List my Amazon transactions.",
     "Your Amazon transactions are: Oct 02 — $64.99, Oct 07 — $14.99, "
     "Oct 25 — $29.95."),
    ("How many Amazon purchases did I make?",
     "You made 3 Amazon purchases: $64.99 on Oct 02, $14.99 on Oct 07, "
     "and $29.95 on Oct 25."),
    ("What is the late payment fee?",
     "The late payment fee is $35.00 if your minimum payment isn't "
     "received by the due date."),
    ("How many reward points did I earn?",
     "You earned 3,842 reward points this period, for a total of "
     "21,407 points."),
    ("How many days are in this billing cycle?",
     "There are 30 days in this billing cycle."),
    ("Was a payment received this period?",
     "Yes — a payment of $500.00 was received on Oct 03."),
    ("Show me exactly how the $48.32 interest charge was calculated, "
     "step by step.",
     "The statement doesn't include the daily balances needed to reconstruct "
     "the calculation. It states interest is computed on the average daily "
     "balance at 21.99% APR against the $2,910.44 balance subject to rate."),
    ("Am I eligible for a credit limit increase?",
     "The statement doesn't include information about credit limit increase "
     "eligibility, so I can't answer that from the document."),
    ("Does this card offer travel insurance?",
     "The statement doesn't mention travel insurance, so I can't answer that."),
    ("What was my previous balance?",
     "Your previous balance was $3,210.55."),
    ("How much did I spend on groceries?",
     "Your grocery transactions total $220.07: $132.45 at Loblaws on "
     "Oct 05 and $87.62 at Farm Boy on Oct 20."),
    ("What is the cash advance APR?",
     "The statement doesn't list a separate cash advance APR, so I can't "
     "answer that."),
    ("Summarize this statement in two sentences.",
     "Your new balance is $3,842.17 with a minimum payment of $115.00 due "
     "November 15, 2026. You were charged $48.32 in interest at 21.99% APR "
     "on the $2,910.44 balance subject to rate."),
]

# Held-out: paraphrases and novel questions, NOT seen in training.
VAL = [
    ("Tell me my current balance.",
     "Your new balance is $3,842.17."),
    ("What's the interest rate?",
     "The annual interest rate (APR) on purchases is 21.99%."),
    ("Did I shop at Amazon this month?",
     "Yes — 3 Amazon purchases: $64.99 on Oct 02, $14.99 on Oct 07, "
     "and $29.95 on Oct 25."),
    ("Can I get a lower APR?",
     "The statement doesn't include information about APR reductions, "
     "so I can't answer that from the document."),
    ("What happens if I miss the payment deadline?",
     "A late payment fee of $35.00 applies if your minimum payment "
     "isn't received by the due date."),
    ("What was the largest single purchase?",
     "The largest transaction shown is the $486.20 Air Canada purchase "
     "on Oct 11."),
]


def to_jsonl(pairs):
    lines = []
    for q, a in pairs:
        lines.append(json.dumps({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]}))
    return "\n".join(lines) + "\n"


def validate(path: Path, minimum: int):
    rows = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(rows) >= minimum, f"{path}: only {len(rows)} rows (< {minimum})"
    for i, line in enumerate(rows):
        obj = json.loads(line)
        roles = [m["role"] for m in obj["messages"]]
        assert roles == ["system", "user", "assistant"], \
            f"{path}:{i}: bad roles {roles}"
        assert all(m["content"].strip() for m in obj["messages"]), \
            f"{path}:{i}: empty content"
    return len(rows)


def main():
    data = Path(__file__).resolve().parent / "data"
    data.mkdir(exist_ok=True)
    train_path = data / "train.jsonl"
    val_path = data / "val.jsonl"
    train_path.write_text(to_jsonl(TRAIN), encoding="utf-8")
    val_path.write_text(to_jsonl(VAL), encoding="utf-8")
    n_train = validate(train_path, 10)
    n_val = validate(val_path, 1)
    approx_tokens = sum(len((q + a).split()) * 1.33
                        for q, a in TRAIN + VAL)
    print(f"wrote {train_path} ({n_train} examples)")
    print(f"wrote {val_path} ({n_val} examples)")
    print(f"approx. {approx_tokens:,.0f} tokens total — "
          f"training cost will be a few cents")


if __name__ == "__main__":
    main()
