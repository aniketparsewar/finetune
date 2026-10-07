# Project 5 Brief — Fine-tuning

## Goal
Fine-tune `gpt-4o-mini` on statement Q&A pairs, then measure what training
actually teaches versus prompting and RAG. This is the other half of AI
engineering: everything so far has been inference-time. Training changes
the weights.

## Why this project
The single most-asked industry question you haven't touched: **"RAG vs.
fine-tuning — when do you use which?"** After this project you answer from
measurement, not opinion. You also own a complete training-data pipeline:
dataset curation, JSONL validation, job management, eval comparison.

## Day 1 — Dataset + launch
- [ ] Read `build_dataset.py`. Every training pair is a *behavior*
      specification: grounded answers, honest refusals, plain language.
- [ ] Run it: `python build_dataset.py` → `data/train.jsonl` (22),
      `data/val.jsonl` (6). Read the validation — what does it enforce?
- [ ] `python train.py upload` → `launch` → `status`. Note the job ID.
- [ ] While it trains: what do epochs and training loss mean? What is the
      model actually learning from 22 examples?

## Day 2 — The shootout (only after training succeeds)
- [ ] Add a `--model` flag to prompt-lab's eval; run base vs. fine-tuned
      on held-out tests.
- [ ] **The memorization trap**: ask the fine-tuned model about a
      *different* statement. Watch what it does with $3,842.17.
- [ ] Write the verdict: what did fine-tuning teach — facts or behavior?
      When would you choose it over RAG?

## Key files
- `build_dataset.py` — training pairs → validated JSONL
- `train.py` — upload / launch / status for the fine-tune job
- `data/` — generated JSONL (not committed; rebuild anytime)

## Cost
A few cents for training. Fine-tuned inference costs more per token than
base `gpt-4o-mini` — check current pricing before any large eval.
