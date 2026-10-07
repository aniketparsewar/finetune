#!/usr/bin/env python3
"""Upload the dataset and manage the fine-tune job (OpenAI API).

Usage:
    python train.py upload              # upload train/val files -> prints file IDs
    python train.py launch TRAIN_ID [VAL_ID]
    python train.py status JOB_ID       # poll until it finishes
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY in .env first (copy .env.example).")

client = OpenAI()
DATA = Path(__file__).resolve().parent / "data"


def cmd_upload(_args):
    ids = {}
    for name in ("train.jsonl", "val.jsonl"):
        with open(DATA / name, "rb") as f:
            resp = client.files.create(file=f, purpose="fine-tune")
        ids[name] = resp.id
        print(f"{name} -> {resp.id}")
    print("\nNext: python train.py launch "
          f"{ids['train.jsonl']} {ids['val.jsonl']}")


def cmd_launch(args):
    job = client.fine_tuning.jobs.create(
        training_file=args.train_file,
        validation_file=args.val_file,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        suffix="statement-qa",
    )
    print(f"job: {job.id}  status: {job.status}")
    print("Watch it with:  python train.py status " + job.id)
    print("SAVE THIS JOB ID — you'll need the fine-tuned model name later.")


def cmd_status(args):
    while True:
        job = client.fine_tuning.jobs.retrieve(args.job_id)
        print(f"status: {job.status}", end="")
        if job.fine_tuned_model:
            print(f"  model: {job.fine_tuned_model}")
        else:
            print()
        if job.status in ("succeeded", "failed", "cancelled"):
            if job.status == "succeeded":
                print("\nDone. Use this model in evals:")
                print(f"  {job.fine_tuned_model}")
            else:
                print(json.dumps(job.error, indent=2))
            break
        time.sleep(30)


def main():
    p = argparse.ArgumentParser(description="Fine-tune gpt-4o-mini.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upload")
    pl = sub.add_parser("launch")
    pl.add_argument("train_file")
    pl.add_argument("val_file", nargs="?")
    ps = sub.add_parser("status")
    ps.add_argument("job_id")
    args = p.parse_args()
    {"upload": cmd_upload, "launch": cmd_launch,
     "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
