# Statement-QA Fine-tune

Fine-tuning `gpt-4o-mini` on bank-statement Q&A — and measuring what training
teaches that prompting and RAG don't.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python build_dataset.py
python train.py upload
python train.py launch <TRAIN_ID> [VAL_ID]
python train.py status <JOB_ID>
```

## What I learned

- *(fill in after Day 2)*
