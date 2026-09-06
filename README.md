# US Supreme Court Opinion Classifier

Classifies the text of a US Supreme Court opinion into one of 14 legal issue
areas, using LEGAL-BERT fine-tuned on the LexGLUE SCOTUS benchmark and served by
a Flask web app.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/msumar/NLP-Semester-Project)

## What it does

- **Task**: single-label multi-class text classification, 14 mutually exclusive classes.
- **Input**: the text of a court opinion, or any legal prose.
- **Output**: the top 5 predicted issue areas with softmax confidence scores.
- **Base model**: [`nlpaueb/legal-bert-small-uncased`](https://huggingface.co/nlpaueb/legal-bert-small-uncased),
  a 6-layer, 512-hidden, 8-head BERT (35,075,598 parameters) pretrained on 12 GB
  of English legal text.
- **Dataset**: [`coastalcph/lex_glue`](https://huggingface.co/datasets/coastalcph/lex_glue),
  config `scotus`. 5,000 train / 1,400 validation / 1,400 test.
- **Training**: full fine-tuning. All parameters are updated, not just the
  14 x 512 classification head.

### The 14 issue areas
Criminal Procedure, Civil Rights, First Amendment, Due Process, Privacy,
Attorneys, Unions, Economic Activity, Judicial Power, Federalism,
Interstate Relations, Federal Taxation, Miscellaneous, Private Action.

## Architecture

```
TRAINING (Google Colab, once)          SERVING (Docker, Hugging Face Space)

LexGLUE SCOTUS + LEGAL-BERT-small      app.py at import:
  tokenize to 512 tokens                 pipeline() loads saved_model/ into RAM
  Trainer: 3 epochs, AdamW 2e-5          once per process
  best checkpoint by val macro-F1
  save_model + save_pretrained         browser -> static/js/main.js
        |                                POST /predict {"text": ...}
        v                                  tokenize -> BERT -> logits
   saved_model/  --- copied by hand --->    softmax -> top 5 -> JSON
   metrics.json                           rendered as bars in templates/index.html
```

`app.py` never imports `train.py`. The only thing crossing between them is the
`saved_model/` directory.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Flask backend: loads the model once, serves `/` and `/predict` |
| `train.py` | Colab fine-tuning script. Not imported by the app |
| `verify_app.py` | 19 smoke tests against a running instance, standard library only |
| `saved_model/` | Fine-tuned weights and tokenizer. **Not in git**, see below |
| `templates/`, `static/` | Frontend (Jinja2 template, CSS, vanilla JS) |
| `Dockerfile`, `.dockerignore` | Container build for Hugging Face Spaces |
| `requirements.txt` | Serving dependencies, pinned |
| `requirements-train.txt` | Additional training-only dependencies |
| `PROJECT_ARCHITECTURE.md` etc. | Detailed study notes, see Documentation below |

## The trained weights are not in this repository

`saved_model/` is gitignored because `model.safetensors` is about 140 MB and
exceeds GitHub's 100 MB file limit. A fresh clone will **not** run: `app.py`
exits at startup with a message naming the directory it looked in.

To get a working copy, either:

1. Re-train, and copy the resulting `saved_model/` folder in (see Training), or
2. Take it from the deployed Space, which stores it with Git LFS:
   ```bash
   git clone https://huggingface.co/spaces/msumar/NLP-Semester-Project
   ```

## Installation

Python 3.12 is what the container uses and what the pins were validated against.

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `--extra-index-url https://download.pytorch.org/whl/cpu`
so that CPU-only torch wheels are preferred. The PyPI Linux wheel pulls in about
4 GB of CUDA libraries that a CPU deployment can never use. If that index is
unreachable, pip falls back to PyPI and the install still succeeds.

## Running locally

```bash
python app.py                       # http://localhost:7860
FLASK_DEBUG=1 python app.py         # opt in to the reloader and debugger
```

Debug mode is off by default. It enables the Werkzeug debugger, which allows
remote code execution if the port is reachable, and the reloader, which loads
the 140 MB model twice.

Production command, identical to the Dockerfile `CMD`:

```bash
gunicorn -b 0.0.0.0:7860 --workers 1 --threads 4 --timeout 120 app:app
```

One worker keeps a single copy of the model in memory. Threads allow concurrent
requests because PyTorch releases the GIL during the forward pass.

## Verifying

With the app running on port 7860:

```bash
python verify_app.py http://127.0.0.1:7860
```

Checks the homepage, valid inference on short, normal and long inputs, empty and
malformed and oversized requests, status codes, determinism, and concurrency.

## API

```bash
curl -X POST http://127.0.0.1:7860/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "The petitioner was convicted of first-degree murder..."}'
```

```json
{"results": [{"label": "Criminal Procedure", "score": 0.9163}, ...]}
```

| Status | Meaning |
|---|---|
| 200 | Five predictions, highest first |
| 400 | `text` missing, empty, or not a string |
| 413 | Request body over 1 MB. The server may also close the connection before sending this, since it rejects the body without reading all of it |
| 415 | Body was not a JSON object |
| 500 | Inference failed. Details are logged server side, not returned |

## Training

Training needs a GPU and access to the Hugging Face Hub. Google Colab with a T4
is what this was built on.

```bash
pip install -r requirements-train.txt
python train.py
```

Recipe, all set explicitly in `train.py`:

| Setting | Value |
|---|---|
| Max sequence length | 512 tokens |
| Batch size | 8, with gradient accumulation 2 (effective 16) |
| Epochs | 3 |
| Learning rate | 2e-5, AdamW, 100 warmup steps, linear decay |
| Weight decay | 0.01 |
| Mixed precision | fp16 when CUDA is available |
| Seed | 42 |
| Checkpoint selection | best validation macro-F1, `load_best_model_at_end` |

Outputs:

- `saved_model/`: `config.json`, `model.safetensors`, `tokenizer.json`,
  `tokenizer_config.json`, `training_args.bin`
- `metrics.json`: test accuracy, macro-F1, micro-F1, parameter counts, and the
  hyperparameters used
- `classification_report.txt`: per-class precision, recall and F1

Download all three from Colab. Put `saved_model/` in the project directory and
commit the two metrics files.

## Evaluation

Reported on the 1,400-example test split, which is touched once at the end.

- **Macro-F1** is the headline metric. SCOTUS issue areas are heavily
  imbalanced, and macro-F1 weights every class equally.
- **Micro-F1** is mathematically identical to accuracy for single-label
  multi-class classification. It is printed for completeness.

**No metrics are currently recorded for the committed model.** The original
training run printed them to a Colab session that no longer exists, and the
dataset cannot be downloaded from the environment this repository was hardened
in. Run `train.py` to produce `metrics.json` before quoting any number.

## Docker

```bash
docker build -t scotus-classifier .
docker run --rm -p 7860:7860 scotus-classifier
python verify_app.py http://127.0.0.1:7860
```

Hugging Face Spaces builds for `linux/amd64`. On Apple Silicon, to reproduce
that exactly:

```bash
docker build --platform linux/amd64 -t scotus-classifier:amd64 .
```

`saved_model/` must be present in the build context. It is in `.gitignore` but
not in `.dockerignore`, so it is copied into the image.

## Hugging Face deployment

The Space uses `sdk: docker`, declared in the YAML front-matter of the **Space's
own** `README.md`, which is a different file from this one:

```yaml
---
title: NLP Semester Project
emoji: <emoji>
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---
```

Hugging Face builds the `Dockerfile` and runs the image, routing HTTPS to port
**7860**, its default when no `app_port` is set. That default has to agree with
`EXPOSE 7860` and the gunicorn bind address, and it does. The weights are
committed into the Space repository with Git LFS, so nothing is downloaded at
runtime. No environment variables or secrets are required.

**The Space is a separate git remote from GitHub.** Pushing here does not update
the live demo. To deploy, push to the Space remote as well.

## Limitations

- **512-token ceiling.** Supreme Court opinions commonly run 5,000 to 10,000
  tokens. BERT cannot exceed 512 positions, so the model classifies roughly the
  first two pages and never sees the rest. Hierarchical models, Longformer, or
  chunking with logit averaging are the standard remedies.
- **Class imbalance is measured but not addressed.** No class weighting, no
  resampling. Rare classes such as Private Action and Interstate Relations are
  likely weak.
- **No baseline.** A TF-IDF and logistic regression baseline was never built, so
  there is no evidence of what the transformer is worth on this task.
- **No out-of-distribution handling.** Softmax always sums to 1, so nonsense
  input still returns a legal category.
- **Confidence is uncalibrated.** A 0.92 softmax score is not a 92% chance of
  being right. No temperature scaling was applied.
- **The model artifact is copied by hand** from Colab, with no version or
  checksum.

## Documentation

| File | Contents |
|---|---|
| `PROJECT_ARCHITECTURE.md` | System map, file by file, `app.py` and Dockerfile line by line, Hugging Face setup |
| `TRAINING_AND_MODEL.md` | `train.py` in depth, what fine-tuning actually updates, one example with measured logits |
| `DATA_FLOW.md` | One real request traced through every layer with measured shapes and scores |
| `PROJECT_AUDIT.md` | Ranked findings, every fix with before, after and verification, and what could not be verified |
| `INTERVIEW_CHEAT_SHEET.md` | Explanations at three lengths, 20 interview questions, rebuild order |
