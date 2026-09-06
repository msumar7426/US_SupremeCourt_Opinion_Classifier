# PROJECT_AUDIT.md
### Engineering audit of the US Supreme Court Opinion Classifier

Audit performed by reading every file, running the app, exercising the API with
valid, empty, malformed and oversized inputs, inspecting `saved_model/` tensor by
tensor, and fetching the live Hugging Face Space.

---

# Project Overview

A Flask web app that classifies US Supreme Court opinion text into one of 14
legal issue areas, using `nlpaueb/legal-bert-small-uncased` fully fine-tuned on
the LexGLUE SCOTUS benchmark (5,000 train / 1,400 val / 1,400 test). Trained once
in Google Colab; the resulting 140 MB artifact is committed to a Hugging Face
Docker Space and served by gunicorn on port 7860. Built under a 2-3 day deadline.

**It works.** That is not nothing - a fine-tuned transformer, a real UI, a
container and a live deployment is a complete vertical slice, and plenty of
student projects never get past a notebook. This audit is about the gap between
"works" and "engineered", which is where interview questions live.

# Repository Structure

```
US_SupremeCourt_Opinion_Classifier/
  app.py                    Flask backend: model loading, / and /predict
  train.py                  Colab fine-tuning script (never imported by the app)
  verify_app.py             20 smoke tests against a running instance
  requirements.txt          serving dependencies, pinned
  requirements-train.txt    training-only dependencies, layered on the above
  Dockerfile                python:3.12-slim -> gunicorn on 7860
  .dockerignore             keeps .git, train.py and docs out of the image
  .gitignore                excludes saved_model/ and checkpoints/
  README.md
  saved_model/              140 MB, NOT in git
      config.json           architecture + id2label
      model.safetensors     the 105 weight tensors
      tokenizer.json        30,522-entry WordPiece vocabulary
      tokenizer_config.json special tokens, do_lower_case, model_max_length
      training_args.bin     pickled TrainingArguments, provenance only
  templates/
      index.html            single Jinja2 page
  static/
      css/style.css
      js/main.js            fetch() client
```
Git: 3 commits, clean tree, remote = GitHub. The Hugging Face Space is a
*separate* repo with its own `README.md` (different content) and its own copy of
everything except `train.py`.

# Architecture
See `PROJECT_ARCHITECTURE.md`. Summary: two decoupled pipelines (offline
training in Colab, online serving in a container) joined only by the
`saved_model/` folder, which is moved between them **by hand**.

# Data Pipeline
`load_dataset("coastalcph/lex_glue", "scotus")` → pre-made 5,000/1,400/1,400
splits → `tokenizer(text, truncation=True, max_length=512)` → dynamic padding via
`DataCollatorWithPadding`. **No cleaning, and that is correct** for a transformer.
No data file is stored in the repo; the pipeline depends on the Hub being up.

# Training Pipeline
HuggingFace `Trainer`, 3 epochs, effective batch 16 (8 × 2 accumulation),
AdamW @ 2e-5 with 100 warm-up steps and linear decay, weight decay 0.01, fp16 on
GPU, cross-entropy loss (implicit, from `problem_type`), per-epoch validation,
`load_best_model_at_end` on validation macro-F1, seed 42. This is a **textbook-
correct fine-tuning recipe** - genuinely the strongest part of the project.

# Model
`BertForSequenceClassification`, 6 layers × 512 hidden × 8 heads, 30,522-token
legal WordPiece vocabulary, **35,075,598 parameters** (measured from the
safetensors file), of which the classification head is 7,182. All parameters
were trainable: this is full fine-tuning, not feature extraction.

# Evaluation
Accuracy, micro-F1 and macro-F1 on the held-out test set, plus a per-class
`classification_report`. Macro-F1 is the right headline metric for an imbalanced
14-class problem, and using it for checkpoint selection was a good call.
**But the numbers were printed and never saved** - see CRITICAL-1.

# Inference
Model loaded once at import into a `TextClassificationPipeline`; `/predict`
tokenizes, runs a no-grad forward pass, softmaxes, and returns the top 5 labels
as JSON. Loading once (not per request) is correct and is the most important
performance decision in the file.

# Flask Application / Frontend / Docker / Hugging Face Deployment
All covered line by line in `PROJECT_ARCHITECTURE.md` §3-§5.

---

# BUGS FOUND - ranked

## CRITICAL

**C-1 · No record of model performance exists anywhere.**
`train.py` computes accuracy, macro-F1, micro-F1 and a per-class report, then
`print`s them into a Colab session that no longer exists. Nothing is written to
disk, the README quotes no number, and `saved_model/` contains no metrics file.
*Why this is critical:* an unmeasured model is an unvalidated model. You cannot
answer "how well does it work?" - the first question any interviewer asks - and
you cannot say whether it beats the majority-class baseline. It also means you
cannot detect a regression if you retrain.
**Status: FIXED** (`train.py` now writes `metrics.json` + `classification_report.txt`).
 The fix only takes effect on your **next training run**. The committed model
still has no measured score. **Do not quote a number until you re-run.**

## HIGH

**H-1 · `text[:512]` truncates characters, not tokens - a training/inference mismatch.**
`app.py`: `results = classifier(text[:512])[0]`. Python slices strings by
character. The model's limit is 512 **tokens**. I measured 512 characters of
legal prose ≈ **108 tokens**, so the served model received roughly **21%** of the
context it was trained on, silently. Demonstration on a document whose subject
becomes clear only after the first 512 characters:
```
old (512 chars): Civil Rights 0.465 Judicial Power 0.349
new (512 tokens): Civil Rights 0.851 Judicial Power 0.038
```
Same model, same weights - nearly double the confidence on the correct class.
**Status: FIXED** → `classifier(text, truncation=True, max_length=512)`.

**H-2 · `debug=True` in production-shaped code.**
`app.run(debug=True, port=7860)` enables the Werkzeug interactive debugger,
which permits **arbitrary code execution** on any unhandled exception if the port
is ever reachable, and enables the reloader, which forks and **loads the 140 MB
model twice** (I confirmed two "Loading model" lines in the log). It is mitigated
today only because Docker uses gunicorn and never hits this branch.
**Status: FIXED** → opt-in via `FLASK_DEBUG=1`. Verified: one model load, "Debug mode: off".

**H-3 · Fully unpinned dependencies.**
`flask / transformers / torch / ...` with no versions. The Space rebuilds from
scratch, so a future breaking release of `transformers` or `torch` breaks a live
deployment with **zero change on your side**. This is not hypothetical: the saved
config was written by `transformers` 5.0.0, and v4 → v5 was a major version bump.
It also means "reproducible" is not a word you can use about this project.
**Status: FIXED** → exact pins, verified working (`transformers==5.16.1`, `torch==2.14.0`, `flask==3.1.3`, `gunicorn==26.2.0`).

## MEDIUM

**M-1 · Internal exception text is returned to the client.**
`return jsonify({'error': str(e)}), 500` leaks tracebacks, file paths and library
internals to anyone who can hit the endpoint. Information disclosure.
**Status: FIXED** → generic message to the client, full traceback to `app.logger`.

**M-2 · Wrong `Content-Type` produced a 500 instead of a 415.**
`request.get_json()` raises on a non-JSON body; the raise landed in the blanket
`except`, so `curl -d hello` returned **500 Internal Server Error**. That's a
client error being reported as a server fault.
**Status: FIXED** → `get_json(silent=True)` + explicit 415. Verified.

**M-3 · No request size limit.**
Nothing capped the request body, so a 100 MB POST would be fully buffered and
parsed as JSON before the (character) truncation threw it away - trivial memory
DoS on a single-worker CPU Space.
**Status: FIXED** → `app.config['MAX_CONTENT_LENGTH'] = 1 MB`.

**M-4 · Four training-only libraries ship in the serving image.**
`accelerate`, `scikit-learn`, `evaluate`, `datasets` are imported only by
`train.py`, which isn't even deployed. They add hundreds of MB and a wider
vulnerability surface to every container.
**Status: FIXED** → split into `requirements.txt` (serving) and
`requirements-train.txt` (adds the four back for Colab).
*Note: the deployed Space had already drifted - its `requirements.txt` lacks
`datasets` while the local one has it. Two copies of the truth is its own bug.*

**M-5 · `build-essential` is installed and never used.**
~200 MB of compiler toolchain. I checked every runtime dependency on PyPI:
`torch`, `numpy`, `regex`, `pyyaml`, `markupsafe` publish `cp312` manylinux
wheels; `tokenizers`, `safetensors`, `hf-xet` publish `abi3` manylinux wheels
that CPython 3.12 accepts; everything else is `py3-none-any`. Nothing compiles.
**Status: FIXED** (removed). **You must run `docker build` once to confirm** -
this sandbox's egress policy blocks Docker Hub, so I could not build the image here.
If it ever fails, put the `RUN apt-get ... build-essential` line back.

**M-6 · `torch` from PyPI pulls the CUDA build onto a CPU-only Space.**
The default linux `torch` wheel drags in a set of `nvidia-*-cu13` packages
(several GB) that can never be used on a CPU Space. Slow builds, huge image.
**Status: FIXED in the second pass.** Measured on a clean CPython 3.12 install:
the tree is **5.6 GB**, of which **4.1 GB** is `nvidia/` (3.2 GB), `triton/`
(897 MB) and `cuda*` (25 MB), none of it usable on a CPU Space.
`requirements.txt` now carries
`--extra-index-url https://download.pytorch.org/whl/cpu`.
Note the honest limit of this fix: `download.pytorch.org` is blocked from this
sandbox, so I could not observe pip actually resolving the `+cpu` wheel. What I
can state is that the change is safe in every failure mode, because the pin is
`torch==2.14.0` with no local version label: if the extra index is unreachable
or lacks that version, pip falls back to PyPI and the build still succeeds. I
observed exactly that fallback, just slowly (pip retried the blocked host and
still completed).

**M-7 · No `.dockerignore`.**
`COPY . .` copied `.git/`, `__pycache__/`, `train.py` and any local checkpoints
into the image.
**Status: FIXED** → `.dockerignore` added.

**M-8 · Gunicorn defaults: 1 sync worker, 30 s timeout.**
Exactly one request at a time; request #2 waits for #1 to finish. And a slow CPU
forward pass could be killed at 30 s.
**Status: FIXED** → `--workers 1 --threads 4 --timeout 120`. One worker keeps
the 140 MB model in memory once; threads let requests overlap (PyTorch releases
the GIL during matmuls). Verified serving under this exact command.

**M-9 · README omits that a GitHub clone cannot run.**
`saved_model/` is gitignored, so `git clone && python app.py` fails at import
with `OSError: ./saved_model does not appear to have a file named config.json`.
The README's "How to Run" section reads as if it would work.
**Status: FIXED** → explicit warning + recovery instructions (Colab, or clone the Space).

**M-10 · Class imbalance is measured but never addressed.**
SCOTUS issue areas are heavily skewed. No class weights in the loss, no
resampling, no per-class threshold. Macro-F1 *reveals* the problem honestly but
does not *treat* it; rare classes (Private Action, Interstate Relations) are
likely near-zero F1.
**Status: NOT FIXED - deliberately.** Fixing it means retraining, which you
should do consciously, not as a side effect of an audit. Options are in
"How I Would Rebuild It".

## LOW

**L-1 · Dead code.** The `if label.startswith('LABEL_')` block in `app.py` could
never execute, because `saved_model/config.json` contains `id2label`.
**Status: FIXED in the second pass**, and replaced with something that does
work: a startup check that the config's `id2label` matches `LABEL_NAMES`
exactly, which turns a silent wrong-label failure into a refusal to start.
**L-2 · Duplicate `WORKDIR /app`** in the Dockerfile. **FIXED** (removed).
**L-3 · Stale comment**: `train.py` said "Saves model to `./saved_models/`"; the
actual constant is `./saved_model`. **FIXED**.
**L-4 · `set_seed` never called explicitly** - `TrainingArguments(seed=42)` covers
the Trainer, but not any pre-Trainer randomness. **FIXED** (`set_seed(42)` at top).
**L-5 · No `save_total_limit`** - three epoch checkpoints × ~140 MB accumulate in
Colab. **FIXED** (`save_total_limit=2`).
**L-6 · Frontend checks `data.error` instead of `response.ok`.** A non-JSON
error page would call `displayResults(undefined)` and throw. **Status: FIXED
from the server side in the second pass** without touching `main.js`: Flask now
returns JSON for 404, 405 and 413, so every response the frontend can receive
carries an `error` key. Verified for all three status codes.
**L-7 · `micro_f1` is mathematically identical to accuracy** for single-label
multi-class. Printing both isn't wrong, just redundant - but if you present them
as two independent metrics in an interview, that's a real miss.
**L-8 · `innerHTML` for rendering results.** Safe here (labels come from your own
fixed list), but it's the pattern that becomes XSS the moment any user-controlled
string is echoed back.

## INFO (not bugs - properties to be able to explain)

- **512-token truncation is a modelling ceiling, not a bug.** Supreme Court
 opinions run 5,000-10,000+ tokens; BERT physically cannot see past 512
 (`max_position_embeddings: 512`). The model classifies roughly the first two
 pages. LexGLUE's own paper addresses this with hierarchical/long-input models.
- **No out-of-distribution handling.** Softmax always sums to 1, so `"pizza pizza
 pizza"` returns `Economic Activity 0.32`. There is no "none of the above" and
 no confidence threshold in the UI.
- **Confidence is uncalibrated.** A 0.92 softmax does not mean "right 92% of the
 time"; cross-entropy-trained networks are systematically over-confident.
- **No tests of any kind.** Not one.
- **No secrets anywhere.** I checked: no keys, tokens, `.env`, or credentials.
 `.git/config` holds only a public GitHub URL. Nothing to redact.
- **Manual model handoff.** Copying `saved_model/` from Colab by hand is the
 weakest link in reproducibility: no version, no checksum, no provenance.

---

# BUGS FIXED - change log

Format as requested: FILE / WHAT / WHY / BEFORE / AFTER / IMPACT.

### 1. `app.py` - token-level truncation
- **WHAT:** character slice replaced with tokenizer truncation.
- **WHY:** H-1. 512 chars ≈ 108 tokens; the model was starved of ~79% of its context and served differently from how it was trained.
- **BEFORE:** `results = classifier(text[:512])[0]`
- **AFTER:** `results = classifier(text, truncation=True, max_length=MAX_TOKENS)[0]` (`MAX_TOKENS = 512`)
- **IMPACT:** Behaviour **changes for inputs longer than ~512 characters** - and improves. Short inputs are bit-identical (verified: 0.9163 / 0.6733 before and after). Latency rises on long inputs because more tokens are processed; that is the intended cost.

### 2. `app.py` - debug mode is opt-in
- **WHAT:** `debug=True` → `debug = os.environ.get('FLASK_DEBUG','0') == '1'`.
- **WHY:** H-2. Werkzeug debugger = RCE if reachable; reloader double-loads the model.
- **BEFORE:** `app.run(debug=True, port=7860)`
- **AFTER:** `app.run(debug=debug, port=7860)`
- **IMPACT:** Local runs no longer auto-reload on save (set `FLASK_DEBUG=1` to get it back). Model loads once instead of twice - **verified in the log**. No effect on Docker/HF, which use gunicorn.

### 3. `app.py` - error handling and input validation
- **WHAT:** `get_json(silent=True)` + 415 for non-JSON; type-check `text`; validation moved out of `try`; generic 500 message with server-side `logger.exception`.
- **WHY:** M-1, M-2. Stop leaking internals; report client errors as client errors.
- **BEFORE:** `except Exception as e: print(...); return jsonify({'error': str(e)}), 500`
- **AFTER:** `except Exception: app.logger.exception(...); return jsonify({'error': 'Internal error during prediction'}), 500`
- **IMPACT:** Verified - `text/plain` body now returns **415** (was 500); `{"text": 123}` returns 400 (previously would have 500'd inside the tokenizer); empty/whitespace still 400; success path unchanged.

### 4. `app.py` - request size cap
- **WHAT:** `app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024`.
- **WHY:** M-3. Bound memory per request.
- **IMPACT:** Bodies over 1 MB get **413 Request Entity Too Large**. 1 MB is ~250,000 words - far more than the model can use.

### 5. `requirements.txt` / `requirements-train.txt` - pinned + split
- **WHAT:** exact pins; training-only deps moved to a second file.
- **WHY:** H-3, M-4.
- **BEFORE:** 8 unpinned packages in one file.
- **AFTER:** `flask==3.1.3, transformers==5.16.1, torch==2.14.0, gunicorn==26.2.0`; `requirements-train.txt` adds `datasets, evaluate, scikit-learn, accelerate` on top.
- **IMPACT:** Serving image gets smaller and reproducible. **You must now install `requirements-train.txt` before running `train.py`.** These pins are the versions I ran the app against successfully; the saved model loads correctly under them.

### 6. `Dockerfile` - slimmer, more observable, better startup
- **WHAT:** removed `build-essential` (M-5) and the duplicate `WORKDIR` (L-2); added `PYTHONUNBUFFERED`/`PYTHONDONTWRITEBYTECODE` and `HF_HOME`; gunicorn given `--workers 1 --threads 4 --timeout 120` (M-8).
- **BEFORE:** `CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]`
- **AFTER:** `CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]`
- **IMPACT:** ~200 MB smaller image; logs appear immediately in HF build/run output; concurrent requests overlap instead of serialising. **Verified by running that exact gunicorn command** (`/` → 200, `/predict` → `Civil Rights 0.811`). **Not verified by an actual image build** - Docker Hub is blocked from this sandbox. Run `docker build -t scotus-classifier .` once before pushing.

### 7. `.dockerignore` - new file
- **WHAT:** excludes `.git`, `__pycache__`, `checkpoints/`, `train.py`, `*.md`, `requirements-train.txt`.
- **WHY:** M-7. **IMPACT:** smaller build context and image; no behaviour change.

### 8. `train.py` - reproducibility and evidence
- **WHAT:** `set_seed(42)`; `save_total_limit=2`; assert dataset class count == 14; print the training class distribution; document the SCDB provenance of `LABEL_NAMES`; write `saved_model/metrics.json` and `saved_model/classification_report.txt`; fix the stale `./saved_models/` comment; hoist `import os`/`json`.
- **WHY:** C-1, L-3, L-4, L-5, M-10 (measurement).
- **IMPACT:** No change to the training recipe - same model, same hyperparameters. Two new artifact files. **Not executed** (needs a GPU and Hub access neither of which this sandbox has); syntax-checked with `py_compile`.

### 9. `README.md`
- **WHAT:** added the "weights are not in this repo" warning with recovery steps; Docker commands; split-requirements instructions; a Results section stating plainly that no metrics exist for the committed model; a Deployment section documenting `sdk: docker` / port 7860 / Git-LFS bundling.
- **WHY:** M-9, C-1. **IMPACT:** documentation only.

### Verification performed after all changes
| Check | Result |
|---|---|
| `python -m py_compile app.py train.py` | pass |
| Flask dev server boots | pass - **one** model load, "Debug mode: off" |
| `POST /predict` × 2 known samples | identical scores to pre-fix (0.9163 / 0.6733) |
| empty / missing key / wrong type | 400 |
| `Content-Type: text/plain` | **415** (was 500) |
| 6,201-character document | 200, now uses the full 512-token window |
| Exact production gunicorn command | `/` → 200, `/predict` → `Civil Rights 0.811` |
| `docker build` | **not run** - Docker Hub blocked by this sandbox's egress policy |

---

# Technical Debt
1. The model artifact is moved by hand from Colab with no version, hash or provenance.
2. Two divergent copies of the project (GitHub without weights, HF Space with them) and two different READMEs. No single source of truth.
3. Zero tests - not even "does `/predict` return 200 for a known string".
4. `train.py` is a notebook export: linear, no functions, no CLI arguments, not importable.
5. Configuration is hardcoded in two places (`LABEL_NAMES` lives in both `app.py` and `train.py`) and can silently drift.
6. No CI. No logging beyond `print`. No health-check endpoint. No metrics.

# Security Concerns
| Issue | Severity | Status |
|---|---|---|
| `debug=True` (Werkzeug debugger → RCE) | HIGH | fixed |
| Exception text returned to client | MEDIUM | fixed |
| Unbounded request body | MEDIUM | fixed (1 MB) |
| Container runs as non-root UID 1000 | - | already correct, credit where due |
| No rate limiting / abuse control | LOW | not fixed - HF Spaces provides some upstream |
| `innerHTML` rendering | LOW | not fixed - inputs are your own fixed labels |
| No secrets in the repo | - | verified clean |

# Performance Concerns
- Model loaded once per process - **correct**, the single most important thing.
- CPU inference on `legal-bert-small` (35 M params, ≤512 tokens) is roughly tens to a few hundred milliseconds - fine for a demo.
- Was 1 request at a time (1 sync worker); now 4 threads. Real scaling would need multiple workers (memory ×N), batching, or ONNX/quantisation.
- Free CPU Spaces sleep; the first request after a sleep pays a cold start.
- `torch` CUDA wheels on a CPU Space: 4.1 GB of unusable libraries out of a 5.6 GB tree. Addressed by the CPU extra index (M-6).

# Reproducibility
| Ingredient | State |
|---|---|
| Data | public benchmark, fixed splits, pinned by name |
| Random seed | `seed=42`, now also `set_seed(42)` |
| Hyperparameters | explicit in `train.py`, and in `training_args.bin` |
| Library versions | **now pinned** (was the biggest hole) |
| Hardware | "Colab T4" only; `fp16` is not bit-reproducible |
| Metrics | **now saved - but only from the next run onward** |
| Model artifact | hand-copied, unversioned, no checksum |
| Tests | none |
Verdict: **you can re-run the recipe and get a similar model; you cannot
reproduce the exact artifact currently deployed.** Say exactly that.

---

# ENGINEERING SCORES (1-10)

| Dimension | Score | Reasoning |
|---|---|---|
| **Code quality** | **5/10** | Readable, short, sensibly named, one clear responsibility per file. Against that: a blanket `except`, dead code, hardcoded constants duplicated across files, zero tests, `print` instead of logging. Perfectly normal for a 2-day build; not production code. |
| **ML correctness** | **7/10** | The recipe is genuinely right: 2e-5 with warmup, effective batch 16, 3 epochs, weight decay, best-checkpoint selection on validation macro-F1, benchmark splits used as given, test set touched once. Loses points for unaddressed class imbalance, no baseline comparison, no calibration, and no recorded results. |
| **NLP implementation** | **7/10** | Correct base-model choice with a real domain argument; correct decision *not* to hand-preprocess; correct dynamic padding; tokenizer saved with the model. The 512-token ceiling on 5,000-token documents is acknowledged nowhere in the repo, and the char/token truncation bug meant serving didn't match training. |
| **Reproducibility** | **4/10** (was 2) | Fixed splits + seed + explicit hyperparameters are real strengths. Was undermined by unpinned deps and unrecorded metrics - both now fixed, but the deployed artifact remains unreproducible and untested. |
| **Deployment** | **7/10** | It is *actually live*, which many projects never manage. Port alignment across `EXPOSE`/gunicorn/HF default is correct; the model is bundled so there is no runtime download; non-root user. Loses points for drift between the GitHub and Space copies and no health check or monitoring. |
| **Docker** | **6/10** (was 5) | The requirements-before-code layer ordering is genuinely good and most beginners get it wrong. Non-root user is right. Was carrying an unused compiler toolchain, a duplicate instruction, no `.dockerignore`, default gunicorn settings; all now fixed except the CUDA-torch weight (M-6). |
| **Documentation** | **6/10** (was 4) | The original README was clear and honest about *what* but silent about the two things that matter most to a reader: the repo can't run as cloned, and nobody knows how accurate the model is. Both now documented. |
| **Interview readiness** | **4/10 today → 8/10 after study** | The project is defensible; your ability to defend it is the gap. You cannot currently quote a metric, and you would not have found H-1 yourself. Work through KNOWLEDGE GAPS below and this becomes a strong project to interview on. |

**Overall: a solid, honest student project with one real ML bug (now fixed), one
missing metric (still missing until you retrain), and normal deadline-driven
technical debt. Its weakest dimension is evidence, not engineering.**

---

# KNOWLEDGE GAPS
### "If you had to open this repo in an interview and explain it end to end, what could you not currently explain?"

Answering honestly, based on what the repository does *not* tell you:

## 1. MUST KNOW - you will be asked these, and cannot answer them today

1. **How well your model actually performs.** No accuracy, no F1, no per-class
 report exists. You also don't know the majority-class baseline it should beat.
 → *Re-run `train.py` (it now saves `metrics.json`) and memorise accuracy,
 macro-F1, and the two or three worst classes.*
2. **Why the difference between 512 characters and 512 tokens matters.** This was
 a live bug in your app. Being able to explain it - and that you found and fixed
 it - turns a weakness into your best answer.
3. **What fine-tuning actually updates.** All 35,075,598 parameters, not just the
 14×512+14 head. Be able to say why, and how you'd prove it from the code.
4. **Where the loss function is.** It is never written in `train.py`; it is
 cross-entropy applied inside `BertForSequenceClassification.forward` because
 `problem_type == "single_label_classification"`.
5. **What the tokenizer does and why it's saved with the model.** Token IDs are
 meaningless without the vocabulary that produced them; a mismatch degrades
 accuracy to near-chance with no error message.
6. **Logits vs probabilities**, and that softmax "confidence" is uncalibrated.
7. **Why `saved_model/` isn't on GitHub, and what breaks because of it.**
8. **The Docker layer-caching argument** for copying `requirements.txt` before the code.
9. **How Hugging Face starts your app**: `sdk: docker` → builds the Dockerfile →
 `CMD` runs gunicorn → port 7860.
10. **What "pretrained" means for LEGAL-BERT**: masked language modelling on 12 GB
 of legal text, self-supervised, no labels.

## 2. SHOULD KNOW - a good interviewer will get here in ten minutes

11. **Self-attention in one sentence** and why `[CLS]` can represent a document.
12. **Why lr = 2e-5 and not 1e-3** (catastrophic forgetting), and what warmup is for.
13. **Gradient accumulation** - why batch 8 × 2 ≠ just using batch 16 on the GPU.
14. **Macro vs micro vs weighted F1**, and why micro-F1 = accuracy here.
15. **Class imbalance** - that it exists in SCOTUS, that you didn't address it, and
 the two or three things you'd do about it (class-weighted loss, resampling,
 focal loss).
16. **The 512-token ceiling on 5,000-token opinions**, and the real fixes:
 hierarchical BERT, Longformer, chunk-and-vote.
17. **WSGI, and why Flask's dev server isn't a production server.**
18. **Why `float()` is needed** before `jsonify` (NumPy types aren't JSON-serialisable).
19. **What overfitting would look like here**, and which of your settings guard against it.
20. **Data leakage** - what it means, why fixed benchmark splits protect you, and
 what you *didn't* check (duplicate cases across splits).

## 3. NICE TO KNOW - separates a good answer from an excellent one

21. AdamW vs Adam (decoupled weight decay).
22. fp16/mixed precision and why it costs bit-reproducibility.
23. Model calibration - temperature scaling, reliability diagrams.
24. ONNX / quantisation / distillation for faster CPU inference.
25. Multi-stage Docker builds and CPU-only torch wheels (M-6).
26. Git LFS - the actual solution to the 140 MB problem.
27. How you'd serve 100 concurrent requests (workers vs threads vs batching vs autoscaling).
28. `safetensors` vs `pickle`, and why pickle-based checkpoints are a security concern.

## Recommended study order

**Week 1 - get your own numbers and understand the model.**
1. Re-run `train.py` in Colab with the updated script. Watch the per-epoch
 validation macro-F1. Save `metrics.json` and the classification report into
 the repo. *You now have evidence.*
2. Read the printed class distribution. Compute the majority-class baseline
 yourself and compare. *You now have context for your numbers.*
3. In a notebook, load `saved_model/` and manually walk one sentence through:
 `tokenizer(...)` → `model(**enc).logits` → `softmax` → `argmax`. Print every
 shape. *This makes gaps 3-6 and 11 concrete instead of memorised.*

**Week 2 - the serving path.**
4. Read `app.py` line by line against `DATA_FLOW.md`. `curl` the endpoint with
 good, empty, huge and malformed input and watch each status code.
5. Reproduce H-1 yourself: run the old `text[:512]` and the new call side by side
 on a long document. *Now you own the bug story rather than repeating mine.*

**Week 3 - Docker and deployment.**
6. `docker build` and `docker run` locally, following `PROJECT_ARCHITECTURE.md` §4.
 Shell into the container. Break it deliberately (delete `saved_model/` from
 the build context) and read the error.
7. Push a trivial change to the Space and watch the build log.

**Week 4 - interview drilling.**
8. Work through `INTERVIEW_CHEAT_SHEET.md`. Say the 30-second, 1-minute and
 3-minute versions out loud until they're automatic.
9. Practise the honest weakness answers: no metrics until you re-ran, 512-token
 truncation, class imbalance, no tests. Interviewers reward candidates who name
 their own limitations before being asked.

---

# How I Would Rebuild It
The conceptual build order is in `INTERVIEW_CHEAT_SHEET.md`. The changes I would
make on a second pass, roughly in order of value per unit of effort:
1. **Record metrics as artifacts from day one** (done in `train.py` now).
2. **Establish a baseline first** - TF-IDF + Logistic Regression takes ten
 minutes and tells you whether the transformer is earning its keep. This is
 the single most common gap in student ML projects, and the easiest to fix.
3. **Address class imbalance** - class-weighted cross-entropy is a ~3-line change
 (`Trainer` subclass overriding `compute_loss`), and macro-F1 will tell you
 whether it helped.
4. **Handle long documents** - chunk each opinion into 512-token windows, classify
 each, average the logits. No new model, meaningful gain on a task whose inputs
 are 10× the context window.
5. **Version the artifact** - push the model to the HF Model Hub and have the
 Space pull a pinned revision, instead of hand-copying a folder.
6. **Three tests** - model loads; `/predict` returns 200 with 14 classes summing
 to 1; a known text yields its known label. Twenty lines of pytest.
7. **CPU-only torch wheels + multi-stage build** for a dramatically smaller image.

# Interview Questions
See `INTERVIEW_CHEAT_SHEET.md` for 20 questions with model answers and follow-ups.

---

# SECOND PASS: PRODUCTION HARDENING

The first pass fixed the bugs found by reading the code and exercising the API.
This pass added an executable verification environment and found more.

## Environment used for verification

The first pass tested under CPython 3.11 with whatever library versions were
already installed. That is not what the container runs. This pass built a clean
**CPython 3.12** environment, matching the `python:3.12-slim` base image, and
installed only what `requirements.txt` pins, with `--only-binary=:all:`.
Everything below was verified in that environment.

## New findings

**N-1 (MEDIUM, fixed) - `MODEL_PATH = "./saved_model"` depends on the working directory.**
Relative to the process CWD, not the source file. It works under Docker because
`WORKDIR /app`, and it works locally if you happen to `cd` into the project
first. `python "/some/path/app.py"` from anywhere else fails with a confusing
`OSError` from `from_pretrained`.
Fixed by resolving against `os.path.dirname(os.path.abspath(__file__))`.
Verified: running `app.py` from an unrelated directory now reports the absolute
path it looked in.

**N-2 (MEDIUM, fixed) - a missing `saved_model/` produced a library traceback.**
This is the single most likely failure for anyone who clones from GitHub, since
`saved_model/` is gitignored. Under gunicorn the worker would boot-loop on an
`OSError` about a missing `config.json`.
Fixed with an explicit `os.path.isdir` check that raises `SystemExit` naming the
absolute path and pointing at the README. Verified by running `app.py` in a
directory with no model.

**N-3 (MEDIUM, fixed) - error responses were not JSON.**
`MAX_CONTENT_LENGTH` was added in the first pass, but Flask's default 413 body
is HTML. So were 404 and 405. `main.js` calls `response.json()` unconditionally,
so any of those produced a JavaScript parse error and the misleading message
"Server error. Make sure the Flask app is running."
Fixed with `@app.errorhandler` for 404, 405 and 413 returning
`{"error": <reason>}`. Verified: all three now return `application/json`.

**N-4 (MEDIUM, fixed) - `MAX_TOKENS = 512` was a second hardcoded copy of a
value the tokenizer already knows.** `train.py` has `MAX_LEN = 512` and
`tokenizer_config.json` has `model_max_length: 512`. Three copies of one number
can drift.
Fixed by reading `classifier.tokenizer.model_max_length` once at startup.
Verified: startup logs `Model loaded: 14 classes, 512 token limit`.

**N-5 (MEDIUM, fixed) - nothing checked that the label order in `app.py` matched
the model.** A mismatch is invisible: every metric is unchanged and every
displayed class name is wrong.
Fixed with a startup comparison of `config.id2label` against `LABEL_NAMES` that
raises `SystemExit` on any difference. This also let the dead `LABEL_` branch
(L-1) be deleted, replacing dead defensive code with live defensive code.

**N-6 (MEDIUM, fixed) - `train.py` wrote its metrics into the gitignored folder.**
The first pass added `metrics.json` and `classification_report.txt` but wrote
them into `SAVE_DIR`, which is `saved_model/`, which `.gitignore` excludes.
They would never have reached the repository.
Fixed by writing them to the project root. Verified by the training smoke test.

**N-7 (MEDIUM, fixed) - `evaluate.load()` put a network call inside training.**
`compute_metrics` called `evaluate.load("accuracy")` and `evaluate.load("f1")`,
which download metric scripts from the Hugging Face Hub. scikit-learn was
already imported and computes identical values.
Fixed by using `accuracy_score` and `f1_score` directly and dropping `evaluate`
from `requirements-train.txt`. The training recipe is unchanged; only the source
of two numbers changed. Verified by the smoke test.

**N-8 (LOW, fixed) - `dataset['train'].features['label'].num_classes` could crash.**
The first pass added this assertion. `.num_classes` exists only on a `ClassLabel`
feature. Since I cannot execute `train.py` against the real dataset here, an
assertion that can itself raise `AttributeError` is a bad trade.
Fixed with `getattr(..., 'num_classes', None)` and a `ValueError` only on a real
mismatch.

**N-9 (LOW, fixed) - emoji in `print()` calls.**
Beyond the style preference, `print("⏳ ...")` raises `UnicodeEncodeError` if
stdout resolves to an ASCII codec, which happens in minimal containers with
`LANG=C`. Removed from `app.py` and `train.py`.

**N-10 (INFO, not fixed, documented) - the checkpoint uses deprecated LayerNorm
key names.** `saved_model/model.safetensors` stores LayerNorm parameters as
`gamma` and `beta`, the old TensorFlow-style names inherited from the original
LEGAL-BERT checkpoint, rather than `weight` and `bias`.
I verified that `transformers` 5.16.1 remaps them correctly:
`LayerNorm.weight` loads bit-identically from `gamma`, `LayerNorm.bias` from
`beta`, and the loaded LayerNorm weights have mean 0.923, not the 1.0 a randomly
initialised LayerNorm would have. So the model is loading fully today.
The risk is future: if that legacy remapping is ever dropped, every LayerNorm
would silently reinitialise and accuracy would collapse with no error. The
mitigation is one line, `model.save_pretrained(...)` with a current
`transformers`, which rewrites the keys. Not done here because it would change
the 140 MB artifact that the live Space is serving, and I cannot verify the
Space end to end from this environment.

**N-12 (HIGH, fixed) - `COPY . .` inherited unreadable file modes and the
container could not start.** Found by running the real `docker build` on the
author's Mac, which I could not do from this environment.

`COPY` preserves the source file mode. The project files on that machine were
mode `600`, owner-only. They were copied into the image as root-owned `600`,
then `USER user` switched to UID 1000, which could no longer read them.
Gunicorn's worker died on boot with:

```
PermissionError: [Errno 13] Permission denied: '/app/app.py'
[ERROR] Worker (pid:7) exited with code 3.
[ERROR] Reason: Worker failed to boot.
```

This never surfaced on Hugging Face because git stores only the executable bit:
every one of these files is recorded as mode `100644`, so a Space checkout
produces world-readable files and the original `COPY . .` happened to work. The
image was therefore silently dependent on the permissions of whatever machine
built it.

FIX: move `RUN useradd` above the source copy and use `COPY --chown=user:user . .`.
The copy is placed after the pip layer so the expensive install stays cached.
Mode `600` owned by `user` is readable by `user`, so the image is now correct
regardless of the build machine's file modes.

**N-11 (INFO, not fixed, documented) - `FROM python:3.12-slim` is a moving tag.**
Pinning to a digest would be strictly more reproducible. I did not do it because
I cannot reach any container registry from this environment, so I cannot verify
that a specific digest or patch tag exists, and an unverified pin would break
the build outright rather than merely drift.

## Change records

**FILE: app.py**
PROBLEM: CWD-dependent model path; opaque failure when the model is absent;
HTML error bodies from a JSON API; hardcoded token limit; no label-order check;
unreachable `LABEL_` branch; emoji in logs.
WHY IT MATTERS: N-1 to N-5 and N-9 above. N-5 is the serious one: it is the only
failure mode in this project that produces confidently wrong output with no
error anywhere.
FIX: path resolved from `__file__`; `SystemExit` with a readable message if the
model directory is missing; `errorhandler` for 404/405/413; `MAX_TOKENS` read
from the tokenizer; startup comparison of `id2label` against `LABEL_NAMES`; dead
branch deleted; emoji removed.
VERIFICATION: 19-check suite passing under both the Flask dev server and the
production gunicorn command; missing-model and label-mismatch paths triggered
deliberately.

**FILE: train.py**
PROBLEM: metrics written into the gitignored `saved_model/`; a Hub download
inside `compute_metrics`; an assertion that could itself raise; emoji in output;
`trainable_params` never recorded.
WHY IT MATTERS: N-6, N-7, N-8, N-9. N-6 silently defeated the fix that was
supposed to solve the project's most serious gap.
FIX: metrics to `./metrics.json` and `./classification_report.txt`;
scikit-learn instead of `evaluate`; defensive class-count check; emoji removed;
`trainable_params` added to `metrics.json` so the full-fine-tuning claim is
evidenced by the artifact rather than asserted.
VERIFICATION: executed end to end on a synthetic 64/32/32 fixture with the same
schema as LexGLUE SCOTUS, against the local model directory. 19 assertions
passed, including that `effective_batch_size == 16`, `learning_rate == 2e-05`,
`trainable_params == total_params == 35,075,598`, `save_total_limit` was
respected, and `id2label` survived into the saved config. The recipe itself is
byte-for-byte unchanged.

**FILE: requirements.txt / requirements-train.txt**
PROBLEM: PyPI's Linux `torch` wheel installs about 4 GB of CUDA libraries onto a
CPU-only Space. `evaluate` was no longer needed.
FIX: `--extra-index-url https://download.pytorch.org/whl/cpu`; `evaluate` removed.
VERIFICATION: measured the CUDA payload at 4.1 GB of a 5.6 GB tree. Confirmed a
clean `--only-binary=:all:` install on CPython 3.12 succeeds from PyPI alone, so
the fallback path is safe. Did **not** verify resolution of the `+cpu` wheel,
because the host is blocked here.

**FILE: Dockerfile**
PROBLEM: comment noise; the no-compiler claim was asserted, not tested.
FIX: comments reduced to the three non-obvious facts (layer-cache ordering, why
no compiler is needed, why the gunicorn flags are what they are). No instruction
changed in this pass.
VERIFICATION: the pip step was reproduced outside Docker on CPython 3.12 with
`--only-binary=:all:`. `docker build` itself could not be run, see below.

**FILE: verify_app.py (new)**
PROBLEM: the project had no tests at all.
FIX: a 19-check suite using only the standard library, pointed at a base URL, so
it runs against the dev server, gunicorn, or a running container unchanged.
VERIFICATION: it is the verification.

**FILE: .gitignore**
PROBLEM: `checkpoints/`, the `TrainingArguments` output directory, was not
ignored and could be committed by accident.
FIX: added.

## Verification matrix

| Check | Result | Evidence |
|---|---|---|
| Python imports (app) | PASS | app boots under 3.12 with pinned deps |
| Python imports (train) | PASS | runs to the `load_dataset` call |
| Requirements install, wheels only, CPython 3.12 | PASS | 57 packages, no source builds, 2m07s |
| Application startup | PASS | `Model loaded: 14 classes, 512 token limit` |
| Model loaded once, not per request | PASS | one `Loading model...` line under both servers |
| GET homepage | PASS | 200, `text/html`, template rendered |
| Valid inference (short/normal/long) | PASS | see suite output |
| Empty, whitespace, missing key, non-string | PASS | 400 with JSON |
| Wrong content type, malformed JSON, array body | PASS | 415 with JSON |
| Oversized body | PASS | 413 with JSON |
| Unknown route, wrong method | PASS | 404 / 405 with JSON |
| Long input uses the full token window | PASS | decoy document classified Civil Rights 0.851 |
| Determinism across calls | PASS | identical float scores |
| Concurrency under gthread | PASS | 12 concurrent requests, all correct, 0.2 s |
| Production gunicorn command | PASS | exact `CMD` from the Dockerfile |
| Clean shutdown on SIGTERM | PASS | `Handling signal: term`, `Shutting down: Master` |
| Tokenizer / model compatibility | PASS | tokenizer vocab 30522 == embedding rows |
| Full weight load, no reinitialisation | PASS | LayerNorm loaded from legacy `gamma`/`beta`, mean 0.923 |
| Parameter count | PASS | 35,075,598 total, 35,075,598 trainable, 7,182 in the head |
| Training script end to end | PASS | synthetic fixture, 19 assertions |
| Real training and metrics | **NOT POSSIBLE** | see below |
| Docker build | **PASS, on the author's Mac** | 584s, all 6 stages, image `scotus-classifier:latest` |
| Docker runtime | **FIXED AFTER A REAL FAILURE** | first run failed with `PermissionError` on `/app/app.py`, see N-12 |
| Hugging Face Space runtime | **NOT VERIFIED** | see below |
| No secrets committed | PASS | repository scanned, none present |
| No large files staged | PASS | largest tracked file is 30 KB |

## What could not be verified here, and why

**Real training, and therefore real metrics.** `train.py` calls
`load_dataset("coastalcph/lex_glue", "scotus")`, which requires
`huggingface.co`. That host is blocked by this environment's egress policy, and
I did not attempt to route around it. I confirmed the exact failure point: the
script executes correctly through imports, seeding and configuration, and raises
at line 56 on the Hub request.
**Consequence: `metrics.json` does not exist and I did not create one. There are
still no measured numbers for this model.** Producing them requires you to run
`train.py` in Colab, which has both a GPU and Hub access. Fabricating plausible
numbers would have been worse than useless.

**Docker build and runtime.** This was resolved by the author running the build
on his own machine, and it immediately found N-12, a real defect that inspection
had missed. The build completed in 584 seconds with no compiler required, which
confirms the `build-essential` removal. Everything below describes why this
environment could not run it.

`docker build` fails here at the first instruction:
`failed to resolve source metadata for docker.io/library/python:3.12-slim:
unexpected status from HEAD request to registry-1.docker.io: 403 Forbidden`.
Docker Hub, and every other container registry I tried, is blocked by policy.
The pip layer, which is the only part of the build that can realistically fail
for project-specific reasons, was reproduced outside Docker on CPython 3.12.
The remaining unverified instructions are `FROM`, `useradd`, `USER`, `COPY` and
`CMD`, all standard, and the `CMD` process itself was verified directly.
**Run `docker build` once on your Mac before pushing to the Space.** If it fails
on a missing compiler, restore the `build-essential` line; the evidence says it
will not.

**Hugging Face Space runtime.** LOCAL VERIFIED, REMOTE NOT VERIFIED.
What I could establish by fetching the Space over HTTP:
- It exists, and its status line reads "Sleeping due to inactivity".
- Its `README.md` front-matter declares `sdk: docker`, no `app_port`, so the
  default 7860 applies, which matches `EXPOSE` and the gunicorn bind.
- Its repository root contains `saved_model/`, `static/`, `templates/`,
  `Dockerfile`, `README.md`, `app.py`, `requirements.txt`, `.gitattributes`,
  141 MB across 2 commits, with the weights in Git LFS.
- **Its `app.py` is still the original, pre-fix version.** It contains
  `results = classifier(text[:512])[0]` and `app.run(debug=True, port=7860)`.
What I could **not** establish: the hardware tier, whether a fresh build
currently succeeds, or the runtime behaviour, because the Space is asleep and I
cannot trigger or observe a build from here.
**Consequence: pushing to GitHub does not update the Space.** They are separate
git remotes. The live demo keeps the character-truncation bug until you push
these changes to the Space remote as well.
