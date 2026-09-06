# PROJECT_ARCHITECTURE.md
### US Supreme Court Opinion Classifier - how this repository actually works

> Everything in this document was read out of the repository and verified by
> running the app. Where the repository does not contain the answer, it says
> **UNKNOWN FROM THE REPOSITORY**.
>
> Section 0 records the **original** state of the repository as found. Everything
> from section 1 onward describes the **current** code. `PROJECT_AUDIT.md` lists
> what changed between the two and why.

---

## 0. Baseline: does it work, and how?

**Yes, it works.** I installed the dependencies, started the app, and sent real requests.

| Question | Verified answer |
|---|---|
| How is it started (local)? | `python app.py` → Flask dev server on `http://127.0.0.1:7860` |
| How is it started (container)? | `gunicorn -b 0.0.0.0:7860 app:app` (the Dockerfile's `CMD`) |
| Dependencies (as found) | `flask, transformers, torch, accelerate, scikit-learn, evaluate, datasets, gunicorn`, all unpinned. Now pinned and split into serving vs training sets. |
| Model | `nlpaueb/legal-bert-small-uncased` fine-tuned for 14-class classification, stored in `saved_model/` |
| Input | JSON `POST /predict` with `{"text": "<legal text>"}` |
| Output | JSON `{"results": [{"label": "...", "score": 0.91}, ... 5 items]}` |
| Consistent with README? | Mostly yes. Two gaps, see §8. |

Actual observed output for the "Criminal Procedure" sample text shipped in `static/js/main.js`:

```
Criminal Procedure 0.9163
Due Process 0.0168
Judicial Power 0.0094
First Amendment 0.0083
Economic Activity 0.0081
```

The model is not a toy - it's clearly learned something real. The engineering
around it has genuine problems, covered in `PROJECT_AUDIT.md`.

---

## 1. What problem is this solving?

**Task type: single-label multi-class text classification.**

Not generation, not NER, not sentiment, not similarity. One document in, one of
14 mutually exclusive categories out. You can prove this from the repo itself:
`saved_model/config.json` contains

```json
"architectures": ["BertForSequenceClassification"],
"problem_type": "single_label_classification",
"id2label": { "0": "Criminal Procedure", ... "13": "Private Action" }
```

`single_label_classification` is the flag that tells `transformers` to use
**cross-entropy loss with a softmax** (exactly one right answer), as opposed to
`multi_label_classification` which would use per-label sigmoids (many right
answers allowed).

- **Input:** the text of a US Supreme Court opinion (or any legal prose).
- **Output:** the predicted *issue area* - the subject-matter bucket the US
 Supreme Court Database assigns to each case - plus a confidence score.
- **Why it's useful:** legal-document triage/routing. Given thousands of
 unlabelled opinions, sort them by legal topic without a lawyer reading each one.

### The 14 classes
`0 Criminal Procedure · 1 Civil Rights · 2 First Amendment · 3 Due Process ·
4 Privacy · 5 Attorneys · 6 Unions · 7 Economic Activity · 8 Judicial Power ·
9 Federalism · 10 Interstate Relations · 11 Federal Taxation · 12 Miscellaneous ·
13 Private Action`

These come from the **LexGLUE / SCOTUS** benchmark (`coastalcph/lex_glue`,
config `scotus`), loaded in `train.py`.

---

## 2. The system map

Two pipelines that meet at one folder on disk.

```
================= TRAINING (runs once, in Google Colab, on a GPU) =================

  HuggingFace Hub
    coastalcph/lex_glue "scotus"          nlpaueb/legal-bert-small-uncased
              |                                        |
              v                                        v
    dataset[train|validation|test]     AutoTokenizer + AutoModelForSequenceClassification
              |                                        |   (BERT body + NEW 14-way head)
              +---------> tokenize_fn --------------->  |
                       (truncate to 512 TOKENS)         |
                                                        v
                                    Trainer.train()  3 epochs, lr 2e-5, AdamW
                                                        |
                                    trainer.predict(test_tok) -> accuracy / macro-F1
                                                        |
                                                        v
                                    trainer.save_model("./saved_model")
                                    tokenizer.save_pretrained("./saved_model")
                                    metrics.json, classification_report.txt
                                                        |
                                                        v
            saved_model/  (config.json, model.safetensors, tokenizer.json,
                           tokenizer_config.json, training_args.bin)
                           140 MB, copied out of Colab BY HAND
                                                        |
================ INFERENCE (runs continuously, on a HF Space) ====================
                                                        |
  USER (browser)                              app.py at module import time:
      |  types text, clicks "Classify"          classifier = pipeline(
      v                                             "text-classification",
  static/js/main.js                                 model="./saved_model",
      |  fetch('/predict', POST, JSON)              top_k=5)
      v                                          loaded ONCE, kept in RAM
  Flask route  @app.route('/predict', methods=['POST'])
      |  text = request.get_json(silent=True)['text']
      |  classifier(text, truncation=True, max_length=MAX_TOKENS)
      v
  tokenizer -> input_ids + attention_mask -> BERT (6 layers) -> [CLS] vector
            -> Linear(512 -> 14) -> logits -> softmax -> 14 probabilities -> top 5
      |
      v
  jsonify({'results': [{label, score} x 5]})
      |
      v
  main.js displayResults() -> innerHTML bars in templates/index.html
```

**The single most important structural fact:** training and serving are
*completely decoupled*. `app.py` never imports `train.py`. The only thing that
crosses the boundary is the `saved_model/` folder. If you understand that
folder, you understand the seam of this project.

### Where Docker fits
Docker packages `app.py` + `saved_model/` + a pinned OS + Python + all libraries
into one immutable image, so the thing that runs on Hugging Face's servers is
byte-identical to the thing you can run on your Mac. It is the *transport
mechanism*, not part of the ML.

### Where Hugging Face fits
Hugging Face appears **three separate times**, which is a common source of
confusion:
1. **As a dataset host** - `load_dataset("coastalcph/lex_glue", "scotus")` in `train.py`.
2. **As a model host** - `from_pretrained("nlpaueb/legal-bert-small-uncased")` in `train.py`.
3. **As the deployment platform** - the Space at
 `huggingface.co/spaces/msumar/NLP-Semester-Project`, which builds the Dockerfile and runs it.

(1) and (2) only matter at training time. (3) only matters at serving time.

---

## 3. File-by-file analysis

### 3.1 `app.py` - 64 lines, the entire backend

**Why it exists:** to hold a trained model in memory and expose it over HTTP.

**Imports and what each is for**

```python
from flask import Flask, render_template, request, jsonify
import torch # ONLY used for torch.cuda.is_available()
from transformers import pipeline # the high-level inference wrapper
```

**Execution flow, in order:**

**A. Application start (module import).** Everything at module level runs
*once*, before any request is served:

```python
app = Flask(__name__)
MODEL_PATH = "./saved_model"
LABEL_NAMES = [...] # 14 names, in index order
classifier = pipeline("text-classification",
 model=MODEL_PATH, tokenizer=MODEL_PATH,
 device=0 if torch.cuda.is_available() else -1,
 top_k=5)
```

`pipeline(...)` is not magic - it does four concrete things:
1. Reads `saved_model/config.json`, sees `"architectures": ["BertForSequenceClassification"]`,
 and instantiates that Python class with `hidden_size=512`, `num_hidden_layers=6`,
 `num_attention_heads=8`, `vocab_size=30522`.
2. Memory-maps `saved_model/model.safetensors` (140 MB) and copies the weights
 into that object's tensors. I watched it load 105 weight tensors.
3. Reads `saved_model/tokenizer.json` + `tokenizer_config.json` and builds a
 fast (Rust-backed) `BertTokenizerFast`.
4. Reads `id2label` from the config so it can turn output index `0` into the
 string `"Criminal Procedure"`.

`device=0 if torch.cuda.is_available() else -1` - `0` means "CUDA GPU #0",
`-1` means CPU. On Hugging Face's free CPU Spaces this always resolves to `-1`.

`top_k=5` makes the pipeline return the 5 highest-probability labels instead of
just the argmax.

> **`classifier` is a `TextClassificationPipeline` object.** It owns a
> `.model` (a `BertForSequenceClassification` `nn.Module`) and a `.tokenizer`.
> When you call `classifier(text)` you are invoking `__call__`, which does
> tokenize → `torch.no_grad()` forward pass → softmax → sort → format.

**B. Homepage - `GET /`**

```python
@app.route('/')
def index():
 return render_template('index.html')
```
`@app.route` registers a URL rule. With no `methods=` argument Flask defaults to
`GET` only. `render_template` finds `templates/index.html` (Flask hardcodes the
`templates/` folder name), runs it through Jinja2 - which resolves the
`{{ url_for('static', filename='css/style.css') }}` calls into real URLs - and
returns the HTML string with a `200 OK`.

**C. Prediction - `POST /predict`**

```python
data = request.get_json(silent=True)
if not isinstance(data, dict):
    return jsonify({'error': 'Request body must be JSON: {"text": "..."}'}), 415

text = data.get('text', '')
if not isinstance(text, str) or not text.strip():
    return jsonify({'error': 'No text provided'}), 400

results = classifier(text, truncation=True, max_length=MAX_TOKENS)[0]
```

- `request` is a thread-local proxy to the *current* HTTP request.
- `get_json(silent=True)` returns `None` instead of raising when the body is not
  JSON, which is what turns a wrong `Content-Type` into a 415 rather than a 500.
- `truncation=True, max_length=MAX_TOKENS` truncates by **token**. `MAX_TOKENS`
  is read from `classifier.tokenizer.model_max_length` at startup rather than
  hardcoded, so it cannot drift from what the tokenizer was saved with.
- `classifier(...)` returns `[[{...} x 5]]`, a list *per input string*, so `[0]`
  unwraps the single input's results.
- `float(r['score'])` converts a NumPy float32 to a Python float, because
  `json` cannot serialise NumPy types. This line is *load-bearing*, not noise.
- `jsonify` serialises a Python dict to JSON and sets `Content-Type: application/json`.

Three `@app.errorhandler` registrations (404, 405, 413) return JSON rather than
Flask's default HTML, so every response from this service is JSON and the
frontend's `response.json()` never fails on an error page.

**D. `if __name__ == '__main__': app.run(debug=True, port=7860)`**
Only runs when you type `python app.py`. In Docker, gunicorn imports `app.py`
as a module, so `__name__ != '__main__'` and this line is skipped entirely.

**Problems this file used to have**, all now fixed and documented in
`PROJECT_AUDIT.md`: `text[:512]` truncated *characters* not tokens;
`debug=True` loaded the model twice and exposed the Werkzeug debugger; `str(e)`
leaked internal errors to the client; a non-JSON body returned 500 instead of
415; there was no request size limit; a `LABEL_` fallback branch was dead code.

---

### 3.2 `train.py` - 175 lines
Fully dissected in **`TRAINING_AND_MODEL.md`**. Summary of its role: it is a
*Colab notebook exported as a script* (note the `# %% Cell N` markers). It is
never imported by anything. Deleting it would not break the running app - but it
is the only record of how `saved_model/` came to exist, which makes it the most
important file in the repo for your understanding.

---

### 3.3 `saved_model/` - the artifact that connects the two halves

| File | Size | What it is | Needed at inference? |
|---|---|---|---|
| `config.json` | 1.6 KB | Architecture hyperparameters + `id2label`/`label2id` | **Yes** - without it, nothing knows the shape of the network |
| `model.safetensors` | 140 MB | The learned weights (105 tensors, float32) | **Yes** |
| `tokenizer.json` | 702 KB | The full fast-tokenizer: 30 522-entry WordPiece vocabulary + normalisation + truncation rules | **Yes** |
| `tokenizer_config.json` | 322 B | Special tokens (`[CLS] [SEP] [PAD] [MASK] [UNK]`), `do_lower_case: true`, `model_max_length: 512` | **Yes** |
| `training_args.bin` | 5.2 KB | A pickled `TrainingArguments` object - a record of the hyperparameters used | **No** (provenance only) |

Key numbers you can read straight out of `config.json` and should be able to
quote in an interview:

```
hidden_size 512 ← each token becomes a 512-dim vector
num_hidden_layers 6 ← 6 transformer blocks (BERT-base has 12)
num_attention_heads 8
intermediate_size 2048 ← the feed-forward inner dimension
vocab_size 30522 ← standard uncased BERT WordPiece vocab
max_position_embeddings 512 ← HARD LIMIT: 512 tokens per input, ever
dtype float32
```

140 MB ÷ 4 bytes ≈ **35 million parameters**. That is why this is
"legal-bert-**small**" and not BERT-base (110 M).

> **Note:** `saved_model/` is listed in `.gitignore`. It is **not** in the GitHub
> repo. It **is** committed to the Hugging Face Space (which uses Git LFS). This
> is the reproducibility trap discussed in §8.

---

### 3.4 `templates/index.html` - 86 lines
A single Jinja2 template. Jinja is only used for two `url_for` calls; there is no
server-side data injection at all - the page is static, and results arrive later
via JavaScript. Structure: header → textarea + Classify/Clear buttons →
three sample buttons → hidden loading spinner → hidden error div → hidden
results div → `<script src="{{ url_for('static', 'js/main.js') }}">`.

### 3.5 `static/js/main.js` - 115 lines
Owns the whole client side. `SAMPLES` is a hard-coded 3-element array of demo
opinions. On click it does `fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})})`,
awaits JSON, and renders. **This is why the backend must accept JSON, not form
data** - the two files are coupled by that content type.

`displayResults` writes model output into the DOM with `innerHTML`. Here the
label strings come from your own fixed 14-name list so it is safe in practice,
but it is the pattern that causes XSS when the interpolated value is ever
user-controlled - a fair thing to flag about your own code.

### 3.6 `static/css/style.css` - 277 lines
Pure presentation: dark gradient theme, `.card`, `.pred-bar` animation. Nothing
functional. Safe to ignore for interview purposes.

### 3.7 `requirements.txt` and `requirements-train.txt`
As found, one file held eight unpinned packages, four of which
(`accelerate`, `scikit-learn`, `evaluate`, `datasets`) are used *only by
`train.py`* and were pure weight in the serving image.

Now `requirements.txt` holds the four serving dependencies at exact versions
plus `--extra-index-url https://download.pytorch.org/whl/cpu`, and
`requirements-train.txt` layers the training-only packages on top of it with
`-r requirements.txt`. Measured on a clean CPython 3.12 install, the PyPI Linux
`torch` wheel drags in 4.1 GB of CUDA libraries out of a 5.6 GB tree, none of
which a CPU Space can use; that is what the extra index avoids.

### 3.8 `.gitignore`
Standard Python ignores, plus `saved_model/` with the honest comment
`# model.safetensors is ~140MB and exceeds GitHub's 100MB limit`. Correct
diagnosis; the fix would have been Git LFS.

### 3.9 `README.md`
Accurate on the model, dataset and the 14 classes. Missing: the fact that a
GitHub clone cannot run, and any record of the model's measured accuracy.

---

## 4. Dockerfile - line by line

The Dockerfile in the repository no longer contains the `build-essential` and
duplicate `WORKDIR` lines. They are kept in this walkthrough because
understanding why they were removed is more useful than pretending they never
existed; each is marked REMOVED.

First, the three words people mix up:
- **Dockerfile** - a text recipe.
- **Image** - the built, immutable, layered filesystem snapshot. Like a class.
- **Container** - a running process using that image as its root filesystem. Like an instance.

```dockerfile
FROM python:3.12-slim
```
Start from Docker Hub's official image containing Debian + Python 3.12, `-slim`
meaning docs/man pages/extra libs stripped (~130 MB instead of ~1 GB).
Every later instruction stacks a new **layer** on top of this.
*Remove it and there is no base filesystem - the build cannot start.*

```dockerfile
WORKDIR /app
```
Creates `/app` if absent and makes it the current directory for all following
`COPY`/`RUN`/`CMD`. `app.py` now resolves the model path relative to its own
file rather than the working directory, so this is about where the code lands,
not about whether the model is found.

```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
```
Without the first, Python buffers stdout when it is not a terminal, so startup
messages and tracebacks can be lost or badly delayed in Hugging Face's log view.
The second stops `.pyc` files being written into the image layer.

```dockerfile
RUN apt-get update && apt-get install -y build-essential \
 && rm -rf /var/lib/apt/lists/*        # REMOVED
```
Installed gcc/g++/make. The `&& rm -rf` in the *same* `RUN` matters: each `RUN`
freezes a layer, so deleting the apt cache in a *later* instruction would not
shrink the image - the bytes would still live in the earlier layer.
**In this project `build-essential` is not needed, and this is now verified by
execution rather than by inspection.** I created a clean CPython 3.12
environment and ran `pip install --only-binary=:all: -r requirements.txt`. The
`--only-binary=:all:` flag makes pip refuse to build anything from source, so a
successful install proves no compiler is required. All 57 packages resolved to
prebuilt wheels: `torch`, `numpy`, `regex`, `pyyaml` and `markupsafe` as `cp312`
manylinux wheels, `tokenizers`, `safetensors` and `hf-xet` as `abi3` manylinux
wheels that CPython 3.12 accepts, and the rest as pure-Python `py3-none-any`.

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```
This ordering is **deliberate and correct**, and is the one genuinely
sophisticated thing in the Dockerfile. Docker caches layers and invalidates a
layer only when its inputs change. By copying *only* `requirements.txt` first,
the expensive `pip install` layer is reused whenever you edit `app.py`. If you
wrote `COPY . .` before the install, every code edit would re-download PyTorch.
`--no-cache-dir` stops pip keeping its ~1 GB download cache inside the image.

```dockerfile
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
```
Containers run as `root` by default; if the process is compromised, it is root
inside the container. Hugging Face Spaces additionally *expects* UID 1000.
`USER user` switches every subsequent instruction and the runtime process to
that account. `ENV` sets environment variables baked into the image, so
`$HOME` resolves for library caches.
The copy is `COPY --chown=user:user . .`, and `useradd` runs before it. This
matters more than it looks. A plain `COPY` **preserves the file mode of the
source**, so if the files on the build machine are mode `600`, they arrive in
the image as root-owned `600` and UID 1000 cannot read them. That is exactly
what happened on a real build: gunicorn died with
`PermissionError: [Errno 13] Permission denied: '/app/app.py'`. It never showed
up on Hugging Face because git records these files as `100644`, so a Space
checkout is world-readable. `--chown` removes the dependency on the build
machine entirely.

```dockerfile
WORKDIR /app        # REMOVED (second occurrence)
```
It was already `/app`, so the repeat did nothing but cost an empty layer.

```dockerfile
EXPOSE 7860
```
**Pure documentation.** It does not open a port. It records intent in the image
metadata for humans and for `docker run -P`. *Removing it changes nothing
functionally*, but HF Spaces reads it as a hint for which port to route to.

```dockerfile
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "app:app"]
```
The default command run when a container starts. **exec form** (a JSON array) -
so gunicorn becomes PID 1 directly and receives `SIGTERM` properly; the shell
form `CMD gunicorn ...` would wrap it in `/bin/sh -c` and swallow signals.
- `app:app` = "in the module `app` (i.e. `app.py`), use the object named `app`".
- `-b 0.0.0.0:7860` binds all interfaces. `127.0.0.1` would make the container
 unreachable from outside - a classic deployment bug.
- Gunicorn is a **WSGI server**: a real process manager that Flask's built-in
 dev server is not. Default here is **1 synchronous worker**, so exactly one
 request is processed at a time.

**`CMD` vs `ENTRYPOINT`:** `CMD` is a default that `docker run <image> <cmd>`
overrides; `ENTRYPOINT` is fixed and receives `CMD` as arguments. `CMD` alone is
the right choice here.

### The Docker lifecycle, and the commands for *this* project on macOS

```
Dockerfile --docker build--> Image --docker run--> Container --> gunicorn serving :7860
```

```bash
cd "$HOME/Downloads/US Supreme Court NLP Project"

# BUILD. "." is the build context: the whole folder is tarred and sent to the
# Docker daemon, which is why a .dockerignore matters.
docker build -t scotus-classifier .

# RUN. -p HOST:CONTAINER maps your Mac's port 7860 to the container's 7860.
docker run --rm -p 7860:7860 scotus-classifier
# then open http://localhost:7860

# Interactive debugging shell inside the image:
docker run --rm -it -p 7860:7860 scotus-classifier bash
# ls -la /app/saved_model ← the #1 thing to check when a Space fails

# Apple Silicon note: HF Spaces run linux/amd64. To reproduce their environment:
docker build --platform linux/amd64 -t scotus-classifier:amd64 .
```

---

## 5. Hugging Face Spaces deployment

Verified by fetching the live Space, not guessed.

- **Is it a Space?** Yes: `huggingface.co/spaces/msumar/NLP-Semester-Project`.
- **SDK:** `docker`. The Space's own `README.md` (223 bytes - a *different* file
 from the GitHub README) carries the YAML front-matter Spaces requires:
 ```yaml
 ---
 title: NLP Semester Project
 emoji:
 colorFrom: blue
 colorTo: green
 sdk: docker
 pinned: false
 license: apache-2.0
 ---
 ```
- **How does HF know how to start it?** Because `sdk: docker`, it ignores any
 Python conventions and simply runs `docker build .` on the repo root, then
 runs the image. The startup command is your `CMD` line. No `app_port` is set
 in the front-matter, so Spaces uses its default of **7860**, which matches
 `EXPOSE 7860` and the gunicorn bind. These three numbers agreeing is what
 makes it work.
- **Where does the model come from?** **Bundled in the repo.** The Space root
 contains `saved_model/`, `static/`, `templates/`, `Dockerfile`, `README.md`,
 `app.py`, `requirements.txt`, `.gitattributes` - total 141 MB, 2 commits. The
 140 MB `model.safetensors` is stored via **Git LFS** (that's what
 `.gitattributes` is for). Nothing is downloaded from the Hub at runtime.
 `train.py` is **not** deployed - correct, it isn't needed to serve.
- **What happens on deploy:** push to the Space's git remote → HF builds the
 Dockerfile (pip downloads PyTorch etc., several minutes) → image stored →
 container started → HF reverse-proxies HTTPS traffic to port 7860.
- **What happens when a user opens it:** the container is already running with
 the model in RAM; `GET /` returns `index.html`; the browser then hits
 `POST /predict`.
- **What happens on restart:** the container is destroyed and recreated from the
 image. **Only files committed to the repo survive.** Anything the app writes
 at runtime is lost. Free CPU Spaces also sleep after inactivity, so the first
 visit after a sleep pays the cold-start cost of loading 140 MB again.
- **Environment variables required:** none. The app reads no `os.environ`.
- **What could make the Space fail:**
 1. **Unpinned dependencies.** A future `transformers` or `torch` release with a
 breaking change would break the next rebuild with no code change on your
 part. This is the single largest deployment risk here.
 2. LFS not configured → `model.safetensors` lands as a 130-byte pointer file
 and `from_pretrained` fails.
 3. Build timeout / free-tier disk limits: the unpinned `torch` from PyPI pulls
 the **CUDA** build (multiple GB) onto a CPU-only Space.
 4. Free CPU Space RAM (16 GB) is fine for a 140 MB model - not a risk here.

**UNKNOWN FROM THE REPOSITORY:** the Space's hardware tier, whether the build
currently succeeds on a cold rebuild, and any Space secrets. To determine these
you'd need to look at the Space's *Settings* and *Build logs* tabs in the HF web UI.

---

## 6. Secrets check

I searched the repository for credentials. **No API keys, tokens, passwords or
`.env` files are present.** `train.py` calls `load_dataset` and `from_pretrained`
on public repos, which need no auth. `.git/config` contains only the public
GitHub remote `https://github.com/msumar7426/US_SupremeCourt_Opinion_Classifier.git`.
Nothing needs redacting.

---

## 7. Git state

3 commits, clean working tree, branch `main` tracking `origin/main` on GitHub.
```
48e724e Refine README: Simplify and focus on LegalBERT
be887f2 Add Hugging Face demo link to README
031505b Initial commit: US Supreme Court NLP Project (9 files, 871 lines)
```
The entire project landed in **one commit** - consistent with a 2-3 day deadline.
There is no HF remote configured here, so the Space was pushed from a separate
clone or via the HF web UI. Be ready for "walk me through your commit history" -
the honest answer is "it was a deadline project committed in one shot; I'd
develop it incrementally now."

---

## 8. README vs reality

| README says | Reality |
|---|---|
| `pip install -r requirements.txt` then `python app.py` | Correct - **but only if you already have `saved_model/`**. It's gitignored, so a fresh `git clone` from GitHub gives you code with no weights and `app.py` crashes at import with `OSError: ./saved_model does not appear to have a file named config.json`. The README never says this. |
| "fine-tuned LegalBERT" | Accurate, and it is genuine full fine-tuning (see `TRAINING_AND_MODEL.md`). |
| Lists the 14 classes | Matches `config.json` exactly. |
| - | **No accuracy or F1 is reported anywhere in the repo.** `train.py` prints them but never saves them. You currently cannot honestly state a number for this model. |
