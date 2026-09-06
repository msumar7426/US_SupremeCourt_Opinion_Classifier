# INTERVIEW_CHEAT_SHEET.md
### US Supreme Court Opinion Classifier - study sheet

**Rule for every answer below: nothing here is invented.** Where the repository
has no evidence for a claim, the answer says so. An interviewer will trust you
far more for "I didn't measure that" than for a number you can't defend.

**One thing to do before your first interview:** re-run `train.py` so you have an
accuracy and a macro-F1. Until then, every performance question gets the honest
answer in §"When you don't have a number".

---

## 30-second explanation

> "I built a web app that classifies US Supreme Court opinions into 14 legal
> issue areas - things like Criminal Procedure, First Amendment, Federal
> Taxation. I fine-tuned LEGAL-BERT, a BERT model pretrained on legal text, on
> the LexGLUE SCOTUS benchmark, then served it with Flask behind a simple web UI,
> containerised it with Docker, and deployed it as a Hugging Face Space. You
> paste in an opinion and it returns the top five predicted issue areas with
> confidence scores."

## 1-minute explanation

> "The problem is single-label multi-class text classification on legal
> documents. The dataset is LexGLUE SCOTUS - 5,000 training, 1,400 validation and
> 1,400 test opinions, each labelled with one of 14 Supreme Court Database issue
> areas.
>
> For the model I fine-tuned `nlpaueb/legal-bert-small-uncased`. It's a BERT
> pretrained by AUEB on 12 GB of legal text, and it's small - 6 layers, 512
> hidden, about 35 million parameters - which mattered because I had a Colab T4
> and a free CPU Space to deploy to. `AutoModelForSequenceClassification` adds a
> fresh 512→14 linear head, and I fine-tuned the whole network for 3 epochs with
> AdamW at 2e-5, effective batch 16, selecting the best checkpoint on validation
> macro-F1 rather than accuracy because the classes are imbalanced.
>
> Training saves the weights and the tokenizer into a `saved_model/` folder.
> The Flask app loads that once at startup into a `transformers` pipeline and
> keeps it in memory; `/predict` takes JSON, tokenizes, runs a forward pass,
> softmaxes, and returns the top 5 labels. The Dockerfile runs it under gunicorn
> on port 7860, which is what Hugging Face Spaces expects."

## 3-minute technical explanation

Everything above, plus:

> "**Preprocessing** is deliberately minimal - tokenization and truncation only.
> No lowercasing by hand, no stopword removal, no stemming. Those help
> bag-of-words models but hurt transformers, because the encoder was pretrained
> on natural text and you want inference to match pretraining. Lowercasing
> happens inside the tokenizer, which is `do_lower_case: true`.
>
> **Tokenization** is WordPiece over a 30,522-token vocabulary that LEGAL-BERT
> trained on legal text - so `petitioner`, `remand` and `habeas` are each a
> single token, while a rarer word like `certiorari` splits into `certi` +
> `##orari`. The tokenizer emits `input_ids` and an `attention_mask`, prepends
> `[CLS]` and appends `[SEP]`, and truncates at 512 tokens.
>
> **The forward pass:** IDs index a 30522×512 embedding table, position
> embeddings are added, and six transformer blocks each run 8-head self-attention
> and a 512→2048→512 feed-forward with residual connections and layer norm. I
> take the final `[CLS]` vector, push it through the pooler and a `Linear(512,14)`
> head, and get 14 logits. Softmax turns them into probabilities.
>
> **Training** is full fine-tuning - every one of the 35 million parameters
> updates, not just the 7,182 in the head. The loss is cross-entropy, which I
> never wrote: `BertForSequenceClassification` applies it internally because
> `problem_type` is `single_label_classification`. I used a 2e-5 learning rate
> with 100 warm-up steps, because large steps early would destroy the pretrained
> weights - catastrophic forgetting - and gradient accumulation of 2 to get an
> effective batch of 16 within T4 memory at 512-token sequences.
>
> **Two things I'd flag about my own project.** First, opinions run 5,000 to
> 10,000 tokens and BERT hard-caps at 512, so the model only ever sees the first
> couple of pages. The right fixes are hierarchical BERT, Longformer, or chunking
> and averaging logits. Second, I found a real bug when I audited it: the Flask
> route was doing `text[:512]`, which slices *characters*, not tokens. That's
> about 108 tokens of legal prose - so I was serving the model a fifth of the
> context I'd trained it on. On a document whose subject only becomes clear after
> the first few sentences, fixing it took the correct class from 0.47 to 0.85
> confidence, with no retraining."

---

## Architecture

```
TRAINING (Colab, once)
    LexGLUE SCOTUS + LEGAL-BERT-small
      -> tokenize to 512 tokens
      -> Trainer: 3 epochs, AdamW 2e-5, effective batch 16
      -> keep best checkpoint by validation macro-F1
      -> save_model + save_pretrained + metrics.json
                    |
             saved_model/   (copied out of Colab by hand)
                    |
SERVING (HF Space, always on)
    app.py at import -> pipeline() loads saved_model/ into RAM once
    browser -> main.js -> POST /predict
      -> tokenize -> BERT -> logits -> softmax -> top 5 -> JSON
      -> animated bars in index.html
```

## Training pipeline (memorise these numbers)
`coastalcph/lex_glue` config `scotus` · 5,000 / 1,400 / 1,400 · 14 classes ·
base `nlpaueb/legal-bert-small-uncased` (6 layers, 512 hidden, 8 heads, 35,075,598
params) · max_length 512 · batch 8 × grad-accum 2 = effective 16 · AdamW ·
lr 2e-5 · warmup 100 · weight decay 0.01 · 3 epochs · fp16 · seed 42 ·
`load_best_model_at_end` on `f1_macro`.

## Inference pipeline
`pipeline("text-classification", model="./saved_model", top_k=5)` loaded at
import → `POST /predict` JSON → `truncation=True, max_length=512` →
`torch.no_grad()` forward → softmax → `id2label` → top 5 → `jsonify`.

## Model explanation
Full fine-tuning (transfer learning) of a domain-pretrained BERT. Body =
pretrained on 12 GB legal text via masked language modelling. Head = new random
`Linear(512, 14)`. Loss = cross-entropy. Prediction = `argmax(softmax(logits))`.
"Confidence" = `max(softmax)`, which is uncalibrated.

## Docker explanation
`python:3.12-slim` → `WORKDIR /app` → copy `requirements.txt` alone and pip
install (so the expensive layer stays cached when code changes) → copy the code →
create non-root UID 1000 (HF convention) → `EXPOSE 7860` (documentation only) →
`CMD` runs gunicorn, a real WSGI server, bound to `0.0.0.0:7860` in exec form so
it gets signals as PID 1. Image = the built snapshot; container = a running
instance of it.

## Hugging Face deployment explanation
The Space's own `README.md` front-matter says `sdk: docker`, so HF builds the
Dockerfile and runs the image, routing HTTPS to port 7860 (its default, matching
`EXPOSE` and the gunicorn bind). The 140 MB model is committed to the Space via
Git LFS - nothing downloads at runtime. No environment variables or secrets.
On restart the container is rebuilt from the image; only committed files survive.

## Key technologies
Python · PyTorch · HuggingFace `transformers` (`AutoTokenizer`,
`AutoModelForSequenceClassification`, `Trainer`, `pipeline`) · `datasets` ·
scikit-learn metrics · Flask · Jinja2 · vanilla JS `fetch` ·
gunicorn · Docker · Hugging Face Spaces · Git / Git LFS.

## Biggest technical challenge
**Honest version:** getting a 140 MB artifact from a Colab session into a live
deployment. GitHub rejects files over 100 MB, so `saved_model/` is gitignored -
which means the GitHub repo cannot actually run. The Space works because HF
Spaces uses Git LFS. *"With hindsight the right answer was Git LFS on GitHub too,
or pushing the model to the HF Model Hub and pulling a pinned revision at build
time, instead of hand-copying a folder."*

## Biggest limitation
**512-token truncation.** Supreme Court opinions are 5,000-10,000+ tokens; BERT
cannot exceed 512 positions. The model classifies roughly the first two pages and
never sees the reasoning or holding. Second limitation: class imbalance in SCOTUS
is real and I didn't address it, so rare classes like Private Action and
Interstate Relations are almost certainly weak.

## What I would improve
1. A TF-IDF + Logistic Regression baseline, so I can say what the transformer bought me.
2. Class-weighted cross-entropy, and check macro-F1 moved.
3. Chunk long opinions into 512-token windows and average the logits.
4. Version the model on the HF Model Hub instead of copying a folder by hand.
5. Three pytest tests: model loads, `/predict` returns 200, a known text → known label.
6. CPU-only torch wheels and a multi-stage Docker build to shrink the image.
7. Calibration (temperature scaling) so the confidence number means something.

---

## Building this from scratch - the conceptual order

| # | Stage | WHAT | WHY | INPUT | OUTPUT | CONCEPTS YOU NEED |
|---|---|---|---|---|---|---|
| 1 | Define the problem | Single-label multi-class classification of legal documents | Task type dictates loss, metrics and output layer | A goal | "14-way classification, softmax + cross-entropy" | classification vs generation vs NER; single- vs multi-label |
| 2 | Get the data | `load_dataset("coastalcph/lex_glue","scotus")` | A public benchmark gives labels *and* comparable published results | Dataset name | `DatasetDict` with 3 splits | HF `datasets`, benchmarks |
| 3 | Define labels | 14 SCDB issue areas, indices 0-13 | The model outputs integers; humans need names, and the mapping must be recorded | Dataset + SCDB codebook | `LABEL_NAMES`, `id2label` | label encoding; why index order is a silent-failure risk |
| 4 | Split | Use the benchmark's splits as given | Custom splits break comparability and invite leakage | `DatasetDict` | train/val/test | train vs validation vs test; data leakage |
| 5 | **Baseline** | TF-IDF + Logistic Regression | Without it you cannot say the transformer helped. *This project skipped it.* | Raw text | A number to beat | bag-of-words, TF-IDF |
| 6 | Choose a pretrained model | `legal-bert-small-uncased` | Domain match + fits a T4 and a free CPU Space | Task + constraints | `MODEL_NAME` | transfer learning; domain pretraining; model size vs cost |
| 7 | Tokenize | `tokenizer(text, truncation=True, max_length=512)` | Models read integers, not text; 512 is BERT's hard cap | Raw text | `input_ids`, `attention_mask` | WordPiece, `[CLS]`/`[SEP]`, attention mask, dynamic padding |
| 8 | Configure training | epochs, lr, batch, accumulation, warmup, metric | These are the knobs that decide whether it learns or forgets | - | `TrainingArguments` | learning rate, warmup, weight decay, gradient accumulation, fp16 |
| 9 | Fine-tune | `trainer.train()` | Adapt 12 GB of legal pretraining to 14 labels using only 5,000 examples | Tokenized train + val | Trained weights | forward pass, cross-entropy, backprop, AdamW, catastrophic forgetting |
| 10 | Evaluate | `trainer.predict(test)` + classification report | An unmeasured model is unvalidated | Test split | accuracy, macro/micro F1, per-class | macro vs micro F1, precision/recall, imbalance |
| 11 | Save | `save_model` + `save_pretrained` + metrics | Weights are useless without the tokenizer and the config | Trained model | `saved_model/` | serialization, safetensors, `id2label` |
| 12 | Inference pipeline | `pipeline(...)` loaded once at import | Loading per request would be catastrophically slow | `saved_model/` | a callable classifier | `torch.no_grad`, softmax, top-k |
| 13 | Flask backend | `/` and `/predict` | Something has to expose the model over HTTP | HTTP request | JSON response | routes, GET vs POST, status codes, JSON |
| 14 | Frontend | textarea + fetch + bars | People don't `curl` | User text | rendered results | DOM, `fetch`, async/await |
| 15 | Connect them | JSON contract on `/predict` | Both sides must agree on shape and content type | - | working app | HTTP content types, CORS (same-origin here) |
| 16 | Test locally | `python app.py`, try edge cases | Find the 415s and the crashes before deployment does | - | confidence | manual + automated testing |
| 17 | Dockerize | Dockerfile + `.dockerignore` | "Works on my machine" isn't deployable | Project folder | an image | images vs containers, layers, caching, non-root |
| 18 | Test the container | `docker build`, `docker run -p 7860:7860` | The container is what actually ships | Image | verified image | port mapping, exec form, logs |
| 19 | Deploy | push to a Space with `sdk: docker` | Someone else has to be able to open it | Repo + LFS | live URL | Spaces, Git LFS, port conventions |
| 20 | Test production | Open it, try the samples, read build logs | Deployment introduces its own failures | Live URL | a working demo | cold starts, build logs, persistence |

---

## Learning roadmap - only what THIS project needs

**1. Python essentials you're already using**
Decorators (`@app.route` registers a function with Flask), context managers,
`if __name__ == '__main__'` (why gunicorn skips `app.run`), slicing (`text[:512]`
 - the bug), f-strings. *Where:* every file.

**2. NumPy - just enough**
`np.argmax(logits, axis=-1)` picks the winning class; NumPy scalars are why
`float()` is needed before `jsonify`. *Tiny example:* `np.argmax([0.1,0.7,0.2]) → 1`.
*Where:* `train.py` `compute_metrics`, `app.py` `float(r['score'])`.

**3. NLP preprocessing & tokenization** ← *most important for this project*
Why transformers skip classical cleaning; WordPiece; `[CLS]`/`[SEP]`; attention
masks; truncation vs padding. *Try:* `tokenizer.tokenize("certiorari")` →
`['certi','##orari']`. *Where:* `tokenize_fn` in `train.py`, the pipeline in `app.py`.

**4. Classification & metrics**
Precision, recall, F1; macro vs micro vs weighted; confusion matrices; why
accuracy misleads on imbalanced data. *Where:* `compute_metrics`,
`classification_report`. *Why:* you will be asked "why macro-F1?".

**5. Neural network basics**
Logits → softmax → probabilities; cross-entropy loss; gradient descent;
backpropagation; epochs, batches, learning rate. *Tiny example:* softmax of
`[2.0, 1.0, 0.1]` → `[0.66, 0.24, 0.10]`. *Where:* implicit in `Trainer`, visible
in the `score` field the app returns.

**6. PyTorch - reading level, not authoring level**
Tensors and shapes, `nn.Linear`, `.eval()`, `torch.no_grad()`,
`torch.cuda.is_available()`. *Where:* `device=0 if torch.cuda.is_available() else -1`.

**7. Transformers & attention**
Self-attention intuition (every token attends to every other), multi-head,
residuals + layer norm, contextual vs static embeddings, encoder-only (BERT) vs
decoder-only (GPT). *Where:* the 6 layers inside `saved_model/`.

**8. Fine-tuning & the HF ecosystem**
`AutoTokenizer` / `AutoModelForSequenceClassification` / `Trainer` /
`TrainingArguments` / `pipeline`; what `from_pretrained` loads and what it
randomly initialises; `save_pretrained`; `config.json` and `id2label`.
*Where:* all of `train.py`, the top of `app.py`.

**9. Flask & HTTP**
Routes, GET vs POST, JSON request/response, status codes (200/400/415/500),
templates and static files, WSGI and why gunicorn. *Where:* `app.py`, `main.js`.

**10. Docker**
Image vs container, layers and cache invalidation, `FROM/WORKDIR/COPY/RUN/CMD/
EXPOSE/USER/ENV`, port mapping, build context and `.dockerignore`.
*Where:* the Dockerfile. *Why:* it's the part you said you understand least, and
it's the easiest to get concrete about - build it and shell into it.

**11. Hugging Face Spaces**
Space vs Model vs Dataset repos; `sdk: docker`; port 7860; Git LFS; ephemeral
filesystems and cold starts. *Where:* the Space's README front-matter.

---

## 20 likely interview questions

Format: **Q** → *what a strong answer contains* → **model answer** → *follow-ups*.

**1. What does your project do?**
*Strong answer:* task type, input, output, deployment, in under 30 seconds.
→ Use the 30-second explanation above.
*Follow-ups:* Who would use it? What's an issue area?

**2. What NLP problem is this?**
*Strong:* names the task type precisely and rules out neighbours.
→ "Single-label multi-class document classification - 14 mutually exclusive
classes. Not multi-label: `config.json` says `single_label_classification`, so
it's softmax + cross-entropy, one answer per document. Not generation, not NER."
*Follow-ups:* How would you change it to multi-label? *(sigmoid per class + BCE
loss + a threshold instead of argmax.)*

**3. What dataset did you use?**
→ "LexGLUE SCOTUS, `coastalcph/lex_glue`. 5,000 train / 1,400 validation / 1,400
test opinions, each labelled with one of 14 Supreme Court Database issue areas. I
used the benchmark's own splits rather than making my own, so results are
comparable and I'm not risking leakage."
*Follow-ups:* Why not split it yourself? Did you check for duplicates across
splits? *(Honest: "No - that's a gap. I trusted the benchmark's construction.")*

**4. What model, and why that one?**
→ "`nlpaueb/legal-bert-small-uncased`. Two reasons. Domain: it was pretrained on
12 GB of legal text, so its vocabulary and representations already fit legal
English - `petitioner` and `habeas` are single tokens in its vocabulary. And
size: 6 layers, 512 hidden, ~35 M parameters, about a third of BERT-base, which
fit a Colab T4 for training and a free CPU Space for serving."
*Follow-ups:* Did you compare against `bert-base-uncased`? → **"No, and I should
have. I chose it on domain fit and resource budget, not on a measured
comparison."** Would a bigger model help?

**5. Is this fine-tuning, or feature extraction?**
→ "Full fine-tuning. There's no `requires_grad = False` and no LoRA anywhere -
all 35,075,598 parameters update. `AutoModelForSequenceClassification` loads the
pretrained body and adds a fresh, randomly initialised `Linear(512, 14)` head,
which is only 7,182 parameters. Almost all the learning capacity is in the body,
which is exactly why you want to fine-tune it rather than freeze it."
*Follow-ups:* When *would* you freeze? *(Very little data, or very tight compute.)*

**6. What happens during tokenization?**
→ Walk it: lowercase → WordPiece → `[CLS]` ... `[SEP]` → vocabulary IDs → attention
mask. Give the real example: `"the"` → 207 in this vocabulary, not 1996 as in
`bert-base-uncased`, because LEGAL-BERT has its own vocabulary.
*Follow-ups:* What is `##`? What if a word isn't in the vocabulary?

**7. Why save the tokenizer with the model?**
→ "Because the weights learned that vocabulary index 1818 means `petitioner`. A
different tokenizer maps that word to a different integer and the model is
reading noise. The worst part is it fails silently - no exception, accuracy just
collapses toward chance."

**8. What loss function did you use?**
→ "Cross-entropy. I never wrote it: `BertForSequenceClassification.forward`
applies `CrossEntropyLoss` internally because `problem_type` is
`single_label_classification`. For a correct prediction with probability 0.92 the
loss is `-log(0.92) ≈ 0.088`; if it had put 0.01 on the right class it'd be 4.6."
*Follow-ups:* Why cross-entropy and not MSE? What loss for multi-label?

**9. What optimizer and learning rate, and why?**
→ "AdamW at 2e-5 with 100 warm-up steps and linear decay. 2e-5 is small on
purpose - a from-scratch net uses ~1e-3, but the pretrained weights are already
good and big steps destroy them. That's catastrophic forgetting. Warmup exists
because the classification head starts random, and its early gradients would
otherwise wreck the body."

**10. Batch size 8 with accumulation 2 - why not just 16?**
→ "512-token sequences are memory-hungry and the T4 wouldn't hold 16. Gradient
accumulation runs 8, keeps the gradients, runs another 8, adds them, then steps -
so the update sees 16 examples' worth of gradient with 8 examples' worth of
memory. It's slower in wall-clock but mathematically close to batch 16."

**11. How did you evaluate it?**
→ "Validation macro-F1 after every epoch, with `load_best_model_at_end` so I keep
the best checkpoint rather than the last one. Then accuracy, macro-F1, micro-F1
and a per-class report on the test set, touched once at the end. Macro-F1 is the
headline because the classes are imbalanced and it weights every class equally.
Micro-F1 is actually identical to accuracy for single-label multi-class, so it
adds nothing - I print it, but I know it's redundant."
*Follow-ups:* Why not accuracy? What's your worst class?

**12. What accuracy did you get?**
→ **See "When you don't have a number" below. Do not guess.**

**13. How does inference work end to end?**
→ Give the `DATA_FLOW.md` chain: fetch POST → Flask route → `get_json` →
tokenizer → `input_ids (1,43)` → embeddings `(1,43,512)` → 6 layers → `[CLS]`
`(1,512)` → `Linear(512,14)` → logits `(1,14)` → softmax → top 5 → JSON → bars.

**14. Where is the model loaded, and why there?**
→ "At module level in `app.py`, so it happens once when the process starts and
the model stays in RAM. If I loaded it inside the route, every request would read
140 MB off disk and rebuild the network - hundreds of times slower. The trade-off
is startup time and one copy of the model per gunicorn worker, which is why I run
one worker with four threads rather than four workers."

**15. Why Flask? Why gunicorn on top?**
→ "Flask because I needed two endpoints and a template - anything heavier would
be scaffolding I don't use. Gunicorn because Flask's built-in server is a
development server: single-threaded by default, no process management, and it
says so in its own startup warning. Gunicorn is a real WSGI server that manages
workers and handles signals properly."
*Follow-ups:* What is WSGI? Why not FastAPI? *(Fair answer: async and automatic
validation would be nice, but this workload is CPU-bound so async buys little.)*

**16. Walk me through your Dockerfile.**
→ Use `PROJECT_ARCHITECTURE.md` §4. Lead with the layer-caching point - copying
`requirements.txt` before the code so a code edit doesn't re-download PyTorch -
because that's the line that shows you understand Docker rather than copied it.

**17. Why Docker at all?**
→ "So the thing running on Hugging Face is byte-identical to the thing I ran
locally: same Python, same library versions, same OS packages. And Spaces' Docker
SDK gives me full control of the runtime rather than fitting into a Gradio or
Streamlit template."

**18. How is it deployed, and what happens on restart?**
→ "It's a Hugging Face Space with `sdk: docker`. HF builds my Dockerfile and runs
the image, routing HTTPS to port 7860 - which matches `EXPOSE` and the gunicorn
bind. The model is committed into the Space repo with Git LFS, so nothing
downloads at runtime. On restart the container is recreated from the image, so
only committed files survive; anything written at runtime is gone. Free CPU
Spaces also sleep, so the first request after idle pays a cold start."

**19. What are the weaknesses of this project?** ← *the question that separates candidates*
→ "Several, and I'd rather name them than have you find them.
> **One:** 512-token truncation. Opinions run 5,000-10,000 tokens; the model sees
> the first two pages. The fixes are hierarchical BERT, Longformer, or chunking
> and averaging logits.
> **Two:** I never built a baseline. A TF-IDF + logistic regression model takes
> ten minutes and would tell me what the transformer actually bought. Skipping it
> is the thing I'd change first.
> **Three:** class imbalance is real in SCOTUS and I didn't address it - no class
> weights, no resampling. Macro-F1 measures the damage but doesn't fix it.
> **Four:** no tests, and the model artifact is hand-copied out of Colab with no
> version or checksum.
> **Five:** I audited the code afterwards and found a real bug: the Flask route
> did `text[:512]`, which slices characters, not tokens - about 108 tokens of
> legal prose. I was serving the model a fifth of the context I trained it on.
> Fixing it to `truncation=True, max_length=512` took the correct class from 0.47
> to 0.85 confidence on a long document, with no retraining."

**20. How would you handle 100 concurrent requests?**
→ "Right now I couldn't - one gunicorn worker with four threads, on CPU. In
order: batch requests inside a short time window so the GPU/CPU does one forward
pass for many inputs; add workers, bounded by memory since each holds its own
140 MB copy; export to ONNX Runtime or quantise to int8 for a several-fold CPU
speedup; then horizontal autoscaling behind a load balancer with a queue.
I'd measure p50/p95 latency first, because the right answer depends on whether
I'm latency-bound or throughput-bound."
*Follow-ups:* Workers vs threads and why? Where would you cache?

### Bonus questions you should be ready for
- **What exactly is updated during fine-tuning?** All 35 M parameters - embeddings, attention matrices, feed-forwards, pooler, and the head.
- **Why is train/test leakage a problem?** Your test score stops estimating performance on unseen data and becomes a memorisation score; you ship something that looks good and isn't.
- **What if the inference tokenizer differed from training?** Same words map to different IDs; the model reads noise; accuracy collapses toward 1/14 - **with no error message**. That's why the tokenizer lives in `saved_model/`.
- **How would you reduce the Docker image?** CPU-only torch wheels instead of the CUDA build; drop the unused `build-essential`; drop training-only libraries from the serving image; multi-stage build; `.dockerignore`. *(All done except the multi-stage build.)*
- **What would you do with two more weeks?** Baseline, class weighting, long-document chunking, tests + CI, model versioning on the HF Hub, calibration.

---

## When you don't have a number

You currently have **no recorded accuracy or F1** - `train.py` printed them into
a Colab session that's gone. Do not invent one; a fabricated metric is the
fastest way to fail an interview, and "what was your worst class?" will expose it
in one follow-up.

**Say this instead:**
> "I can't quote you a number honestly, and that's a real gap. The script
> computed accuracy, macro-F1 and a per-class report, but it only printed them -
> nothing was saved, and the Colab session is gone. I've since changed the
> training script to write `metrics.json` and `classification_report.txt` into
> the repository root, so it cannot happen again. I would rather tell you that
> than guess."

That answer demonstrates measurement discipline, honesty and a fix. It is worth
more than a number you can't defend.

**Then go re-run `train.py` and replace this section with your real numbers.**
When you have them, be ready to state: accuracy, macro-F1, your best and worst
classes, and how the macro-F1 compares to a majority-class baseline.
