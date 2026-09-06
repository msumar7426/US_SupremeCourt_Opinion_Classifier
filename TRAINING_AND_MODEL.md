# TRAINING_AND_MODEL.md
### `train.py` and the model, explained from the ground up

---

## PART A - What kind of "training" is this, really?

You asked me not to accept the word "fine-tuning" just because the repo uses it.
Here is the technical answer.

There are four things people call "training" with a pretrained model:

| Approach | What happens | Is this it? |
|---|---|---|
| **Training from scratch** | Random weights everywhere; the model learns English *and* the task | No |
| **Feature extraction** (frozen encoder) | BERT weights frozen; only a small classifier on top is trained | No |
| **Full fine-tuning** | Pretrained weights loaded, then *all* of them keep updating on the new task | **Yes, this is it** |
| **Parameter-efficient FT** (LoRA/adapters) | Base frozen; small trainable modules inserted | No |

**Proof from the code, not from the label.** In `train.py` there is:
```python
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=14)
```
and then `trainer.train()`. There is **no** `param.requires_grad = False`
anywhere, no `freeze` call, no LoRA config. In PyTorch every parameter has
`requires_grad=True` by default, and `from_pretrained` does not change that.
So **all 35,075,598 parameters are trainable** and all of them get updated.

So: this is **genuine, full, supervised fine-tuning** - a form of **transfer
learning**. Say exactly that in an interview; it is both true and the strongest
version of the claim.

### What "pretrained" means, concretely
`nlpaueb/legal-bert-small-uncased` was trained by the NLP group at the Athens
University of Economics and Business (Chalkidis et al., *LEGAL-BERT: The Muppets
straight out of Law School*, 2020) on **12 GB of English legal text** - EU and UK
legislation, European Court and US court cases, US contracts. Its pretraining
task was **masked language modelling**: hide ~15% of tokens and predict them.
Nobody labelled anything; the text is its own supervision. The result is a
network whose internal representations already encode legal vocabulary and
sentence structure - it "knows" that *certiorari*, *petitioner* and *remand*
belong to the same world.

**Why this base model and not `bert-base-uncased`?**
- **Domain match.** BERT-base was pretrained on Wikipedia + BooksCorpus. Legal
 English is a genuinely different register; a domain-pretrained encoder
 typically outperforms general BERT on legal tasks (this is the core finding of
 the LEGAL-BERT paper).
- **Size.** The `-small` variant is ~33% of BERT-base: 6 layers instead of 12,
 512 hidden instead of 768, 8 heads instead of 12 → 35 M parameters instead of
 110 M, roughly 4× faster.
- **Practical constraints.** A 140 MB artifact fits comfortably on a free CPU
 Hugging Face Space and trains in a Colab T4 session. A BERT-base or
 Longformer would have been slower to train and heavier to serve.

That is a defensible engineering justification. Be honest that you did not
benchmark `legal-bert-small` against `bert-base` - you chose it on domain fit
and resource budget, which is a legitimate reason.

---

## PART B - `train.py` walked through

The `# %% Cell N` markers mean this was written as a Colab notebook and exported.
It is a linear script; there are no functions to call from elsewhere.

### 1. Where the data comes from

```python
dataset = load_dataset("coastalcph/lex_glue", "scotus")
```
Downloaded at runtime from the Hugging Face Hub. **No data file is stored in this
repository.** `"scotus"` selects one of LexGLUE's seven sub-tasks.

**Split sizes (from the dataset card): 5,000 train / 1,400 validation / 1,400 test.**
The splits ship pre-made with the dataset - `train.py` does not split anything
itself, which is exactly right: the benchmark's splits are what makes results
comparable to published numbers.

### 2. What the raw data looks like
Two columns:
- `text` - a `string`: the full opinion. These are **long**; many run to several
 thousand tokens.
- `label` - an `int` from 0 to 13.

### 3. How labels are represented
As plain integers. The mapping from integer to human name is *not* in the
dataset object - `train.py` supplies it:

```python
LABEL_NAMES = ['Criminal Procedure', 'Civil Rights', ...] # 14 names
model.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
model.config.id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
```

 **This is a trust point, and you should understand exactly what it rests on.**
I checked the LexGLUE dataset card: the SCOTUS config's own `class_label` names
are literally the strings `'1'` ... `'14'` - **the dataset does not ship human-
readable class names at all.** The words "Criminal Procedure", "Civil Rights",
... come from outside the dataset: they are the *issue area* codes of the
**Supreme Court Database (SCDB)**, which LexGLUE inherited, and `train.py`
hard-codes them from that external source.

So the correctness of every label your app displays depends on a hand-typed list
matching an external codebook. Two consequences you should be able to state:
1. You **cannot** write `assert LABEL_NAMES == dataset.features['label'].names` -
 there is nothing to assert against. The provenance has to be documented instead.
2. If the order were wrong, training, accuracy and macro-F1 would be
 **completely unaffected** - the model would still learn perfectly - and every
 prediction shown to a user would carry the wrong name. It is a silent-failure
 class of bug with no test that catches it.

The order used here matches the standard SCDB `issueArea` ordering (1 = Criminal
Procedure ... 14 = Private Action), and the live model's behaviour corroborates it:
press-censorship text → "First Amendment", employment-discrimination text →
"Civil Rights", coerced-confession text → "Criminal Procedure". That is
verification by *observation*, which is real evidence but weaker than a test.

### 4. Cleaning
**There is none, and that is correct.** No lowercasing, no stopword removal, no
stemming, no punctuation stripping. Classical NLP pipelines do that because
bag-of-words models can't handle variation. Transformers must *not* have it:
the model was pretrained on natural text and needs to see natural text.
Lowercasing is handled inside the tokenizer (`do_lower_case: true`), and
subword splitting handles rare words. If asked "what preprocessing did you do?"
the strong answer is: *"Tokenization and truncation only - deliberately.
Aggressive preprocessing would create a mismatch with how the encoder was
pretrained."*

### 5. Tokenization

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
 return tokenizer(examples['text'], truncation=True, max_length=512, padding=False)

train_tok = dataset['train'].map(tokenize_fn, batched=True, remove_columns=['text'])
```

The tokenizer is a **WordPiece** tokenizer with a 30,522-token vocabulary.
Concretely - these are the *real* IDs produced by this repo's tokenizer, which
I ran against `saved_model/`:

```
"The petitioner was convicted of first-degree murder"
 → lowercase + WordPiece + special tokens
 ['[CLS]', 'the', 'petitioner', 'was', 'convicted', 'of', 'first', '-', 'degree', 'murder', '[SEP]']
 → vocabulary indices
 [ 101, 207, 1818, 246, 3276, 210, 296, 116, 2030, 4127, 102 ]
```

Two things worth noticing:
- `[CLS]` is id **101** and `[SEP]` is **102** (the BERT convention), but
 `"the"` is **207**, not 1996 as it is in `bert-base-uncased`. **LEGAL-BERT has
 its own 30,522-token vocabulary trained on legal text.** That is a second,
 often-overlooked reason it beats general BERT on legal tasks - and a concrete
 reason the tokenizer must be saved with the model.
- `petitioner`, `remand` and `habeas` are each **one** token in this vocabulary.
 In general BERT they would be split into pieces. Rarer words still get split:
 `certiorari` → `['certi', '##orari']`, where `##orari` means "attach this piece
 to the previous token". That is how a fixed 30 k vocabulary covers unlimited
 English.

- `[CLS]` is a special slot prepended to every input. Its final hidden vector is
 what the classifier reads. `[SEP]` marks the end of the sequence.

The tokenizer also produces an **attention mask**: `1` for real tokens, `0`
 for padding, so the model ignores padding.

`truncation=True, max_length=512` → **everything past the first 512 tokens of the
opinion is thrown away.** This is not a bug in the script; it is a hard limit of
BERT (`max_position_embeddings: 512`). But it *is* the single biggest modelling
limitation of the project, and you must be able to say so: a Supreme Court
opinion is often 5,000-10,000 tokens, so **the model classifies roughly the first
two pages and never sees the rest.**

`padding=False` here is a deliberate efficiency choice: instead of padding
everything to 512, padding is deferred to the collator so each *batch* is padded
only to its own longest member (**dynamic padding**):
```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

`remove_columns=['text']` drops the raw strings after tokenizing - the model
never sees text, only integer IDs.

### 6. Building the model

```python
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=14)
```

Two things happen, and understanding the split is essential:
1. The **encoder body** (embeddings + 6 transformer layers + pooler ≈
 35,068,416 params) is loaded *with the pretrained legal weights*.
2. A **brand-new classification head** - `nn.Linear(512, 14)`, i.e.
 `classifier.weight [14,512]` + `classifier.bias [14]` = **7,182 randomly
 initialised parameters** - is bolted on top. It has never seen anything.

`transformers` prints a warning about newly initialised weights at this point.
That warning is *expected and correct* for fine-tuning; people often panic about it.

**Parameter budget (measured from `model.safetensors`):**
```
total 35,075,598
 word embeddings (30522×512) 15,627,264 ← 45% of the model is the vocabulary table
 position embeddings (512×512) 262,144
 6 transformer layers ~18,900,000
 pooler (512×512) 262,656
 classifier head (14×512+14) 7,182 ← 0.02% of the model
```
So the head is a rounding error; almost all of the learning capacity is in the
pretrained body, which is precisely why fine-tuning it (rather than freezing it)
matters.

### 7. Training configuration

```python
TrainingArguments(
 num_train_epochs=3,
 per_device_train_batch_size=8,
 gradient_accumulation_steps=2, # → effective batch size 16
 learning_rate=2e-5,
 weight_decay=0.01,
 warmup_steps=100,
 eval_strategy='epoch',
 save_strategy='epoch',
 load_best_model_at_end=True,
 metric_for_best_model='f1_macro', greater_is_better=True,
 fp16=torch.cuda.is_available(),
 seed=42,
)
```

What each one means and why it's set that way:

- **`num_train_epochs=3`** - three full passes over the 5,000 training examples.
 3 is the standard BERT fine-tuning default (from the original BERT paper's
 2-4 recommendation). More epochs on 5,000 examples risks memorising them.
- **batch 8 × accumulation 2 = effective batch 16.** *Gradient accumulation*
 means: run 8 examples, compute gradients, don't update; run another 8, add
 their gradients; *then* update. It buys a larger effective batch without
 needing the GPU memory for it - necessary because 512-token sequences are
 memory-hungry on a Colab T4.
- **`learning_rate=2e-5`** - deliberately tiny. A from-scratch network uses
 ~1e-3. Fine-tuning uses 1e-5 to 5e-5 because the pretrained weights are already
 good and large steps would destroy them (**catastrophic forgetting**).
- **`weight_decay=0.01`** - L2 regularisation, pulls weights toward zero, mild
 overfitting protection.
- **`warmup_steps=100`** - the LR ramps 0 → 2e-5 over the first 100 steps, then
 decays linearly. Large early steps on a randomly-initialised head can wreck
 the pretrained body; warmup avoids that.
- **`fp16=True` on GPU** - half-precision arithmetic: ~2× faster, half the
 memory. Side effect: fp16 is **not bit-reproducible**, so identical reruns can
 differ slightly even with `seed=42`.
- **`load_best_model_at_end=True` + `metric_for_best_model='f1_macro'`** - this
 is the **most valuable line in the file**. It evaluates on the validation set
 after each epoch and, at the end, restores the checkpoint with the best
 validation macro-F1 rather than the last one. That is real, correct early-model
 selection and it is a genuine overfitting defence.
- **`seed=42`** - controls weight init, shuffling and dropout. Partial
 reproducibility; combined with fp16 and unpinned library versions, an exact
 rerun is not guaranteed.

**Optimizer and loss are not written in this file - they are `Trainer` defaults,
and you must know them anyway:**
- **Optimizer: AdamW** (Adam with decoupled weight decay), with a linear decay
 schedule after warmup.
- **Loss: cross-entropy.** `BertForSequenceClassification.forward` sees
 `problem_type="single_label_classification"` and applies
 `nn.CrossEntropyLoss()(logits, labels)` internally. You never write it.
 This is why `train.py` looks like it has no loss function - it does, it's just
 inside the model class.

### 8. Evaluation

```python
def compute_metrics(eval_pred):
 logits, labels = eval_pred
 preds = np.argmax(logits, axis=-1)
 return {'accuracy': accuracy_score(labels, preds),
 'f1_macro': f1_score(labels, preds, average='macro')}
```
Run at the end of every epoch on the **validation** set. Then, once, on the
**test** set:
```python
results = trainer.predict(test_tok)
preds = np.argmax(results.predictions, axis=-1)
acc, macro_f1, micro_f1 = ...
print(classification_report(y_test, preds, target_names=LABEL_NAMES, digits=4))
```

- **Accuracy** = fraction correct. On an imbalanced 14-class problem this is a
 weak metric - always predicting the majority class already scores well.
- **Micro-F1** - with single-label multi-class, micro-F1 is *mathematically
 identical to accuracy*. It's printed anyway; harmless, but know that it adds no
 information.
- **Macro-F1** = the F1 of each of the 14 classes, averaged **unweighted**. A
 class with 20 test examples counts as much as one with 400. This is the
 right headline metric here, and choosing it for `metric_for_best_model` was a
 good decision.

**Is there data leakage?** No leakage is *introduced by this script*: the splits
come pre-made from the benchmark, the test set is touched exactly once at the
end, and nothing is fit on test data. The one thing that would need checking to
be fully rigorous is whether the same case appears in more than one split -
that's a property of the LexGLUE release, not of this code. **Class imbalance
definitely exists** (SCOTUS issue areas are heavily skewed toward Criminal
Procedure, Civil Rights and Economic Activity; classes like Private Action and
Interstate Relations are rare). The script does **not** address it - no class
weights in the loss, no resampling. Using macro-F1 *measures* the problem
honestly but does not *fix* it. That's a fair, honest weakness to state.

**Overfitting, in this context:** with 35 M parameters and only 5,000 training
documents, the model has vastly more capacity than data. Overfitting would look
like training loss falling while validation macro-F1 stalls or drops. The
defences present are: only 3 epochs, weight decay, dropout 0.1 (from the config),
and best-checkpoint restoration. **UNKNOWN FROM THE REPOSITORY:** whether it
actually overfit - nothing recorded the per-epoch curves.

### 9. Saving

```python
trainer.save_model(SAVE_DIR) # → config.json, model.safetensors, training_args.bin
tokenizer.save_pretrained(SAVE_DIR) # → tokenizer.json, tokenizer_config.json
```

The script then writes `./metrics.json` and `./classification_report.txt` to the
project root, deliberately **outside** `SAVE_DIR`, because `saved_model/` is
gitignored. Metrics written inside it would never reach the repository, which
defeats the purpose of recording them.

An earlier version of `compute_metrics` used `evaluate.load("accuracy")` and
`evaluate.load("f1")` from the `evaluate` library. Those calls download a small
metric script from the Hugging Face Hub at training time, adding a network
dependency in the middle of a training run for numbers that scikit-learn, which
was already imported, computes identically. They were replaced with
`accuracy_score` and `f1_score`, and `evaluate` was dropped from
`requirements-train.txt`. The metric values are unchanged.

**Why save the tokenizer too?** Because the model has learned that vocabulary
index 9599 means the word-piece `petition`. A tokenizer with a different
vocabulary would map that word to a different integer, and the model would be
reading essentially random inputs - accuracy collapses to chance with *no error
message*. The tokenizer and weights are one inseparable artifact.

`trainer.save_model` also writes `model.config`, which is how the `id2label`
map set back in step 3 survives into `saved_model/config.json` and lets
`app.py` display real class names.

Then a human copies the folder from Colab into the project - the manual,
unversioned step that connects the two halves of the project.

---

## PART C - One example, end to end, with the maths

Input: `"The petitioner was convicted of first-degree murder..."`, true label
`0 = Criminal Procedure`.

**1. Text → token IDs**
```
[CLS] the petitioner was convicted of first - degree murder ... [SEP]
[101, 207, 1818, 246, 3276, 210, 296, 116, 2030, 4127, ..., 102]
```
Say this comes to 58 tokens. Shapes: `input_ids [1, 58]`, `attention_mask [1, 58]`
(all ones - no padding for a single input).

**2. Embedding layer**
Each ID indexes the `word_embeddings` table `[30522, 512]` → a 512-dim vector.
Added to it: a **position embedding** (from `[512, 512]`, because self-attention
has no inherent notion of order) and a **token-type embedding**. Result:
`[1, 58, 512]`.

**3. Six transformer layers**
Each layer does, in order: multi-head self-attention (8 heads) → residual add +
LayerNorm → feed-forward 512→2048→512 with GELU → residual add + LayerNorm.
Self-attention is the key idea: each token computes a weighted average of every
other token's vector, with weights learned from content. That is how
`"suppress"` gets to be interpreted in light of `"evidence"` and `"Fifth
Amendment"` elsewhere in the sentence. Shape is unchanged throughout: `[1, 58, 512]`.
These vectors are **contextual embeddings** - the same word gets a different
vector in a different sentence, unlike word2vec/GloVe.

**4. Pooling**
Take the vector at position 0 (the `[CLS]` slot), push it through
`pooler.dense [512,512]` + `tanh`. Result: one `[1, 512]` vector meant to
summarise the whole document.

**5. Classifier head**
```
logits = h · Wᵀ + b h: [1,512], W: [14,512], b: [14] → logits: [1,14]
```
Logits are **unbounded raw scores**, not probabilities. Real logits I measured
from this model on the First Amendment sample text are:
```
[-0.187, -1.265, 3.179, -0.418, -0.156, -0.621, 0.515,
 0.834, -0.011, 0.501, -0.715, -0.318, -1.007, -1.203]
```
Index 2 (`First Amendment`) has by far the largest value.

**6. Softmax → probabilities**
```
p_i = exp(logit_i) / Σ_j exp(logit_j)
```
Exponentiate (everything positive), divide by the sum (everything adds to 1).
For the logits above this gives `p[2] = 0.6733`, `p[7] = 0.0645`, `p[6] = 0.0469`
 - exactly what the running app returned.

**Predicted class** = `argmax(p)`. For the Criminal-Procedure sample the app
returned `p[0] = 0.9163` → index 0 → **Criminal Procedure**, matching the true label.
**"Confidence" is just `max(p)`.** Be careful in an interview: this is the
softmax output of a network trained with cross-entropy, and such networks are
typically **over-confident** - a 0.92 does not mean "right 92% of the time".
Proper calibration (temperature scaling, reliability diagrams) was not done here.

**7. Loss**
Cross-entropy with true label `y = 0` and `p_0 = 0.9163`:
```
loss = -log(p_y) = -log(0.9163) = 0.0874
```
If the model had put 0.92 on the wrong class and only 0.01 on the right one,
loss would be `-log(0.01) = 4.6` - a much bigger number, hence a much bigger
push.

**8. Backpropagation and the update**
`loss.backward()` walks the computation graph in reverse, computing
`∂loss/∂θ` for every one of the 35,075,598 parameters via the chain rule. Then
AdamW takes a step roughly of the form
```
θ ← θ − lr · (smoothed gradient) − lr · weight_decay · θ
```
with `lr ≈ 2e-5`. Every weight moves a tiny amount: the classifier head, the
pooler, the feed-forward layers, the attention matrices, and even the word
embeddings. Repeat for 5,000 examples × 3 epochs ≈ 940 optimizer steps
(5000/16 ≈ 312 steps per epoch).

**During inference none of steps 7-8 happen.** The pipeline runs steps 1-6
inside `torch.no_grad()`, no gradients are computed, and no weight ever changes.
