# DATA_FLOW.md
### One real input, traced through every layer of this system

I actually ran this request against the running app. Every number below is
measured, not illustrative.

**The input** - sample #2 from `static/js/main.js` (254 characters):

> "The state legislature enacted a statute prohibiting the publication of certain
> political advertisements within 30 days of a general election. The plaintiff
> newspaper argues this constitutes a prior restraint on speech in violation of
> the First Amendment."

---

## Step 0 - Before any request: the server has already booted

When gunicorn imports `app.py`, module-level code runs **once**:

```python
classifier = pipeline("text-classification", model="./saved_model",
 tokenizer="./saved_model",
 device=0 if torch.cuda.is_available() else -1, top_k=5)
```

At this moment 105 weight tensors (35,075,598 float32 numbers, 140 MB) are read
from `saved_model/model.safetensors` into RAM and stay there for the container's
whole life. **The model is loaded once per process, not once per request.** That
is the single most important performance property of this design, and it is
correct.

---

## Step 1 - Browser: the click

The user pastes text into `<textarea id="inputText">` (`templates/index.html`)
and clicks `<button id="classifyBtn">`.

`static/js/main.js` line ~48:
```javascript
classifyBtn.addEventListener('click', async () => {
 const text = textarea.value.trim();
 if (!text) { showError('Please enter some text to classify.'); return; }
 classifyBtn.disabled = true; // prevent double-submit
 loading.classList.add('active'); // show the spinner
```
There is a **client-side empty check** here, and a second one on the server.
Client checks are a UX convenience only - anyone can `curl` the endpoint
directly - so the server-side check is the one that matters.

---

## Step 2 - The HTTP request

```javascript
const response = await fetch('/predict', {
 method: 'POST',
 headers: { 'Content-Type': 'application/json' },
 body: JSON.stringify({ text })
});
```

On the wire:

```http
POST /predict HTTP/1.1
Host: localhost:7860
Content-Type: application/json
Content-Length: 268

{"text":"The state legislature enacted a statute prohibiting the publication of certain political advertisements within 30 days of a general election. The plaintiff newspaper argues this constitutes a prior restraint on speech in violation of the First Amendment."}
```

Why `POST` and not `GET`: the payload is a document, potentially thousands of
characters. `GET` would have to put it in the URL, which browsers and proxies
cap around 2-8 KB, and it would end up in server logs. `POST` puts it in the
body. It is also semantically a "process this", not a "fetch a resource".

The `Content-Type: application/json` header is **required** - `request.get_json()`
refuses to parse without it. (I verified: sending `text/plain` currently produces
a 500, which is a bug - it should be a 415. See `PROJECT_AUDIT.md`.)

---

## Step 3 - Flask routing

Werkzeug matches the path `/predict` and the method `POST` against the URL map
built by the decorators. `@app.route('/')` is `GET`-only and does not match;
`@app.route('/predict', methods=['POST'])` does. Flask calls `predict()`.

If you sent `GET /predict`, Flask would return **405 Method Not Allowed**
without ever touching your function.

---

## Step 4 - Inside `predict()`

```python
data = request.get_json()
```
`request` is a **thread-local proxy**: within this request's execution, it
resolves to this request's object. `get_json()` returns
`{'text': 'The state legislature enacted...'}`.

```python
text = data.get('text', '')
if not text or not text.strip():
 return jsonify({'error': 'No text provided'}), 400
```
`.get(...)` with a default avoids a `KeyError` if the key is missing. The
`.strip()` check catches whitespace-only input. The tuple `(response, 400)` is
Flask's way of setting the status code. **Verified:** `{"text":" "}` → `400`,
`{}` → `400`. That part is correct.

```python
results = classifier(text, truncation=True, max_length=MAX_TOKENS)[0]
```

### The one line you must understand and be honest about

This line used to read `classifier(text[:512])`. `text[:512]` is **Python string
slicing, 512 CHARACTERS.** The model's limit is **512 TOKENS.**

I measured it: **512 characters of this kind of legal prose is about 108
tokens.** So the app fed the model roughly **21% of the context it was trained
to use**, and silently discarded the rest. On a real 40,000-character opinion,
the model saw the first two sentences.

Why it never crashed: because 512 characters can never exceed 512 tokens, the
slice accidentally guaranteed the model's hard limit was never breached. It was
a *wrong fix for a real problem*. It was a **training/inference mismatch**:
`train.py` trained on 512-token windows, `app.py` served about 108-token
windows. Exactly the class of bug an interviewer hopes you will find in your own
project.

`MAX_TOKENS` is not hardcoded; it is read once at startup from
`classifier.tokenizer.model_max_length`, so serving cannot drift from whatever
the tokenizer was saved with.

### What the pipeline call actually does, in order

`classifier` is a `TextClassificationPipeline`. Calling it runs:

**4a. Tokenize** - `self.tokenizer(text, return_tensors='pt')`:
```
['[CLS]','the','state','legislature','enact','##ed','a','statute','prohibiting',
 'the','publication','of','certain','political','advertisement','##s', ...,'[SEP]']

input_ids = [101, 207, 264, 3666, 4042, 252, 145, 1119, 5076, 207, 1279,
 210, 411, 1897, 6083, 189, ..., 102] shape (1, 43)
attention_mask = [1, 1, 1, ... 1] shape (1, 43)
```
43 tokens, mask all ones (a single input needs no padding).
Note `enacted → ['enact','##ed']` and `advertisements → ['advertisement','##s']`:
WordPiece at work.

**4b. Forward pass** - inside `torch.no_grad()` (no gradient graph is built,
which roughly halves memory and speeds things up):
```
input_ids (1,43) --embeddings--> (1,43,512)
 --6 transformer layers--> (1,43,512)
 --take position 0 ([CLS]) --> (1,512)
 --pooler dense+tanh--> (1,512)
 --classifier Linear(512→14)--> logits (1,14)
```
Measured logits:
```
[-0.187, -1.265, 3.179, -0.418, -0.156, -0.621, 0.515,
 0.834, -0.011, 0.501, -0.715, -0.318, -1.007, -1.203]
```

**4c. Softmax** - because `config.problem_type == "single_label_classification"`,
the pipeline applies softmax (not sigmoid) across the 14 logits:
```
p_i = exp(logit_i) / Σ exp(logit_j) → sums to exactly 1.0
```

**4d. Label lookup + sort + top-k** - the pipeline reads `id2label` from
`saved_model/config.json`, so index 2 becomes the string `"First Amendment"`.
It sorts descending and keeps 5 (`top_k=5`).

**4e. Return shape.** The pipeline returns a list **per input string**, so:
```python
[[{'label': 'First Amendment', 'score': 0.6733},
 {'label': 'Economic Activity', 'score': 0.0645},
 {'label': 'Unions', 'score': 0.0469},
 {'label': 'Federalism', 'score': 0.0462},
 {'label': 'Judicial Power', 'score': 0.0277}]]
```
The trailing `[0]` unwraps the outer list. If you passed a list of 10 texts,
you would get 10 inner lists, and this line would silently return only the
first one's results.

---

## Step 5 - Formatting the response

```python
formatted_results = [
    {'label': r['label'], 'score': float(r['score'])} for r in results
]
return jsonify({'results': formatted_results})
```

- `float(r['score'])` is **necessary**: the score is a `numpy.float32`, and
  Python's `json` encoder raises `TypeError: Object of type float32 is not JSON
  serializable`. Not decoration.
- `jsonify` serialises and sets `Content-Type: application/json`, status 200.
- The label strings come straight from `id2label` in `saved_model/config.json`.
  An earlier version of this function carried a `if label.startswith('LABEL_')`
  fallback that remapped indices through a local `LABEL_NAMES` list. That branch
  could never execute, because the config does contain `id2label`. It has been
  replaced by a startup check: if the config's `id2label` does not match
  `LABEL_NAMES` exactly, the app refuses to start. A wrong label order is
  otherwise invisible, because it leaves every metric unchanged.

## Step 6 - The response on the wire

```http
HTTP/1.1 200 OK
Content-Type: application/json

{"results":[{"label":"First Amendment","score":0.6732975244522095},
 {"label":"Economic Activity","score":0.06450720876455307},
 {"label":"Unions","score":0.0469125434756279},
 {"label":"Federalism","score":0.046222344040870667},
 {"label":"Judicial Power","score":0.027706580236554146}]}
```

Note the five scores sum to ~0.859, **not 1.0** - because they're the top 5 of
14. The other 9 classes hold the remaining ~0.141.

---

## Step 7 - Back in the browser

```javascript
const data = await response.json();
if (data.error) { showError(data.error); } else { displayResults(data.results); }
```

 This checks `data.error`, **not `response.ok`**. On a 500 the body happens to
contain `error`, so it works - but a 502 from a proxy with an HTML body would
fall into `displayResults(undefined)` and throw. Minor fragility worth knowing.

`displayResults(preds)`:
```javascript
const top = preds[0];
topPrediction.innerHTML = `... ${top.label} ... Confidence: ${(top.score*100).toFixed(1)}% ...`;
predictionsList.innerHTML = preds.map(p => `
 <div class="prediction-item">
 <div>${p.label}</div>
 <div class="pred-bar-bg">
 <div class="pred-bar" style="width:0%" data-width="${(p.score*100).toFixed(1)}%"></div>
 </div>
 <div>${(p.score*100).toFixed(0)}%</div>
 </div>`).join('');
results.classList.add('active');
setTimeout(() => { document.querySelectorAll('.pred-bar')
 .forEach(bar => bar.style.width = bar.getAttribute('data-width')); }, 50);
```
Bars are rendered at `width: 0%` and then set to the real width 50 ms later -
that one-frame delay is what makes the CSS `transition` in `style.css` animate
instead of jumping.

**Final rendering:**
```
+--------------------------------------------------+
| Classification Results                           |
|                                                  |
| First Amendment                                  |
| Confidence: 67.3%                                |
|                                                  |
| First Amendment    [##############....]   67%    |
| Economic Activity  [#.................]    6%    |
| Unions             [#.................]    5%    |
| Federalism         [#.................]    5%    |
| Judicial Power     [..................]    3%    |
+--------------------------------------------------+
```

---

## The whole trace on one line

```
textarea.value
 → JSON.stringify({text})
 → POST /predict (application/json)
 → Flask URL map → predict()
 → request.get_json(silent=True)['text']
 → truncation=True, max_length=512 (tokens)
 → BertTokenizerFast → input_ids (1,43) + attention_mask (1,43)
 → embeddings (1,43,512)
 → 6 × transformer layer (1,43,512)
 → [CLS] vector (1,512) → pooler+tanh (1,512)
 → Linear(512→14) → logits (1,14)
 → softmax → 14 probabilities
 → sort, top_k=5, id2label
 → [{'label':'First Amendment','score':0.6733}, ...]
 → float() cast → jsonify → 200 OK
 → await response.json()
 → displayResults() → innerHTML → animated bars
```

## Sanity check: what does it do with nonsense?

I sent `"pizza pizza pizza"`. Output: `Economic Activity 0.32, Federalism 0.12,
Judicial Power 0.11 ...`. **The model has no "none of the above" option** - softmax
always sums to 1, so it must distribute probability across the 14 legal
categories no matter what you feed it. The lower peak (0.32 vs 0.92) is a weak
signal of uncertainty, but there is no out-of-distribution detection and no
confidence threshold in the UI. That is an honest limitation to raise yourself
in an interview.
