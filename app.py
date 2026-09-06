import os

from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model")

LABEL_NAMES = [
    'Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process',
    'Privacy', 'Attorneys', 'Unions', 'Economic Activity', 'Judicial Power',
    'Federalism', 'Interstate Relations', 'Federal Taxation',
    'Miscellaneous', 'Private Action'
]

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

if not os.path.isdir(MODEL_PATH):
    raise SystemExit(
        f"Model directory not found: {MODEL_PATH}\n"
        "saved_model/ is excluded from git because model.safetensors exceeds "
        "GitHub's 100 MB limit. See README.md for how to obtain it."
    )

print("Loading model...")
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1,
    top_k=5,
)

# A label-order mismatch would leave metrics untouched while every displayed
# class name is wrong, so fail loudly at startup instead.
_id2label = {int(k): v for k, v in classifier.model.config.id2label.items()}
_loaded_labels = [_id2label[i] for i in sorted(_id2label)]
if _loaded_labels != LABEL_NAMES:
    raise SystemExit(
        "Label mismatch between saved_model/config.json and app.py LABEL_NAMES:\n"
        f"  config.json: {_loaded_labels}\n"
        f"  app.py:      {LABEL_NAMES}"
    )

MAX_TOKENS = classifier.tokenizer.model_max_length
print(f"Model loaded: {len(LABEL_NAMES)} classes, {MAX_TOKENS} token limit")


@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(413)
def handle_http_error(error):
    return jsonify({'error': error.name}), error.code


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # silent=True turns a wrong Content-Type into a 415 instead of a 500.
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be JSON: {"text": "..."}'}), 415

    text = data.get('text', '')
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Truncation is by token, not character: 512 characters of legal prose is
        # only about 108 tokens, which would discard most of the model's context.
        results = classifier(text, truncation=True, max_length=MAX_TOKENS)[0]
        formatted_results = [
            {'label': r['label'], 'score': float(r['score'])} for r in results
        ]
        return jsonify({'results': formatted_results})
    except Exception:
        app.logger.exception("Error during prediction")
        return jsonify({'error': 'Internal error during prediction'}), 500


if __name__ == '__main__':
    # Debug mode enables the Werkzeug debugger, which allows remote code
    # execution if the port is reachable, and reloads the app, which loads the
    # 140 MB model twice.
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', port=7860)
