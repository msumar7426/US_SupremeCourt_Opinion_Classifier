from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline


app = Flask(__name__)

# ── Load Model ──────────────────────────────────────────────
MODEL_PATH = "./saved_model"

LABEL_NAMES = [
    'Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process',
    'Privacy', 'Attorneys', 'Unions', 'Economic Activity', 'Judicial Power',
    'Federalism', 'Interstate Relations', 'Federal Taxation',
    'Miscellaneous', 'Private Action'
]

print("⏳ Loading model into Flask...")
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1,
    top_k=5,
)
print("✅ Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
            
        # Get predictions
        results = classifier(text[:512])[0]
        
        formatted_results = []
        for r in results:
            label = r['label']
            # Handle LABEL_X format
            if label.startswith('LABEL_'):
                idx = int(label.split('_')[-1])
                label = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else label
            
            formatted_results.append({
                'label': label,
                'score': float(r['score'])
            })
            
        return jsonify({'results': formatted_results})
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=7860)
