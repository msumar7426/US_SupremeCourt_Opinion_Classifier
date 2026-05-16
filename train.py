# ============================================================
# train.py — Run this in Google Colab (GPU: T4)
# Fine-tunes LegalBERT on US Supreme Court Opinion Classification
# Saves model to ./saved_models/
# ============================================================



# %% Cell 2: Imports
import numpy as np
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate as hf_evaluate
from sklearn.metrics import classification_report, accuracy_score, f1_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {DEVICE}")

# %% Cell 3: Configuration
MODEL_NAME = "nlpaueb/legal-bert-small-uncased"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
SAVE_DIR = "./saved_model"

LABEL_NAMES = [
    'Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process',
    'Privacy', 'Attorneys', 'Unions', 'Economic Activity', 'Judicial Power',
    'Federalism', 'Interstate Relations', 'Federal Taxation',
    'Miscellaneous', 'Private Action'
]
NUM_LABELS = len(LABEL_NAMES)
print(f"✅ Config: {MODEL_NAME}, {NUM_LABELS} classes, {EPOCHS} epochs")

# %% Cell 4: Load Dataset
print("\n⏳ Loading SCOTUS dataset from HuggingFace...")
dataset = load_dataset("coastalcph/lex_glue", "scotus")

for split in ['train', 'validation', 'test']:
    print(f"  {split}: {len(dataset[split]):,} examples")

# %% Cell 5: Tokenize
print("\n⏳ Tokenizing with LegalBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True,
                     max_length=MAX_LEN, padding=False)

train_tok = dataset['train'].map(tokenize_fn, batched=True, remove_columns=['text'])
val_tok = dataset['validation'].map(tokenize_fn, batched=True, remove_columns=['text'])
test_tok = dataset['test'].map(tokenize_fn, batched=True, remove_columns=['text'])

train_tok.set_format('torch')
val_tok.set_format('torch')
test_tok.set_format('torch')
print("✅ Tokenization complete")

# %% Cell 6: Load Model
print(f"\n⏳ Loading {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

# Set label mappings in model config (needed for inference pipeline later)
model.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
model.config.id2label = {i: name for i, name in enumerate(LABEL_NAMES)}

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded: {total_params:,} parameters")

# %% Cell 7: Define Metrics
metric_acc = hf_evaluate.load("accuracy")
metric_f1 = hf_evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)['accuracy']
    f1 = metric_f1.compute(predictions=preds, references=labels, average='macro')['f1']
    return {'accuracy': acc, 'f1_macro': f1}

# %% Cell 8: Train
print("\n" + "=" * 60)
print("🚀 TRAINING LegalBERT")
print("=" * 60)

training_args = TrainingArguments(
    output_dir='./checkpoints',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to='none',
    seed=42,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

start = time.time()
trainer.train()
train_time = time.time() - start
print(f"\n✅ Training complete in {train_time/60:.1f} minutes")

# %% Cell 9: Evaluate on Test Set
print("\n" + "=" * 60)
print("📊 TEST SET EVALUATION")
print("=" * 60)

results = trainer.predict(test_tok)
preds = np.argmax(results.predictions, axis=-1)
y_test = np.array(dataset['test']['label'])

acc = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average='macro')
micro_f1 = f1_score(y_test, preds, average='micro')

print(f"  Accuracy:  {acc:.4f}")
print(f"  Macro F1:  {macro_f1:.4f}")
print(f"  Micro F1:  {micro_f1:.4f}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, preds,
                           labels=list(range(NUM_LABELS)),
                           target_names=LABEL_NAMES, digits=4))

# %% Cell 10: Save Model
print("\n" + "=" * 60)
print(f"💾 SAVING MODEL TO {SAVE_DIR}/")
print("=" * 60)

trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Model saved to {SAVE_DIR}/")
print("\n📁 Files saved:")
import os
for f in sorted(os.listdir(SAVE_DIR)):
    size = os.path.getsize(os.path.join(SAVE_DIR, f))
    print(f"  • {f} ({size/1024:.1f} KB)")

print("\n" + "=" * 60)
print("📦 NEXT STEPS:")
print("  1. Download the saved_model/ folder from Colab")
print("  2. Place it in your project directory")
print("  3. Run app.py to serve the model")
print("=" * 60)
