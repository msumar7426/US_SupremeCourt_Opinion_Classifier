# ============================================================
# train.py - Run this in Google Colab (GPU: T4)
# Fine-tunes LegalBERT on US Supreme Court Opinion Classification
# Saves the model to ./saved_model/ and metrics to ./metrics.json
# ============================================================

import collections
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import classification_report, accuracy_score, f1_score

# fp16 training is still not bit-reproducible, so reruns can differ slightly.
set_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# %% Configuration
MODEL_NAME = "nlpaueb/legal-bert-small-uncased"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
SAVE_DIR = "./saved_model"
METRICS_PATH = "./metrics.json"
REPORT_PATH = "./classification_report.txt"

# The LexGLUE SCOTUS config labels examples 0-13 and ships no human-readable
# class names. These are the Supreme Court Database issueArea codes 1-14 in
# order. A wrong order leaves every metric unchanged and every displayed label
# wrong, so it cannot be caught by evaluation.
LABEL_NAMES = [
    'Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process',
    'Privacy', 'Attorneys', 'Unions', 'Economic Activity', 'Judicial Power',
    'Federalism', 'Interstate Relations', 'Federal Taxation',
    'Miscellaneous', 'Private Action'
]
NUM_LABELS = len(LABEL_NAMES)
print(f"Config: {MODEL_NAME}, {NUM_LABELS} classes, {EPOCHS} epochs")

# %% Load Dataset
print("\nLoading SCOTUS dataset from HuggingFace...")
dataset = load_dataset("coastalcph/lex_glue", "scotus")

for split in ['train', 'validation', 'test']:
    print(f"  {split}: {len(dataset[split]):,} examples")

n_classes = getattr(dataset['train'].features['label'], 'num_classes', None)
if n_classes is not None and n_classes != NUM_LABELS:
    raise ValueError(
        f"Dataset has {n_classes} classes but LABEL_NAMES has {NUM_LABELS}"
    )

# SCOTUS is heavily imbalanced, which is why macro-F1 rather than accuracy
# selects the best checkpoint below.
print("  train class distribution:",
      dict(sorted(collections.Counter(dataset['train']['label']).items())))

# %% Tokenize
print("\nTokenizing with LegalBERT tokenizer...")
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
print("Tokenization complete")

# %% Load Model
print(f"\nLoading {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

# Written into saved_model/config.json, which is where the serving pipeline
# reads class names from.
model.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
model.config.id2label = {i: name for i, name in enumerate(LABEL_NAMES)}

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model loaded: {total_params:,} parameters, {trainable_params:,} trainable")

# %% Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }

# %% Train
print("\n" + "=" * 60)
print("TRAINING LegalBERT")
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
    save_total_limit=2,
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
print(f"\nTraining complete in {train_time/60:.1f} minutes")

# %% Evaluate on Test Set
print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

results = trainer.predict(test_tok)
preds = np.argmax(results.predictions, axis=-1)
y_test = np.array(dataset['test']['label'])

acc = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average='macro')
micro_f1 = f1_score(y_test, preds, average='micro')
report_txt = classification_report(y_test, preds, labels=list(range(NUM_LABELS)),
                                   target_names=LABEL_NAMES, digits=4)

print(f"  Accuracy:  {acc:.4f}")
print(f"  Macro F1:  {macro_f1:.4f}")
print(f"  Micro F1:  {micro_f1:.4f}")
print("\nClassification Report:")
print(report_txt)

# %% Save Model
print("\n" + "=" * 60)
print(f"SAVING MODEL TO {SAVE_DIR}/")
print("=" * 60)

trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Written outside SAVE_DIR because saved_model/ is gitignored and these are the
# only durable record of how the committed model performed.
with open(METRICS_PATH, 'w') as fh:
    json.dump({
        'base_model': MODEL_NAME,
        'dataset': 'coastalcph/lex_glue (scotus)',
        'max_length': MAX_LEN,
        'epochs': EPOCHS,
        'per_device_train_batch_size': BATCH_SIZE,
        'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
        'effective_batch_size': BATCH_SIZE * training_args.gradient_accumulation_steps,
        'learning_rate': LR,
        'seed': 42,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'test_accuracy': float(acc),
        'test_macro_f1': float(macro_f1),
        'test_micro_f1': float(micro_f1),
        'train_minutes': round(train_time / 60, 1),
    }, fh, indent=2)
with open(REPORT_PATH, 'w') as fh:
    fh.write(report_txt)

print(f"Model saved to {SAVE_DIR}/")
print(f"Metrics saved to {METRICS_PATH} and {REPORT_PATH}")
print("\nFiles saved:")
for f in sorted(os.listdir(SAVE_DIR)):
    size = os.path.getsize(os.path.join(SAVE_DIR, f))
    print(f"  {f} ({size/1024:.1f} KB)")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("  1. Download saved_model/, metrics.json and classification_report.txt")
print("  2. Place saved_model/ in your project directory")
print("  3. Commit metrics.json and classification_report.txt")
print("  4. Run app.py to serve the model")
print("=" * 60)
