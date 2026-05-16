# 🏛️ US Supreme Court Opinion Classification — NLP Semester Project

## Project Overview

This project classifies **US Supreme Court opinions** into **14 legal issue areas** using four different NLP approaches, comparing their performance on the same benchmark dataset.

**🚀 Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/msumar/NLP-Semester-Project)

**Dataset:** [LexGLUE SCOTUS](https://huggingface.co/datasets/coastalcph/lex_glue) — A legal NLP benchmark from HuggingFace containing ~7,800+ Supreme Court opinions.

## 14 Issue Area Classes

| ID | Issue Area | ID | Issue Area |
|----|------------|----|------------|
| 0 | Criminal Procedure | 7 | Economic Activity |
| 1 | Civil Rights | 8 | Judicial Power |
| 2 | First Amendment | 9 | Federalism |
| 3 | Due Process | 10 | Interstate Relations |
| 4 | Privacy | 11 | Federal Taxation |
| 5 | Attorneys | 12 | Miscellaneous |
| 6 | Unions | 13 | Private Action |

## Methodology — 4 Approaches Compared

| # | Approach | Models | Feature Extraction |
|---|----------|--------|--------------------|
| 1 | **Classical ML** | Logistic Regression, SVM, Random Forest | TF-IDF (unigrams + bigrams) |
| 2 | **Deep Learning** | BiLSTM with Attention | Learned word embeddings |
| 3 | **Transformer** | DistilBERT (fine-tuned) | Contextual embeddings |
| 4 | **Domain Transformer** | LegalBERT (fine-tuned) | Legal-domain embeddings |

## Evaluation Metrics

- Accuracy
- Macro F1-Score (handles class imbalance)
- Micro F1-Score
- Per-class Precision, Recall, F1
- Confusion Matrices

## How to Run

### Google Colab (Recommended for Training)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `train.py`
3. Go to **Runtime → Change runtime type → GPU (T4)**
4. Run the script to fine-tune LegalBERT and save the model.

### Local Deployment
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the Flask app:
```bash
python app.py
```
3. Open `http://localhost:7860` in your browser.

## Computational Resources
- **Platform:** Google Colab (free tier)
- **GPU:** Tesla T4 (16GB VRAM)
- **Estimated Runtime:** ~45-60 minutes total

## Project Structure
```
├── app.py              # Flask web application
├── train.py            # Model training script (LegalBERT)
├── Dockerfile          # Container configuration for deployment
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── saved_model/        # Fine-tuned model weights (LegalBERT)
├── static/             # Frontend assets (CSS/JS)
└── templates/          # HTML templates (Flask)
```
