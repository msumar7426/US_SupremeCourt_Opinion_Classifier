# 🏛️ US Supreme Court Opinion Classifier

An AI-powered application that classifies **US Supreme Court opinions** into **14 legal issue areas** using a fine-tuned **LegalBERT** transformer model.

**🚀 Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/msumar/NLP-Semester-Project)

---

## ⚖️ How it Works
The model is fine-tuned on the **LexGLUE SCOTUS** dataset, a benchmark for legal NLP. It analyzes the text of a court opinion and predicts which of the 14 legal issue areas it belongs to (e.g., Criminal Procedure, Civil Rights, First Amendment).

### 14 Issue Areas
- Criminal Procedure
- Civil Rights
- First Amendment
- Due Process
- Privacy
- Attorneys
- Unions
- Economic Activity
- Judicial Power
- Federalism
- Interstate Relations
- Federal Taxation
- Miscellaneous
- Private Action

---

## 🚀 How to Run

### 1. Model Training (Google Colab)
If you want to re-train the model:
1. Upload `train.py` to [Google Colab](https://colab.research.google.com/).
2. Set Runtime to **GPU (T4)**.
3. Run the script to save the model to `./saved_model`.

### 2. Local Deployment (Flask)
To run the web app locally:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   python app.py
   ```
3. Visit `http://localhost:7860` in your browser.

---

## 📂 Project Structure
- `app.py`: Flask backend for inference.
- `train.py`: Training script for LegalBERT.
- `saved_model/`: Directory containing fine-tuned model weights.
- `static/` & `templates/`: Frontend files (HTML/CSS/JS).
- `Dockerfile`: Configuration for containerized deployment.
