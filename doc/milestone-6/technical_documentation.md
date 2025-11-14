# **CleanSpeech Milestone 6: Technical Documentation**

This document provides a comprehensive technical overview of the CleanSpeech project. It is intended for developers, researchers, and evaluators who wish to understand, reproduce, or extend this work.

## Table of Contents
1.  [Environment Setup](#1-environment-setup)
2.  [Data Pipeline](#2-data-pipeline)
3.  [Model Architecture](#3-model-architecture)
4.  [Training Summary](#4-training-summary)
5.  [Evaluation Summary](#5-evaluation-summary)
6.  [Inference Pipeline](#6-inference-pipeline)
7.  [Deployment Details](#7-deployment-details)
8.  [System Design Considerations](#8-system-design-considerations)
9.  [Error Handling & Monitoring](#9-error-handling--monitoring)
10. [Reproducibility Checklist](#10-reproducibility-checklist)

---

## 1. Environment Setup

### 1.1 Dependencies
The project is built on Python and utilizes several key libraries for deep learning, data manipulation, and web deployment.

**Core Libraries:**
*   `torch` & `transformers`: For model training and inference.
*   `accelerate`: For multi-GPU and mixed-precision training.
*   `scikit-learn`: For metrics and data splitting.
*   `pandas` & `numpy`: For data handling.
*   `streamlit`: For the interactive frontend UI.
*   `google-generativeai`: For the text rewriting module.
*   `bert-score`: For evaluating semantic similarity of rewrites.

### 1.2 Installation
A `requirements.txt` file should be created with the following content:
```
transformers
torch
accelerate
scikit-learn
pandas
numpy
streamlit
google-generativeai
bert-score
tqdm
seaborn
matplotlib
fastapi
uvicorn
```

Install all dependencies using pip:
```bash
pip install -r requirements.txt
```

### 1.3 Environment
*   **Python Version:** 3.10
*   **Hardware (Training):** 2 x NVIDIA T4 GPUs (as provided on Kaggle Notebooks).
*   **Hardware (Inference/UI):** CPU is sufficient. A GPU will accelerate inference.

---

## 2. Data Pipeline

### 2.1 Dataset Sources
The project employs Jigsaw Toxic Comment Dataset with 150,000 test and train samples for robust training and generalization.

1.  **Primary Dataset (Training & Validation):**
    *   **Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
    *   **Description:** Contains ~160,000 Wikipedia talk page comments with six binary labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
    *   **Licensing:** Competition-specific license. Please refer to the Kaggle page for terms of use.


### 2.2 Preprocessing Summary
The data pipeline is detailed in `Milestone_2.md` and implemented in `toxic_comment_classification.py`.

*   **Jigsaw Data:**
    1.  **Cleaning:** Lowercasing text; removing URLs, HTML tags, emojis, and non-ASCII characters.
    2.  **Normalization:** Collapsing multiple whitespaces into a single space.
    3.  **Deduplication:** Removing duplicate comments to prevent data leakage.
    4.  **Splitting:** An 80/20 train/validation split, stratified on the `toxic` label to ensure a balanced distribution of toxic examples in both sets.

---

## 3. Model Architecture

The core of the system is a fine-tuned transformer model for multi-label text classification.

*   **Base Model:** `microsoft/mdeberta-v3-base`
*   **Justification:**
    *   **Disentangled Attention:** DeBERTa's key innovation separates content and position encodings, leading to a superior understanding of token relationships and context, which is critical for detecting nuanced toxicity.
    *   **Multilingual:** Natively supports multiple languages, making it ideal for handling multilingual toxic classification.
    *   **Proven Performance:** Achieves state-of-the-art results on numerous NLP benchmarks.


**Figure 2:** High-level diagram of the mDeBERTa-v3 architecture.

### 3.1 Architecture Details
*   **Input:** Text is tokenized with a maximum sequence length of 256 tokens.
*   **Body:** The standard `mDeBERTa-v3-base` transformer encoder (12 layers, 768 hidden size).
*   **Output Head:** A single linear layer is added on top of the pooled output, mapping the hidden state to 6 logits (one for each toxicity label).
*   **Activation:** A `Sigmoid` function is applied to the logits during inference to produce independent probabilities for each class.

### 3.2 Key Hyperparameters
*   **Learning Rate:** `2e-5`
*   **Batch Size:** `16` (per GPU, effective batch size of 32)
*   **Max Sequence Length:** `256`
*   **Weight Decay:** `0.01`

---

## 4. Training Summary

The model was fine-tuned using the script `toxic_comment_classification.py`.

*   **Framework:** Hugging Face `Accelerate` was used for distributed, mixed-precision (`fp16`) training on two T4 GPUs.
*   **Optimizer:** `AdamW` with a decoupled weight decay of `0.01`.
*   **Scheduler:** A linear learning rate scheduler with a 10% warm-up phase was used for stable convergence.
*   **Loss Function:** **Weighted Binary Cross-Entropy with Logits**. Class weights were calculated based on the inverse frequency of each label, giving higher importance to rare classes like `threat` and `identity_hate`.
*   **Regularization:**
    *   **Early Stopping:** Training was stopped after 2 epochs of no improvement in validation AUC. The best model was saved from epoch 3.
    *   **Gradient Clipping:** Gradients were clipped at a max norm of 1.0 to prevent explosions.
*   **Training Time:** Approximately 1-2 hours on the specified hardware.
*   **Key Metric Achieved (Validation):** The model achieved a **Macro ROC-AUC of 0.989**.

---

## 5. Evaluation Summary

The final model was rigorously evaluated on the held-out test set and analyzed for performance, explainability, and rewrite quality in `explainbility+rewrite (2).py`.

*   **Quantitative Performance:**
    *   **Macro ROC-AUC:** **0.983** (demonstrating excellent generalization).
    *   **Macro F1-Score:** **0.602** (with individually tuned decision thresholds per label).
*   **Key Insight: Threshold Tuning:** A critical finding was that a single 0.5 decision threshold is suboptimal for all classes due to severe class imbalance. Tuning the threshold for each label (e.g., a low threshold of 0.1 for `threat`) significantly improved the F1-score, making the model more practical for real-world use.
*   **Rewrite Quality:** The Gemini-powered rewriting module was evaluated for semantic preservation using **BERTScore**. It achieved an average F1-score of **0.948**, confirming that the rewritten text successfully maintains the original meaning.

---

## 6. Inference Pipeline

The inference pipeline takes raw text and produces a structured JSON with probabilities and explanations.

### 6.1 Data Flow
`Raw Text` → `clean_text()` → `tokenizer()` → `model()` → `torch.sigmoid()` → `Probabilities & Explanation Scores`

### 6.2 Python Inference Snippet

```python
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "milanchndr/toxicity-classifier-mdeberta"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LEN = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
model.eval()

def clean_text(text):
    # (Same cleaning function as in training)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_toxicity(text: str):
    cleaned_text = clean_text(text)
    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
    
    return {label: float(prob) for label, prob in zip(LABELS, probs)}

# Example usage
example = "You are an absolute idiot. I hate you."
predictions = predict_toxicity(example)
print(predictions)
```

---

## 7. Deployment Details

The project's components are deployed across several platforms for accessibility and modularity.

*   **Platform:**
    *   **Backend API:** FastAPI application deployed on **Hugging Face Spaces**.
    *   **Model Weights:** Hosted on **Hugging Face Hub**.
    *   **Frontend UI:** Streamlit application designed to be run **locally**.
*   **API Endpoint:**
    *   **URL:** `https://milanchndr-Toxic-Comment-Classifier-Explainer.hf.space/predict`
    *   **Method:** `POST`
*   **Example API Request:**
    ```bash
    curl -X POST "https://milanchndr-Toxic-Comment-Classifier-Explainer.hf.space/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "You are such an idiot!"}'
    ```
*   **Running the UI Locally:**
    ```bash
    # Navigate to the project root directory
    streamlit run ui/app.py
    ```

---

## 8. System Design Considerations

*   **Modularity:** The system is intentionally designed with decoupled components (Detection, Explanation, Rewriting, UI). This allows for independent updates. For example, the `mDeBERTa-v3` classifier could be replaced with a newer model without altering the UI or the rewriting module, as long as the API contract is maintained.
*   **Scalability:**
    *   The current **Hugging Face Spaces** deployment is suitable for demos and moderate traffic.
    *   For production-level scale, the FastAPI backend could be containerized with Docker and deployed on a scalable cloud service like AWS SageMaker, Google AI Platform Endpoints, or a Kubernetes cluster.
    *   The Streamlit UI is stateful and best suited for single-user sessions or small teams. A production frontend would likely be rebuilt in a framework like React or Vue.js.
*   **Data Flow:** The linear data flow (`Input` -> `API` -> `UI`) is simple and robust. It ensures that all data processing and model logic are centralized in the backend, making the frontend a lightweight client responsible only for rendering results.

---

## 9. Error Handling & Monitoring

*   **Error Handling:**
    *   The backend API includes basic error handling. Invalid requests (e.g., malformed JSON) will return standard `422 Unprocessable Entity` errors.
    *   The Gemini API call in the rewriting module is wrapped in a `try...except` block to gracefully handle API failures or rate limits, returning the original text instead of crashing.
*   **Monitoring (Future Work):**
    *   The current system does not have a formal monitoring stack.
    *   For a production deployment, key metrics to monitor would include: API latency, error rates (HTTP 5xx), and costs (especially for the Gemini API).
    *   Model performance monitoring could be implemented by logging predictions and using tools like Evidently AI to detect data drift or concept drift over time.

---

## 10. Reproducibility Checklist

To fully reproduce the results of this project, follow these steps:

*   **✅ Environment:** Set up a Python 3.10 environment and install all packages from the `requirements.txt` file provided in [Section 1](#1-environment-setup).
*   **✅ Data:** Download the **Jigsaw** and **HASOC** datasets from their respective sources. Place them in a `data/` directory.
*   **✅ Random Seed:** The training script `toxic_comment_classification.py` uses a fixed random seed for all libraries (`seed = 42`) to ensure reproducible splits and weight initialization.
*   **✅ Training Script:** Run the training notebook/script to fine-tune the model.
    ```bash
    # (Assuming conversion to a .py script)
    python toxic_comment_classification.py
    ```
*   **✅ Model Checkpoint:** The final, trained model is available on the Hugging Face Hub at **`milanchndr/toxicity-classifier-mdeberta`**. You can download this directly to skip the training step. The best model artifacts are saved to the `best_model/` directory during training.
*   **✅ Evaluation Script:** Use the notebook `explainbility+rewrite (2).py` to run evaluation, threshold tuning, and XAI analysis.
*   **✅ API & UI:**
    1.  Place the `best_model/` directory in the same location as your FastAPI app file.
    2.  Launch the API: `uvicorn app:app --host 0.0.0.0 --port 8000`
    3.  Launch the UI: `streamlit run ui/app.py`
