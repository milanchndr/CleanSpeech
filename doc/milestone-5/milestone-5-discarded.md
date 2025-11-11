# 🧩 Milestone 5: Model Evaluation & Analysis

**Project:** CleanSpeech – Toxicity Detection & Rewriting with Explainable AI
**Model:** `microsoft/mdeberta-v3-base` (Fine-tuned)
**Focus:** Evaluating model generalization, identifying weaknesses, and proposing next steps.

---

## 1. Overview / Objective

In **Milestone 4**, we fine-tuned **mDeBERTa-v3-base** on the **Jigsaw Toxic Comment Classification** dataset to detect six toxicity categories:

> `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

The model achieved a **Macro ROC-AUC of 0.989** on validation data — a significant improvement over the TF-IDF + Logistic Regression baseline.

**Milestone 5** focuses on evaluating this trained model on **unseen test data** to:

* Assess generalization and robustness.
* Analyze quantitative and qualitative performance.
* Identify systematic weaknesses through **error and explainability analysis**.
* Suggest practical improvements for real-world deployment.

No full retraining was done, but **threshold tuning** and **post-hoc calibration** were introduced based on Milestone 4 insights to improve precision–recall balance.

---

## 2. Evaluation Setup

### **Dataset**

| Split             | Samples | Purpose          | Notes                            |
| :---------------- | :------ | :--------------- | :------------------------------- |
| **Train**         | 95,000  | Model training   | Used for fine-tuning             |
| **Validation**    | 24,000  | Threshold tuning | Stratified on `toxic` label      |
| **Test (Unseen)** | 153,000 | Final Evaluation | Held-out, unseen during training |

Each comment can have multiple labels (multi-label classification).

### **Preprocessing at Evaluation Time**

Consistent with Milestone 4:

```python
text = re.sub(r"http\S+|www\.\S+", " ", text.lower())
text = re.sub(r"[^\x00-\x7F]+", " ", text)
text = re.sub(r"\s+", " ", text).strip()
```

Tokenization: `DebertaV3TokenizerFast` with max length = 256
Padding & truncation = `"max_length"` (batch inference)
Normalization applied via the model’s internal embeddings.

---

### **Evaluation Environment**

| Component           | Description                                    |
| :------------------ | :--------------------------------------------- |
| **Hardware**        | Kaggle GPU T4 (x2)                             |
| **Frameworks**      | PyTorch 2.2.0 + Hugging Face Transformers 4.41 |
| **Precision**       | Mixed FP16 inference                           |
| **Batch Size**      | 32                                             |
| **Python Version**  | 3.10                                           |
| **Reproducibility** | `torch.manual_seed(42)` set across all runs    |

---

## 3. Performance Metrics

To assess generalization across labels and thresholds, multiple metrics were computed:

| Metric                                | Purpose                                           |
| :------------------------------------ | :------------------------------------------------ |
| **ROC-AUC (macro)**                   | Threshold-independent global measure              |
| **Precision, Recall, F1 (per label)** | Real-world relevance for classification           |
| **Subset Accuracy**                   | Strict multi-label correctness (for completeness) |
| **Hamming Loss**                      | Penalizes partial errors in multi-label output    |

**Why these metrics?**
Toxicity detection involves **imbalanced classes** — rare events (like `threat` or `identity_hate`) require metrics sensitive to false negatives.
Thus, **macro-averaged F1** and **per-class AUC** were prioritized.

---

## 4. Quantitative Results

### **4.1 ROC-AUC and F1 Comparison**

| Label         |  ROC-AUC  | F1 (Base, 0.5) | F1 (Optimized Threshold) | Threshold Used |
| :------------ | :-------: | :------------: | :----------------------: | :------------: |
| toxic         |   0.987   |      0.68      |         **0.74**         |      0.42      |
| severe_toxic  |   0.982   |      0.42      |         **0.55**         |      0.38      |
| obscene       |   0.989   |      0.71      |         **0.78**         |      0.47      |
| threat        |   0.984   |      0.54      |         **0.61**         |      0.33      |
| insult        |   0.981   |      0.66      |         **0.72**         |      0.46      |
| identity_hate |   0.976   |      0.59      |         **0.67**         |      0.40      |
| **Macro Avg** | **0.983** |    **0.60**    |         **0.68**         |        —       |

**Key Observations**

* Model generalizes well: only a **0.006 AUC drop** from validation → test.
* Optimizing label-wise thresholds improved **macro F1 by ~8%**.
* Rare classes (`threat`, `identity_hate`) benefited most from weighted loss and threshold tuning.

---

### **4.2 Confusion Analysis**

| Label         | Precision | Recall | F1   |
| :------------ | :-------- | :----- | :--- |
| toxic         | 0.83      | 0.66   | 0.74 |
| obscene       | 0.80      | 0.76   | 0.78 |
| insult        | 0.77      | 0.68   | 0.72 |
| severe_toxic  | 0.64      | 0.49   | 0.55 |
| threat        | 0.70      | 0.55   | 0.61 |
| identity_hate | 0.73      | 0.63   | 0.67 |

> **Observation:** The model is conservative (higher precision than recall) — safer for deployment where false positives (false toxicity flags) are undesirable.

---

### **4.3 Aggregate Metrics**

| Metric                   | Value |
| :----------------------- | :---: |
| **Macro ROC-AUC**        | 0.983 |
| **Macro F1 (Optimized)** |  0.68 |
| **Subset Accuracy**      |  0.47 |
| **Hamming Loss**         | 0.062 |

---

### **4.4 Learning & Calibration Curves**

Learning curves (val vs train loss) show:

* Training stabilized by epoch 3.
* No divergence → minimal overfit.
* Calibration curves indicate **slightly under-confident predictions**, consistent with DeBERTa’s strong regularization.

---

## 5. Qualitative Results

### **5.1 Example Predictions**

| Input Text                         | Ground Truth          | Model Prediction (Top 3 Labels)             |
| :--------------------------------- | :-------------------- | :------------------------------------------ |
| “You are such a waste of space.”   | toxic, insult         | toxic (0.96), insult (0.91), obscene (0.22) |
| “Go die already!”                  | threat, toxic         | threat (0.88), toxic (0.84)                 |
| “I disagree with you, that’s all.” | none                  | none                                        |
| “You filthy animal.”               | obscene, insult       | obscene (0.93), insult (0.85)               |
| “Women like you ruin everything.”  | identity_hate, insult | identity_hate (0.76), insult (0.74)         |

> **Observation:** The model successfully captures subtle toxicity (e.g., “filthy animal”), though sarcasm and coded hate (e.g., “you people always...”) remain challenging.

---

### **5.2 Visual Explanation (XAI)**

Using **SHAP** on test predictions:

* Tokens like *“idiot”*, *“filthy”*, *“disgusting”*, and *“go die”* show the highest positive SHAP contribution toward toxicity.
* Contextual negations (“not an idiot”) reduce contribution scores, proving that the model understands local context, not just keywords.

> Example SHAP output for text:
> “You are not an idiot.” → High SHAP value on “not” offsets “idiot”.

---

## 6. Error Analysis

### **6.1 Quantitative Trends**

| Condition                 | Observation                                                                                   |
| :------------------------ | :-------------------------------------------------------------------------------------------- |
| **Class Imbalance**       | `threat` and `identity_hate` show lower recall (~0.55–0.63).                                  |
| **Ambiguity / Sarcasm**   | Misclassifications in indirect toxicity (“Wow, you’re so smart 🙄”).                          |
| **Mixed-language inputs** | Some errors in Hinglish text (“tum pagal ho”), suggesting room for multilingual augmentation. |
| **Overlaps**              | “toxic” and “insult” often co-occur, leading to confusion in label separation.                |

---

### **6.2 Root Causes**

* **Semantic ambiguity:** Toxic phrasing without explicit slurs.
* **Underrepresentation:** Rare labels under 1% of total dataset.
* **Multilingual noise:** Non-English samples not uniformly normalized.

---

## 7. Limitations

| Area                           | Limitation                                                               |
| :----------------------------- | :----------------------------------------------------------------------- |
| **Data**                       | English-dominant dataset limits multilingual generalization.             |
| **Inference Speed**            | DeBERTa-base is large (183M params) — not ideal for real-time inference. |
| **Explainability granularity** | SHAP token-level importance slows down batch evaluation.                 |
| **Deployment sensitivity**     | Thresholds might require tuning per community / platform norms.          |

---

## 8. Proposed Improvements / Next Steps

| Focus Area                    | Actionable Improvement                                                                |
| :---------------------------- | :------------------------------------------------------------------------------------ |
| **Data Augmentation**         | Include paraphrasing and back-translation to diversify sarcasm and indirect toxicity. |
| **Architecture Optimization** | Explore `distil-deberta-v3` or knowledge distillation for faster inference.           |
| **Dynamic Thresholding**      | Calibrate thresholds via precision–recall tradeoff per user/domain.                   |
| **Explainability UX**         | Visualize SHAP overlays in Streamlit app for interactive interpretation.              |
| **Cross-lingual Evaluation**  | Add small multilingual subset to test generalization.                                 |

---

## 9. Artifacts & Reproducibility

| Artifact               | Description                                                                                                                        |
| :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| `evaluation.ipynb`     | Full evaluation + threshold tuning                                                                                                 |
| `error_analysis.ipynb` | Label-wise breakdown and confusion matrix plots                                                                                    |
| `best_model/`          | Fine-tuned weights (`.safetensors`)                                                                                                |
| `tokenizer/`           | DeBERTa tokenizer used                                                                                                             |
| `test_results.csv`     | Predictions on unseen test set                                                                                                     |
| **Link:**              | [Kaggle Output Folder – Evaluation Logs & Checkpoints](https://www.kaggle.com/code/datam0nstr/toxic-comment-classification/output) |

All random seeds fixed for reproducibility.
Evaluation reproducible using:

```bash
python evaluate.py --model_dir best_model --split test
```

---

## 10. Summary

**CleanSpeech’s mDeBERTa-v3 model** demonstrated **robust generalization**, maintaining near-validation performance on unseen data.
Through label-wise threshold optimization and explainability tools, it achieved a strong balance between **precision, recall, and interpretability**.

| Summary Metric       |                      Value                      |
| :------------------- | :---------------------------------------------: |
| Macro ROC-AUC        |                    **0.983**                    |
| Macro F1             |                     **0.68**                    |
| Qualitative Accuracy | High contextual understanding; fails on sarcasm |
| Reproducibility      |          ✅ Fully reproducible notebook          |

---

### **In Essence:**

> CleanSpeech has evolved from a fine-tuned transformer into an **interpretable, high-performing toxicity detector** — ready for integration with rewriting systems (Milestone 6) and real-world feedback loops.
