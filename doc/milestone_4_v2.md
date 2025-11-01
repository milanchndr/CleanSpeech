# Milestone 4: Model Training
**CleanSpeech: Toxicity Detection & Rewriting with Explainable AI**

---

## Milestone Overview

This milestone marks the transition of **CleanSpeech** from traditional linear models to a **deep transformer-based toxicity classifier**.  
We fine-tuned **`microsoft/mdeberta-v3-base`** on the **Jigsaw Toxic Comment Classification** dataset to detect six toxicity categories —  
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

Key highlights:

- **Model:** mDeBERTa-v3-base (183 M parameters) fine-tuned end-to-end with a sigmoid multi-label head.  
- **Optimization:** AdamW (2e-5 LR, weight_decay 0.01) + linear warm-up scheduler (10 %) for stable convergence.  
- **Loss Function:** Weighted BCEWithLogitsLoss to handle class imbalance (rare labels weighted higher).  
- **Regularization:** Gradient clipping (1.0), early stopping (patience = 2), and mixed-precision (fp16) training.  
- **Hardware:** 2 × T4 GPUs via Hugging Face Accelerate for distributed training.  
- **Performance:** Macro ROC-AUC = **0.989**, with consistently high scores across all labels.  
- **Outcome:** The model surpassed the TF-IDF + Logistic Regression baseline and now serves as the **foundation for explainability and rewriting** in Milestones 5 & 6.

> _Essence:_  
> CleanSpeech now has a strong, multilingual, context-aware toxicity detector — stable, reproducible, and ready for explainable AI integration.

---
## 1. Objective

This milestone covers **initial model training and experimentation** — testing optimization, hyperparameters, and regularization strategies.

We built upon **Milestone 3** (data preprocessing + TF-IDF baseline) and fine-tuned a **transformer-based classifier (`mDeBERTa-v3-base`)** to identify six toxicity types:

`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

**Goal:** Establish a strong, well-regularized transformer baseline to outperform the earlier linear TF-IDF model and prepare for explainability (Milestone 5).

---

## 2. Dataset Details

| Split | Samples | Purpose | Notes |
|:------|:---------|:---------|:------|
| **Train** | 95 000 | Model fitting | Cleaned & de-duplicated comments |
| **Validation** | 24 000 | Hyperparameter tuning | Stratified on `toxic` label |
| **Test** | 153 000 | Final Kaggle submission | Used for evaluation only |

**Input:** Cleaned comment text  
**Labels:** 6 binary columns → multi-label setup  
**Class Imbalance:** Addressed using inverse-frequency weights in the loss.

### *Example preprocessing*
```python
import re
text = re.sub(r"http\S+|www\.\S+", " ", text.lower())
text = re.sub(r"[^\x00-\x7F]+", " ", text)   # remove non-ASCII / emojis
text = re.sub(r"\s+", " ", text).strip()
```

---
## 3. Model Architecture

| Component | Description |
|:-----------|:-------------|
| **Backbone** | `microsoft/mdeberta-v3-base` (183 M parameters) |
| **Input Length** | Up to 256 tokens |
| **Output** |6 logits. These are passed through a sigmoid function to produce independent probabilities (0 to 1) for each of the six toxicity labels. |
| **Problem Type** | `multi_label_classification` |
| **Fine-tuning** | All layers trainable + new classification head |

_Essence:_  
Multilingual DeBERTa V3 encoder fine-tuned end-to-end for six toxicity categories.

> _Why DeBERTa?_ It disentangles position vs content embeddings, improving token-level context capture — ideal for nuanced text like toxic comments.

---

## 4. Training Setup

| Item | Configuration | Why / Effect |
|:------|:---------------|:-------------|
| **Loss** | Weighted BCEWithLogitsLoss | Emphasize rare labels |
| **Optimizer** | AdamW (lr = 2e-5, weight_decay = 0.01) | Stable convergence; decoupled weight decay |
| **Scheduler** | Linear warm-up (10%) + decay | Smooth start, avoids spikes |
| **Batch Size** | 16 per GPU (≈ 32 effective) | Fits GPU RAM; balanced gradient noise |
| **Epochs** | 6 max + early stopping (patience = 2) | Prevent overfitting |
| **Gradient Clipping** | 1.0 norm | Prevent gradient explosion |
| **Mixed Precision** | fp16 (Accelerate) | Faster & memory-efficient |
| **Hardware** | 2 × T4 GPUs (Kaggle) | Multi-GPU setup |
| **Metric** | ROC-AUC (per-label + macro) | Threshold-free evaluation |

---

### _Weighted loss calculation_

```python
label_sum = train_df[LABELS].sum(axis=0)
label_freq = label_sum / len(train_df)
class_weights = (1 / (label_freq + 1e-6))
class_weights = class_weights / class_weights.sum() * len(LABELS)
criterion = WeightedBCEWithLogitsLoss(torch.tensor(class_weights.values))
```

> _Logic:_ rare labels (e.g. threat) receive higher weight → better recall on minority classes.

---

## 5. Hyperparameter Experiments

| Parameter | Values Tried | Observation |
|:------------|:--------------|:-------------|
| **Learning Rate** | 2e-5 – 3e-5 | 2e-5 was most stable |
| **Batch Size** | 8 – 16 | 16 balanced speed + gradient noise |
| **Sequence Length** | 128 vs 256 | 256 slightly better AUC |
| **Weight Decay** | 0 vs 0.01 | 0.01 reduced overfit |
| **Warm-up Ratio** | 0.05 vs 0.10 | 0.10 yielded smoother curve |

**Best Config →** `lr = 2e-5`, `batch = 16`, `max_len = 256`, `weight_decay = 0.01`

> _Takeaway:_ Longer sequences + smaller learning rate produced the most stable and generalizable results.

---

## 6. Regularization & Optimization Techniques

| Technique | Purpose | Observed Effect |
|:------------|:----------|:----------------|
| **Weight Decay (AdamW)** | Penalize large weights | Reduced variance; smoother training |
| **Early Stopping (p=2)** | Stop when val AUC plateaus | Best epoch = 3 (out of 6) |
| **Gradient Clipping (1.0)** | Avoid fp16 instability | No NaN losses |
| **Warm-up LR** | Gradual ramp-up | Prevented initial divergence |
| **Class-Weighted Loss** | Handle imbalance | Improved AUC for `threat` & `identity_hate` |

> _Essence:_ A blend of early stopping, warmup scheduling, and class weighting made training stable across GPUs and prevented overfitting.

---

## 7. Initial Training Results

| Label           | ROC-AUC | F1 (at 0.5 threshold, not optimal) |
|:----------------|:--------|:----------------------------------:|
| toxic           | 0.97    | 0.68 |
| severe_toxic    | 0.98    | 0.42 |
| obscene         | 0.98    | 0.71 |
| threat          | 0.99    | 0.54 |
| insult          | 0.97    | 0.66 |
| identity_hate   | 0.98    | 0.59 |
| **Macro Average** | **0.982** | **0.60** |


**Observations**
- Validation AUC peaked at epoch 3 → early stopping triggered correctly.  
- Training and validation losses tracked closely → no overfitting.  
- Model remained stable under mixed precision and gradient clipping.

**Qualitative Example**
> “You are a disgusting idiot.” → Example Probabilities: {'toxic': 0.98, 'severe_toxic': 0.15, 'obscene': 0.85, 'threat': 0.01, 'insult': 0.95, 'identity_hate': 0.05}

> _Summary:_ The model achieved near state-of-the-art AUC values across all classes with consistent convergence and balanced predictions.

---

## 8. Model Artifacts

| Artifact | Description |
|:-----------|:-------------|
| **`best_model/`** | Saved checkpoint (epoch 3) – Hugging Face format |
| **`tokenizer/`** | Matching DeBERTa tokenizer |
| **`toxic-comment-classification.ipynb`** | Main training notebook |
| **`submission.csv`** | Kaggle test predictions |
| **`src/models/baseline_pipeline.joblib`** | TF-IDF + Logistic Regression baseline |
| **`baseline_meta.json`** | Metadata for baseline model (UI reference) |

All runs fixed `seed = 42` for reproducibility.  
Training reproducible across multi-GPU runs via Hugging Face Accelerate.

> _Tip:_ Both the model and tokenizer directories are ready for reuse in Milestone 5 explainability and deployment.

---

## 9. **Observations / Notes for Next Milestone**

#### **Early Indications of Model Performance**

*   **Strong Predictive Power:** The model achieved an excellent macro ROC-AUC of **0.989** on the validation set. This metric is particularly important at this stage as it evaluates the quality of the model's raw probability outputs, independent of any specific decision threshold. It confirms the model is highly effective at distinguishing between toxic and non-toxic content.
*   **Stable and Efficient Training:** The training process was stable, with smooth loss curves and no signs of divergence. Early stopping correctly triggered at epoch 3, preventing overfitting and confirming our regularization strategy was effective.
*   **Successful Baseline Improvement:** This transformer-based model represents a significant performance leap over the TF-IDF + Logistic Regression baseline from Milestone 3, establishing a state-of-the-art foundation for the next project phases.

#### **Issues or Unexpected Behavior Observed During Training**

*   **Threshold-Dependence of Final Predictions:** While the underlying probabilities are strong, applying a naive default threshold of 0.5 for all labels results in suboptimal performance on precision/recall-based metrics. This is especially true for rare classes, where a lower threshold might be needed to achieve reasonable recall.
*   **Under-confidence on Ambiguous Comments:** The model sometimes assigns moderate probabilities (e.g., 0.3-0.6) to comments that are sarcastic or subtly toxic. These borderline cases are the primary source of classification errors when using a fixed threshold.

#### **Ideas for Further Tuning or Improvements to be Explored in Milestone 5**

1.  **Introduce Threshold-Dependent Metrics:** Evaluate the model using a comprehensive suite of metrics including Precision, Recall, and F1-score to understand the practical trade-offs of different decision boundaries.
2.  **Tune Per-Label Decision Thresholds:** Perform a systematic search on the validation set to find the optimal probability threshold for each toxicity label individually. The goal will be to maximize a metric like the macro F1-score, which balances precision and recall across all classes.
3.  **Perform Detailed Error Analysis:** Manually review misclassified examples from the validation set to identify patterns and common failure modes (e.g., difficulty with sarcasm, context-dependent insults). This will inform potential data augmentation or model refinement strategies.
4.  **Implement Explainability (XAI):** Begin implementing explainability techniques like SHAP or attention visualization to interpret model predictions. This is a critical step before moving to text rewriting, as it helps ensure the model is focusing on relevant toxic cues.
5.  **Integrate and Prototype:** Integrate the saved `best_model` into the Streamlit UI for live inference and begin prototyping the text rewriting workflow using the Gemini API.


