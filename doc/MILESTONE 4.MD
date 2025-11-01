Of course. I have restructured your detailed report to perfectly match the requested milestone format. I've integrated all your content, tables, and code snippets into the required sections, ensuring the document is self-contained and clear.

Here is the revised report:

---

### **Milestone 4: Model Training [October 31]**

Train initial models. Experiment with hyperparameters, optimization methods, and regularization techniques.

### **Overview / Objective**

This milestone covers the initial training and experimentation phase for the **CleanSpeech** toxicity classifier. We transitioned from the traditional TF-IDF + Logistic Regression baseline established in Milestone 3 to a state-of-the-art transformer-based model.

The primary objective was to fine-tune **`microsoft/mdeberta-v3-base`** on the Jigsaw Toxic Comment Classification dataset to accurately detect six types of toxicity: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. The goal was to establish a strong, well-regularized deep learning baseline that outperforms the previous linear model and serves as a robust foundation for the explainability and text rewriting components planned for Milestones 5 and 6.

### **Dataset Details**

The Jigsaw Toxic Comment Classification dataset was used for all training and evaluation tasks. The data was split as follows:

| Split | Samples | Purpose | Notes |
|:---|:---|:---|:---|
| **Train** | ~95,000 | Model fitting | Cleaned & de-duplicated comments |
| **Validation** | ~24,000 | Hyperparameter tuning & early stopping | Stratified on the `toxic` label |
| **Test** | ~153,000 | Final Kaggle submission | Used for final evaluation only |

**Data Preprocessing:**
Input comment text was cleaned to remove noise and normalize the content before tokenization.

*Example Preprocessing Steps:*
```python
import re
# Convert to lowercase and remove URLs
text = re.sub(r"http\S+|www\.\S+", " ", text.lower())
# Remove non-ASCII characters and emojis
text = re.sub(r"[^\x00-\x7F]+", " ", text)
# Normalize whitespace
text = re.sub(r"\s+", " ", text).strip()
```
**Handling Imbalanced Data:**
The dataset exhibits significant class imbalance, with labels like `threat` being extremely rare. This was addressed by implementing a weighted loss function during training to give higher importance to minority classes.

### **Model Architecture**

We validated our choice from Milestone 3 by fine-tuning a pre-trained transformer model. The architecture is summarized below:

| Component | Description |
|:---|:---|
| **Backbone** | `microsoft/mdeberta-v3-base` (183M parameters) |
| **Input Shape** | Up to 256 tokens per comment |
| **Output Shape** | 6 logits (one for each toxicity label) |
| **Architecture Choice**| Transformer-based encoder |
| **Fine-tuning** | All layers were made trainable, with a new classification head initialized for the multi-label task. |

**Rationale for DeBERTa-V3:** We chose DeBERTa for its architecture, which disentangles content and position embeddings. This design allows for a more nuanced understanding of token-level context, which is ideal for interpreting the subtle and often contextual nature of toxic comments.

### **Training Setup**

| Item | Configuration |
|:---|:---|
| **Loss Function** | Weighted BCEWithLogitsLoss |
| **Evaluation Metric** | ROC-AUC (per-label and macro-averaged) |
| **Optimizer** | AdamW (lr = 2e-5, weight_decay = 0.01) |
| **Learning Rate Schedule**| Linear warm-up (10% of steps) followed by linear decay |
| **Batch Size** | 16 per GPU (effective batch size of 32) |
| **Epochs** | 6 maximum, with early stopping (patience = 2) |
| **Hardware Used** | 2 × NVIDIA T4 GPUs on Kaggle |
| **Specific Strategies**| Gradient clipping (max norm = 1.0), mixed-precision (fp16) training via Hugging Face Accelerate. |

*Weighted Loss Calculation:*
To counteract class imbalance, weights were calculated based on the inverse frequency of each label in the training set.
```python
label_sum = train_df[LABELS].sum(axis=0)
label_freq = label_sum / len(train_df)
class_weights = (1 / (label_freq + 1e-6))
class_weights = class_weights / class_weights.sum() * len(LABELS)
criterion = WeightedBCEWithLogitsLoss(torch.tensor(class_weights.values))
```

### **Hyperparameter Experiments**

Several key hyperparameters were tuned to find the optimal configuration for stability and performance.

| Parameter | Values Tried | Observation & Best Choice |
|:---|:---|:---|
| **Learning Rate** | `2e-5`, `3e-5` | `2e-5` was the most stable and produced the best results. |
| **Batch Size** | 8, 16 (per GPU) | `16` offered a good balance between training speed and gradient stability. |
| **Sequence Length**| 128, 256 | `256` tokens captured more context, leading to a slight but consistent improvement in AUC. |
| **Weight Decay** | 0, 0.01 | `0.01` helped regularize the model and reduce overfitting. |
| **Warm-up Ratio**| 0.05, 0.10 | `0.10` yielded a smoother training curve at the start. |

The final configuration used was: `lr = 2e-5`, `batch_size = 16`, `max_len = 256`, `weight_decay = 0.01`.

### **Regularization & Optimization Techniques**

A combination of techniques was used to ensure stable training and good generalization.

| Technique | Purpose | Observed Effect |
|:---|:---|:---|
| **Weight Decay (AdamW)** | Penalize large model weights to prevent overfitting. | Reduced variance between training and validation loss. |
| **Early Stopping (patience=2)**| Stop training when validation AUC stops improving. | Prevented overfitting; training stopped at epoch 3. |
| **Gradient Clipping (norm=1.0)** | Prevent exploding gradients, especially with fp16. | Ensured stable training with no `NaN` loss values. |
| **Warm-up LR Scheduler** | Gradually increase the learning rate at the start. | Prevented initial divergence and large, unstable updates. |
| **Class-Weighted Loss** | Handle extreme class imbalance. | Significantly improved AUC for rare classes like `threat` and `identity_hate`. |

### **Initial Training Results**

The model achieved a strong macro-average ROC-AUC score of **0.989** on the validation set.

| Label | Validation ROC-AUC |
|:---|:---|
| toxic | 0.985 |
| severe_toxic | 0.992 |
| obscene | 0.994 |
| threat | 0.981 |
| insult | 0.989 |
| identity_hate | 0.984 |
| **Macro Average** | **0.989** |

**Observed Behavior:**
*   **Convergence:** The validation AUC peaked at the 3rd epoch, and early stopping correctly terminated the training run, saving computational resources and selecting the best-performing model.
*   **Overfitting:** Training and validation loss curves tracked each other closely, indicating that the regularization techniques were effective in preventing overfitting.
*   **Qualitative Example:** The model correctly identified multiple labels in complex comments.
    > *Input:* “You are a disgusting idiot.”
    > *Predicted Labels:* **toxic + insult**

### **Model Artifacts**

All artifacts are versioned and stored for reproducibility and use in subsequent milestones.

| Artifact | Description |
|:---|:---|
| **`best_model/`** | Saved Hugging Face model checkpoint from the best epoch (epoch 3). |
| **`tokenizer/`** | The corresponding `mDeBERTa-v3-base` tokenizer files. |
| **`toxic-comment-classification.ipynb`** | The main Jupyter/Colab notebook used for training and experimentation. |
| **`submission.csv`** | Generated predictions for the Kaggle test set. |

*Reproducibility:* The training process is fully reproducible using `seed = 42`. The use of Hugging Face Accelerate ensures consistent results across multi-GPU setups.

### **Observations / Notes for Next Milestone**

**Early indications of model performance:**
*   The fine-tuned mDeBERTa-v3 model is highly effective, achieving near state-of-the-art performance on this task.
*   The combination of a weighted loss function and a warm-up scheduler proved crucial for stable convergence and balanced performance across all labels.

**Issues or unexpected behavior observed during training:**
*   The model can be slightly under-confident on comments that are borderline toxic.
*   A single prediction threshold (e.g., 0.5) is not optimal for all labels due to the class imbalance. This affects precision/recall metrics but not the threshold-independent AUC score.

**Ideas for further tuning or improvements to be explored in Milestone 5 (Evaluation & Explainability):**
1.  **Threshold-based Metrics:** Evaluate the model using Precision, Recall, and F1-score.
2.  **Threshold Tuning:** Optimize per-label decision thresholds on the validation set to maximize the macro F1-score.
3.  **Explainability:** Implement SHAP or analyze attention heatmaps to understand model predictions and build user trust.
4.  **Integration:** Integrate the fine-tuned model into the Streamlit UI for live inference.
5.  **Text Rewriting:** Begin prototyping the text rewriting module using the Gemini API, which will use the model's outputs as a trigger.
