### **Milestone 4: Model Training**

#### **Objective**
To train an initial baseline model for toxic comment classification, experimenting with optimization and regularization techniques to achieve high performance.

---

#### **1. Model and Hyperparameters**

An initial model was trained using the following configuration:
*   **Model:** `microsoft/mdeberta-v3-base` (a powerful transformer model)
*   **Framework:** PyTorch with Hugging Face `transformers` and `accelerate` for multi-GPU training.
*   **Key Hyperparameters:**
    *   **Epochs:** 6 (with early stopping)
    *   **Batch Size:** 16 per GPU (Effective: 32)
    *   **Learning Rate:** `2e-5`
    *   **Max Sequence Length:** 256 tokens

---

#### **2. Experimentation: Optimization & Regularization**

Several techniques were implemented to improve training stability and model performance.

*   **Optimization Method:**
    *   **Optimizer:** `AdamW` with a weight decay of `0.01` was used.
    *   **Scheduler:** A linear learning rate scheduler with a warmup phase (10% of training steps) was applied to stabilize training.

*   **Handling Class Imbalance:**
    *   A **Weighted Binary Cross-Entropy Loss** was implemented. This gives more importance to rare, positive classes (like `threat` and `severe_toxic`), forcing the model to learn them better.
      ```python
      # Logic for calculating class weights
      label_sum = train_df[LABELS].sum(axis=0)
      label_freq = label_sum / len(train_df)
      class_weights = (1 / (label_freq + 1e-6))
      class_weights = class_weights / class_weights.sum() * NUM_LABELS
      class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float)
      
      # Using the weights in the loss function
      criterion = WeightedBCEWithLogitsLoss(class_weights_tensor)
      ```

*   **Regularization Techniques:**
    *   **Early Stopping:** Training was configured to stop if the validation AUC score did not improve for 2 consecutive epochs (`patience=2`). The best-performing model was saved.
    *   **Gradient Clipping:** The gradients were clipped to a maximum norm of `1.0` to prevent them from exploding and destabilizing the training process.

---

#### **3. Code Snippet: Training Setup**

The core components of the training pipeline were set up as follows:

```python
# Model Initialization
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
num_training_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)

# Multi-GPU Preparation with Accelerate
accelerator = Accelerator(mixed_precision="fp16")
train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(
    train_loader, val_loader, model, optimizer, scheduler
)
```

---

#### **4. Results**

The model training was successful, establishing a strong performance baseline.

*   **Best Performance:** The model achieved a **macro average ROC AUC of 0.9888** on the validation set.
*   **Early Stopping:** Training was halted after 5 epochs, as performance on the validation set did not improve beyond the 3rd epoch.
*   **Saved Model:** The model weights from the best epoch (Epoch 3) were saved for final inference.

**Per-Class Validation AUC Scores:**

| Category        | ROC AUC Score |
| --------------- | ------------- |
| toxic           | 0.9854        |
| severe_toxic    | 0.9915        |
| obscene         | 0.9936        |
| threat          | 0.9812        |
| insult          | 0.9890        |
| identity_hate   | 0.9840        |
| **Macro Average** | **0.9888**    |
