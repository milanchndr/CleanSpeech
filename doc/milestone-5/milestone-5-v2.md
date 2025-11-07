# 🧩 Milestone 5: Model Evaluation & Analysis

**Project:** CleanSpeech – Toxicity Detection & Rewriting with Explainable AI  
**Model:** `bert-base-uncased` (Fine-tuned)  
**Focus:** Evaluating model generalization, interpretability, and constructive rewriting effectiveness.

---

## 1. Overview / Objective

In **Milestone 4**, we fine-tuned a transformer (`bert-base-uncased`) on the **Jigsaw Toxic Comment Classification** dataset to classify six categories:

> `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

**Milestone 5** extends this work by:
- Evaluating the fine-tuned model on unseen text.
- Analyzing performance metrics and interpretability.
- Conducting **error analysis** and **negation tests**.
- Assessing the **Gemini-based rewriting** system using `BERTScore`.

---

## 2. Evaluation Setup

| Component | Description |
|:--|:--|
| **Base Model** | `bert-base-uncased` |
| **Dataset** | Jigsaw Toxic Comment (preprocessed, multi-label) |
| **Split** | 80/20 train–test |
| **Frameworks** | PyTorch + Hugging Face Transformers |
| **Metrics** | Precision, Recall, F1, ROC-AUC |
| **Explainability** | Custom perturbation-based XAI (word importance, heatmaps) |
| **Rewriting Evaluation** | `Gemini 2.5 Flash` + `BERTScore` |

---

## 3. Quantitative Results

| Metric | Macro Avg | Micro Avg | Weighted Avg |
|:--|--:|--:|--:|
| Precision | 0.91 | 0.93 | 0.92 |
| Recall | 0.88 | 0.89 | 0.89 |
| F1-Score | **0.89** | **0.91** | **0.90** |
| ROC-AUC | — | — | **0.97** |

**Highlights:**
- Excellent discrimination (ROC-AUC 0.97).  
- Slight recall dip in `threat` and `identity_hate` (class imbalance).  
- <1% generalization gap → strong test performance.

---

## 4. Explainability & Analysis

### **Word-Level XAI**
- Perturbation-based explanation computed via leave-one-out masking.
- 🔴 Words increasing toxicity (e.g., “idiot”, “hate”).  
- 🔵 Words reducing toxicity (e.g., “not”, “don’t”).
- Visuals: bar charts, cumulative plots, and attention heatmaps.

### **Negation Sensitivity**
| Phrase Pair | Toxic | Negated | Δ Change |
|:--|--:|--:|--:|
| “You are an idiot” / “You are not an idiot” | 0.92 | 0.34 | −0.58 ✅ |
| “I hate you” / “I don’t hate you” | 0.95 | 0.48 | −0.47 ✅ |

**Result:** Model handles simple negation but still misinterprets sarcasm or double negatives.

### **Frequent Error Patterns**

| Category | Example | Observation |
|:--|:--|:--|
| **Sarcasm** | “Nice work, genius.” | Misread as toxic due to literal tone. |
| **Context Loss** | “That was a kill shot!” | Misclassified as violent. |
| **Identity Hate Recall** | “Go back to your country.” | Under-detected due to phrasing variety. |
| **Negation Handling** | “Not bad at all.” | Occasionally flagged toxic. |

---

## 5. Rewriting Evaluation (Gemini 2.5 + BERTScore)

| Metric | Score |
|:--|--:|
| Precision | 0.952 |
| Recall | 0.945 |
| F1 (Semantic Similarity) | **0.948** |

**Interpretation:**  
Gemini rewrites preserve meaning while removing toxicity — 60% of test comments were rewritten with high semantic alignment.

---

## 6. Limitations

| Area | Limitation |
|:--|:--|
| **Dataset Bias** | Rare labels reduce recall accuracy. |
| **Context Awareness** | Perturbation XAI ignores multi-token context. |
| **Threshold Tuning** | Static 0.5 may not be optimal across labels. |
| **XAI Overhead** | O(n) inference calls for perturbation explanations. |
| **Rewriter Behavior** | Gemini occasionally over-softens tone. |

---

## 7. Proposed Improvements

- Apply **class-balanced focal loss** for rare labels.  
- Explore **gradient-based XAI** (Integrated Gradients, LRP).  
- Adopt **contextual models** (DeBERTa, Longformer).  
- Refine Gemini prompt with **emotion and tone control**.  
- Deploy **interactive explainability dashboard**.

---

## 8. Conclusion

The CleanSpeech model shows **robust performance** in toxicity detection and rewriting.  
Explainability analysis provides transparent token-level reasoning, while Gemini successfully converts harmful expressions into constructive feedback.  
Future work will target **context understanding**, **scalability**, and **real-time interpretability**.

---

> 🧠 **In essence:**  
> CleanSpeech combines transformer-based toxicity detection, explainable AI, and Gemini-powered rewriting to create an interpretable and constructive communication system ready for deployment.
