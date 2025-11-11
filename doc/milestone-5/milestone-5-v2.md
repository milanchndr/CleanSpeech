# Milestone 5: Model Evaluation & Analysis
**CleanSpeech: Toxicity Detection & Rewriting with Explainable AI**

---
## **Milestone overview**

**CleanSpeech’s BERT-based toxicity detector** demonstrates strong generalization and interpretability, achieving:

* **Macro ROC-AUC:** 0.983  
* **Macro F1:** 0.68  
* **BERTScore (rewriting):** 0.948  

Explainability visualizations confirmed token-level reasoning, while Gemini rewrites effectively converted harmful expressions into constructive feedback.

> In essence, Milestone 5 validated both **model trustworthiness** and **rewrite reliability**, preparing CleanSpeech for deployment and user-facing evaluation in Milestone 6.
---

## **1. Objective**

Following **Milestone 4**, the fine-tuned `bert-base-uncased` model trained on the **Jigsaw Toxic Comment Classification** dataset was evaluated in this milestone. The goal was to analyze performance on unseen data, interpret model predictions, and test the integrated rewriting pipeline using Gemini 2.5.

Key objectives:

* Evaluate model generalization on held-out test data.
* Perform both **quantitative and qualitative analyses**.
* Identify systematic weaknesses through **error and explainability studies**.
* Validate rewrite quality using **BERTScore**.

**Re-training:** No full retraining was performed. Label-wise threshold tuning and post-hoc calibration were applied to improve F1 balance across toxicity types.

---

## **2. Evaluation Setup**

### **Dataset Split**

| Split             | Samples | Purpose                         |
| :---------------- | ------: | :------------------------------ |
| **Train**         |  95,000 | Model fine-tuning (Milestone 4) |
| **Validation**    |  24,000 | Threshold tuning                |
| **Test (Unseen)** | 153,000 | Final Evaluation                |

Each text instance could have multiple labels:  
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

### **Preprocessing**

Applied during evaluation (consistent with training):

```python
text = re.sub(r"http\S+|www\.\S+", " ", text.lower())
text = re.sub(r"[^\x00-\x7F]+", " ", text)
text = re.sub(r"\s+", " ", text).strip()
```

Tokenization used `AutoTokenizer` with `max_length=256` and `padding="max_length"`.

### **Evaluation Environment**

| Component           | Description                      |
| :------------------ | :------------------------------- |
| **Hardware**        | Kaggle GPU T4 (x2)               |
| **Frameworks**      | PyTorch 2.2.0, Transformers 4.41 |
| **Python Version**  | 3.10                             |
| **Precision**       | Mixed FP16 inference             |
| **Reproducibility** | `torch.manual_seed(42)` set      |

---

## **3. Performance Metrics**

| Metric                                | Description                                   |
| :------------------------------------ | :-------------------------------------------- |
| **ROC-AUC (macro)**                   | Threshold-independent discrimination ability  |
| **Precision, Recall, F1 (per label)** | Evaluate class-wise performance               |
| **Subset Accuracy / Hamming Loss**    | Measure exact match and label-wise error rate |

**Why chosen:** Multi-label toxicity detection involves class imbalance and overlapping labels. ROC-AUC captures separability, while macro F1 balances precision and recall across rare classes.

---

## **4. Quantitative Results**

| Label         |   ROC-AUC | F1 (Base=0.5) | F1 (Tuned) | Threshold |
| :------------ | --------: | ------------: | ---------: | --------: |
| toxic         |     0.987 |          0.68 |   **0.74** |      0.42 |
| severe_toxic  |     0.982 |          0.42 |   **0.55** |      0.38 |
| obscene       |     0.989 |          0.71 |   **0.78** |      0.47 |
| threat        |     0.984 |          0.54 |   **0.61** |      0.33 |
| insult        |     0.981 |          0.66 |   **0.72** |      0.46 |
| identity_hate |     0.976 |          0.59 |   **0.67** |      0.40 |
| **Macro Avg** | **0.983** |      **0.60** |   **0.68** |         — |

**Summary:**

* Model generalizes well: only 0.006 AUC drop from validation → test.  
* Threshold tuning improved macro F1 by ~8%.  
* Performance stable across labels; rare classes improved most.

### **Aggregate Metrics**

| Metric                   | Value |
| :----------------------- | ----: |
| **Macro ROC-AUC**        | 0.983 |
| **Macro F1 (Optimized)** |  0.68 |
| **Subset Accuracy**      |  0.47 |
| **Hamming Loss**         | 0.062 |

**Learning Curve:** Validation and training losses converged by epoch 3, indicating minimal overfitting.

---

## **5. Qualitative Results**

### **Example Predictions**

| Input                             | True Labels   | Predicted (Top-3)           |
| :-------------------------------- | :------------ | :-------------------------- |
| “You are such an idiot.”          | toxic, insult | toxic (0.91), insult (0.85) |
| “You are not an idiot.”           | none          | none                        |
| “Go die already!”                 | threat, toxic | threat (0.89), toxic (0.84) |
| “That was disgusting.”            | obscene       | obscene (0.80)              |
| “I completely disagree with you.” | none          | none                        |

### **Explainability Visualization**

* Perturbation-based importance scores highlight key words contributing to toxicity.  
* ![red](https://img.shields.io/badge/Increases%20Toxicity-red) → words like *“idiot”*, *“hate”*  
* ![blue](https://img.shields.io/badge/Reduces%20Toxicity-blue) → words like *“not”*, *“don’t”*  
* Cumulative impact plots visualize how toxicity probability builds progressively across tokens.

<!doctype html>
<html>
<head><meta charset="utf-8" /><title>SHAP Text Explanation</title></head>
<body style="font:16px/1.6 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding:24px;">
  <h2 style="margin:0 0 8px;">SHAP Text Explanation</h2>
  <div style="color:#555; margin-bottom:4px;">Label: <b>toxic</b></div>
  <div style="color:#555; margin-bottom:4px;">Predicted P(toxic): <b>0.986</b></div>
  <div style="color:#777; margin-bottom:16px;">Max |SHAP|: 0.4052</div>
  
    <!-- <div style="font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;">
      <div style="margin-bottom:8px;"><b>Legend</b> — <span style="color:#c00;"><b>red</b></span>: pushes <i>up</i> (increases toxicity), <span style="color:#08f;"><b>blue</b></span>: pushes <i>down</i> (decreases toxicity); intensity ∝ |SHAP|</div>
      <div style="display:flex; gap:12px; align-items:center; margin-bottom:12px;">
        <span style="background:rgba(255,0,0,0.35); padding:4px 8px; border-radius:4px;">push ↑ (more toxic)</span>
        <span style="background:rgba(0,120,255,0.35); padding:4px 8px; border-radius:4px;">pull ↓ (less toxic)</span>
      </div>
    </div> -->
    
  <div style="border:1px solid #eee; padding:12px; border-radius:8px;">
    <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.15000001378932548);"></span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(255,0,0,0.2788920604093911);">You</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(255,0,0,0.15497692885270387);"> are</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(255,0,0,0.189523408759821);"> an</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.15213593549372117);"> </span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.23522983731119324);">absolute</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(255,0,0,0.7499999851938245);"> idiot</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.16587771057154777);"> and</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.15026574098634257);"> </span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.20129744777849087);">a</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(255,0,0,0.2944799400429505);"> dis</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.27408209408875445);">grace</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.20120076943848897);">.</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(255,0,0,0.15);"></span>
  </div>
</body>
</html>

<!doctype html>
<html>
<head><meta charset="utf-8" /><title>SHAP Text Explanation</title></head>
<body style="font:16px/1.6 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding:24px;">
  <h2 style="margin:0 0 8px;">SHAP Text Explanation</h2>
  <div style="color:#555; margin-bottom:4px;">Label: <b>toxic</b></div>
  <div style="color:#555; margin-bottom:4px;">Predicted P(toxic): <b>0.000</b></div>
  <div style="color:#777; margin-bottom:16px;">Max |SHAP|: 0.1241</div>
  
    <div style="font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;">
      <div style="margin-bottom:8px;"><b>Legend</b> — <span style="color:#c00;"><b>red</b></span>: pushes <i>up</i> (increases toxicity), <span style="color:#08f;"><b>blue</b></span>: pushes <i>down</i> (decreases toxicity); intensity ∝ |SHAP|</div>
      <div style="display:flex; gap:12px; align-items:center; margin-bottom:12px;">
        <span style="background:rgba(255,0,0,0.35); padding:4px 8px; border-radius:4px;">push ↑ (more toxic)</span>
        <span style="background:rgba(0,120,255,0.35); padding:4px 8px; border-radius:4px;">pull ↓ (less toxic)</span>
      </div>
    </div>
    
  <div style="border:1px solid #eee; padding:12px; border-radius:8px;">
    <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.15000002565066425);"></span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.6723397901783577);">Thanks</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.2725782440169456);"> for</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.346742515877133);"> the</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.5471902722875057);"> clar</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.6437238944841895);">ification</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.37438450812256063);">,</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.17355492501077102);"> </span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.7499999516405974);">appreciate</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.20577583419241335);"> it</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(0,120,255,0.3922865757603108);">.</span> <span style="padding:2px 4px; margin:1px; border-radius:4px; background-color: rgba(255,0,0,0.15);"></span>
  </div>
</body>
</html>

---

![Attention Heatmap](attention_heatmap.png)
**Figure:** Attention weights visualization shows model focus on toxic keywords.


### **Rewriting Examples (Gemini 2.5)**

| Toxic Input                              | Constructive Rewrite                                                         |
| :--------------------------------------- | :--------------------------------------------------------------------------- |
| “You are such an idiot and I hate you!”  | “I strongly disagree with your approach and find it frustrating.”            |
| “This politician is a liar and a thief.” | “I question this politician’s honesty and integrity based on their actions.” |
| “That guy is a complete moron.”          | “I think that person made poor decisions.”                                   |

**Observation:** Rewrites preserve semantic meaning (BERTScore F1 = 0.948).

---

## **6. Error Analysis**

### **6.1 Quantitative Trends**

| Issue               | Observation                                                  |
| :------------------ | :----------------------------------------------------------- |
| Class Imbalance     | `threat` and `identity_hate` have lower recall (~0.55–0.63). |
| Ambiguity / Sarcasm | “Nice job, genius.” often misclassified as toxic.            |
| Context Dependence  | “That was a kill shot!” flagged as violent.                  |
| Mixed Language      | Words like “pagal” not recognized.                           |

### **6.2 Root Causes**

* Semantic ambiguity (implied toxicity).  
* Limited multilingual exposure.  
* Phrase-level context missing from word-based perturbation method.

---

## **7. Limitations**

1. Imbalanced data lowers rare-class recall.  
2. Token-level XAI lacks phrase-level understanding.  
3. Static thresholding unsuitable across all labels.  
4. Perturbation explanations are computationally expensive (O(n)).  
5. Gemini rewrites can over-soften original tone.

---

## **8. Proposed Improvements / Next Steps**

* Use **class-balanced focal loss** to improve minority labels.  
* Explore **gradient-based XAI** (Integrated Gradients, LRP).  
* Test **context-aware models** (DeBERTa, Longformer).  
* Add **tone-controlled rewrite prompts** for better emotion preservation.  
* Integrate **interactive visualization dashboard** for real-time explanations.

---

## **9. Artifacts & Reproducibility**

| Artifact             | Description                                |
| :------------------- | :----------------------------------------- |
| `evaluation.ipynb`   | Metric computation and plots               |
| `xai_analysis.ipynb` | Explainability via perturbation & heatmaps |
| `rewrite_eval.ipynb` | Gemini rewrite + BERTScore evaluation      |
| `best_model/`        | Fine-tuned weights (`.safetensors`)        |
| Dataset Split        | 80% train / 20% test                       |

**Reproducibility Command:**

```bash
python evaluate.py --model_dir best_model --split test
```

---