<div align="center">

<h1 style="align=center">CleanSpeech</h1>

<p><i>Toxicity Detection and Rewriting with Explainable AI</i></p>

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-NLP-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Classifier-HuggingFace%20(mDeBERTa)-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Rewriter-Gemini%20API-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Explainability-SHAP%20%2F%20LIME-purple?style=flat-square"/>
</p>


> CleanSpeech fine-tunes **mDeBERTa-v3-base** for multi-label toxicity classification and  
> uses **Gemini API** to rewrite harmful text into non-toxic alternatives.
> A Streamlit UI provides real-time predictions and **SHAP-based explainability**.

---

## Project Stack
- Frontend/UI: Streamlit (interactive app interface)  
- Backend: Python (Pandas, Scikit-learn, Hugging Face Transformers)  
- Explainability: SHAP / LIME visualizations  
- Data: Jigsaw Toxic Comment Classification Dataset (Kaggle)

---

## Current Progress
| Milestone | Description | Status |
|------------|--------------|--------|
| 1. Problem Definition & Literature Review | Defined objectives and reviewed existing toxicity detection methods. | ![Done](https://img.shields.io/badge/-Completed-green) |
| 2. Data Preparation & Preprocessing | Cleaned dataset, handled duplicates, created `clean_text`, and generated train/val/test splits. | ![Done](https://img.shields.io/badge/-Completed-green) |
| 3. Model Architecture Design | Designing baseline and transformer models for toxicity classification. | ![Done](https://img.shields.io/badge/-Completed-green) |
| 4. Model Training & Evaluation | To be performed after architecture finalization. | ![In Progress](https://img.shields.io/badge/-In%20Progress-yellow) |
| 5. Deployment & UI Integration | Streamlit app to host detection and rewriting module. | ![Planned](https://img.shields.io/badge/-Planned-lightgrey) |

---

## Streamlit Interface
The **CleanSpeech UI** enables users to:
- Enter free-form text.
- View toxicity predictions with probability bars.
- Inspect model explanations (SHAP/LIME visualization).
- Generate **non-toxic rewrites** using the Gemini API.

---

## Repository Structure

```text
CleanSpeech  
│   .gitignore               ← Ignore large model files and virtual envs  
│   README.md                ← Project overview and usage guide  
│
├── doc                      ← Documentation for each milestone  
│   ├── milestone-1          ← Problem definition & literature review  
│   │       Milestone 1.md  
│   │
│   ├── milestone-2          ← Data preprocessing & exploratory analysis  
│   │       classify.png  
│   │       explain.png  
│   │       milestone_2.md  
│   │
│   ├── milestone-3          ← Baseline model (TF-IDF + Logistic Regression)  
│   │       diagram.png  
│   │       formula.png  
│   │       mDeBERTa.png  
│   │       Milestone_3.md  
│   │
│   ├── milestone-4          ← Transformer-based model training (mDeBERTa-v3)  
│   │       milestone-4.md  
│   │
│   └── milestone-5          ← Explainability & text rewriting  
│
├── src                      ← Core data science and modeling pipeline  
│   ├── data                 ← Raw and cleaned dataset splits  
│   │       clean_train.csv  
│   │       clean_val.csv  
│   │       clean_test.csv  
│   │       train_data.csv  
│   │       test_data.csv  
│   │
│   ├── mdeberta-v3-base     ← Transformer fine-tuning, inference & explainability  
│   │   ├── code             ← Modular notebooks for the pipeline  
│   │   │       00_config.ipynb  
│   │   │       01_data.ipynb  
│   │   │       02_modeling.ipynb  
│   │   │       03_train.ipynb  
│   │   │       04_infer.ipynb  
│   │   │       05_explain.ipynb  
│   │   │
│   │   ├── models           ← Fine-tuned DeBERTa model (ignored in Git)  
│   │   │   └── best  
│   │   │           config.json  
│   │   │           tokenizer.json  
│   │   │           model.safetensors  
│   │   │
│   │   ├── notebooks-test   ← Experimental and test notebooks  
│   │   │       HASOC_Preparation.ipynb  
│   │   │       train-v2.ipynb  
│   │   │       toxic_comment_classification.py  
│   │   │       toxic_comment_classification_inference.ipynb  
│   │   │
│   │   └── reports          ← Metrics, SHAP explanations, and preview data  
│   │       │   config_snapshot.json  
│   │       │
│   │       ├── figs         ← Generated SHAP HTML visualizations  
│   │       │   └── shap_gallery_toxic  
│   │       │           index.html  
│   │       │           shap_toxic_01.html ... shap_toxic_05.html  
│   │       │
│   │       └── previews     ← Head previews of training & validation splits  
│   │               train_head.csv  
│   │               val_head.csv  
│   │
│   ├── model-artifacts      ← Saved baseline (TF-IDF + Logistic Regression) models  
│   │   └── tf-idf-log-reg  
│   │           baseline_meta.json  
│   │           baseline_pipeline.joblib  
│   │
│   └── tf-idf-logistic-reg  ← Baseline experimentation notebooks  
│           01_eda.ipynb  
│           02_preprocess.ipynb  
│           03_train-infer.ipynb  
│           04_explain.ipynb  
│
└── ui                       ← Streamlit-based interface for prediction & explainability  
    │   app.py  
    │   charts.py  
    │   components.py  
    │   config.py  
    │   explain.py  
    │   inference.py  
    │   paths.py  
    │   requirements.txt  
    │   __init__.py  
    │
            
```
---

## Getting Started
_Run locally:_
```bash
cd ui
pip install -r requirements.txt
streamlit run app.py
```

---

## Ongoing Steps:
- Finalize model architecture and begin training.
- Develop and integrate the Streamlit UI for user interaction.  

---

<p align="center"> <b>Maintained by Team-10 · DS-Lab Project: CleanSpeech</b><br/> <sub>IIT Madras · Data Science and Applications Program</sub> </p>
