<h1 align="center">CleanSpeech</h1>
<h3 align="center">Toxicity Detection & Rewriting with Explainable AI</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-NLP-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Classifier-HuggingFace%20(mDeBERTa)-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Rewriter-Gemini%20API-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Explainability-SHAP%20%2F%20LIME-purple?style=flat-square"/>
</p>

---

## Overview
CleanSpeech is an academic DS-Lab project (Team-10) focused on building an AI system that detects and rewrites toxic text using explainable models.  
The system aims to identify different types of toxicity (toxic, obscene, insult, threat, identity hate, etc.) and provide meaningful, non-toxic reformulations.

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

## Streamlit UI
The CleanSpeech project includes a Streamlit-based user interface for easy interaction with the toxicity detection and rewriting models. The UI allows users to input text, view toxicity predictions, explanations, and rewritten non-toxic versions.

---

## Repository Structure

```text

CleanSpeech/
|   .gitignore
|   README.md
|
+---doc
|       architecture.jpg
|       architecture.png
|       classify.png
|       explain.png
|       formula.png
|       Milestone 1.md
|       milestone_2.md
|       Milestone_3.md
|       milestone_3_temp.md
|
+---src
|   |   baseline.ipynb
|   |   base_exp.ipynb
|   |   eda.ipynb
|   |   HASOC_Preparation.ipynb
|   |   prep.ipynb
|   |
|   +---data
|   |       clean_test.csv
|   |       clean_train.csv
|   |       clean_val.csv
|   |       test_data.csv
|   |       train_data.csv
|   |
|   \---models
|           baseline_meta.json
|           baseline_pipeline.joblib
|
\---ui
    |   app.py
    |   charts.py
    |   components.py
    |   config.py
    |   explain.py
    |   inference.py
    |   paths.py
    |   requirements.txt
    |   __init__.py
```

---

## Folder Descriptions
| Folder | Description |
|:--------|:-------------|
| doc/ | Contains project milestone documents and progress reports. |
| src/ | Source notebooks for analysis, preprocessing, and modeling, along with data files. |
| src/data/ | Processed datasets for training, validation, and testing. |
| src/models/ | Saved model artifacts including pipelines and metadata. |
| ui/ | Streamlit-based user interface for running and visualizing model outputs. |

---

## Ongoing Steps:
- Finalize model architecture and begin training.
- Develop and integrate the Streamlit UI for user interaction.  

---

<p align="center"> <b>Maintained by Team-10 · DS-Lab Project: CleanSpeech</b><br/> <sub>IIT Madras · Data Science and Applications Program</sub> </p>
