# CleanSpeech
**CleanSpeech: Toxicity Detection & Rewriting with Explainable AI**

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
| 1. Problem Definition & Literature Review | Defined objectives and reviewed existing toxicity detection methods. | Completed |
| 2. Data Preparation & Preprocessing | Cleaned dataset, handled duplicates, created `clean_text`, and generated train/val/test splits. | Completed |
| 3. Model Architecture Design | Designing baseline and transformer models for toxicity classification. | In Progress |
| 4. Model Training & Evaluation | To be performed after architecture finalization. | Pending |
| 5. Deployment & UI Integration | Streamlit app to host detection and rewriting module. | Planned |

---

## Streamlit UI (Coming Soon)
A user-friendly web interface will allow:
- Entering comments to detect and rewrite toxic text  
- Viewing model predictions, category-wise scores, and explanation highlights  
- Real-time demonstration of the toxicity rewriting process

---

## Repository Structure

```text
CleanSpeech/
│
├── doc/
│   ├── Milestone 1.md
│   └── Milestone 2.md
│
├── src/
│   ├── eda.ipynb               # Exploratory Data Analysis
│   ├── prep.ipynb              # Data Preparation & Cleaning
│   ├── baseline.ipynb          # Baseline Model
│   └── data/
│       ├── train_data.csv
│       ├── test_data.csv
│       ├── clean_train.csv
│       ├── clean_val.csv
│       └── clean_test.csv
│
└── ui/
    ├── app.py                  # Streamlit Interface
    └── requirements.txt        # UI Dependencies



### Folder Descriptions

| Folder | Description |
|:--------|:-------------|
| doc/ | Contains project milestone documents and progress reports. |
| src/ | Source notebooks for analysis, preprocessing, and modeling, along with data files. |
| ui/ | Streamlit-based user interface for running and visualizing model outputs. |

---

## Next Steps
- Finalize baseline (TF-IDF + Logistic Regression)  
- Begin transformer model design  
- Integrate results into Streamlit UI  

---

*Maintained by Team-10, DS-Lab Project: CleanSpeech.*
