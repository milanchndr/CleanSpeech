# Milestone 2 — Data Preparation & Preprocessing

## Objective
The goal of this stage was to prepare a clean and structured dataset suitable for model training.  
All preprocessing actions were guided by the observations made during the Exploratory Data Analysis (EDA) stage.

---

## Exploratory Data Analysis (EDA) Summary
Before preprocessing, an exploratory analysis was performed to understand the dataset.  
Key findings included:

- Verified the presence of six toxicity label columns (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).  
- Identified that the dataset is multi-label, with some overlapping toxic categories.  
- Observed imbalance among labels (for example, `toxic` more frequent, while `threat` and `identity_hate` are relatively rare).  
- Detected unwanted text patterns such as URLs, HTML tags, emojis, and non-ASCII characters.  
- These insights directly guided the preprocessing and cleaning decisions described in this milestone.

---

## 1. Data Loading and Inspection
- Loaded `train_data.csv` and `test_data.csv` from the `src/data/` directory.  
- Verified file structure, shapes, and column consistency (`id`, `comment_text`, label columns).  
- Confirmed that text data was correctly formatted and readable.

---

## 2. Data Quality Checks
- Checked for missing and duplicate comments.  
- Identified **8 rows** with missing or empty text and **316 duplicate comments**.  
- Examined label balance and confirmed the presence of six toxicity categories.

---

## 3. Text Cleaning
Applied minimal but effective preprocessing steps to clean comment text:

1. Converted text to lowercase  
2. Removed URLs and HTML tags  
3. Removed emojis and non-ASCII characters  
4. Normalized extra spaces  

A new column `clean_text` was created to store the processed text while keeping the original comments intact.

---

## 4. Final Cleanup and Split
- Removed missing and duplicate entries.  
- Created an `any_toxic` column (binary indicator for toxic vs. non-toxic comments).  
- Performed an **80–20 stratified train-validation split** to maintain label balance.  
- Final dataset sizes:  
  - **Train:** 127,397 rows  
  - **Validation:** 31,850 rows

---

## 5. Saving the Processed Data
Saved final cleaned files for consistent downstream use.

