### **Milestone 2: Dataset Preparation and Preprocessing**


#### **Objective**

The primary objective of this milestone was to select, justify, and prepare high-quality, clean, and well-structured datasets for the training, validation, and testing of the CleanSpeech toxicity detection models. A robust and thoughtful data preparation pipeline is critical to ensure the model's performance, fairness, and ability to generalize to real-world scenarios.

---

#### **1. Dataset Selection and Justification**

A two-tiered dataset strategy was adopted to build a robust model and rigorously evaluate its real-world applicability.

##### **1.1 Primary Dataset: Jigsaw Toxic Comment Classification Challenge**

The Jigsaw dataset was selected as the primary corpus for model training and initial validation.

*   **Justification:**
    *   **Scale and Diversity:** With over 150,000 comments from Wikipedia talk page edits, it provides a large and diverse set of examples for training a deep learning model.
    *   **Multi-Label Classification:** The dataset features six distinct toxicity labels (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`), which directly aligns with our project objective of developing a multi-class toxicity classifier.
    *   **Benchmark Standard:** It is a well-established benchmark in the NLP community, allowing us to compare our model's performance against a wide array of published results and academic baselines.

##### **1.2 Secondary & Testing Dataset: HASOC (Hate Speech and Offensive Content)**

The HASOC (Hate Speech and Offensive Content Identification) dataset was chosen as our secondary dataset, primarily for testing the model's generalization capabilities.

*   **Justification:**
    *   **Multilingual and Code-Mixed Content:** A core objective of CleanSpeech is to support non-English and code-mixed languages. The HASOC dataset contains a rich collection of English, Hindi, and "Hinglish" (a mix of Hindi and English) comments from platforms like Twitter and Facebook. This directly addresses one of the key research gaps we identified.
    *   **Domain Generalization:** The Jigsaw dataset is sourced from Wikipedia, which has a distinct conversational style. HASOC's data comes from social media, which is often less formal, contains more slang, and better represents the target environment for a tool like CleanSpeech. Testing on HASOC will prove our model’s ability to generalize beyond its primary training domain.
    *   **Real-World Complexity:** HASOC data reflects the messy, nuanced, and culturally specific nature of online toxicity, providing a challenging and realistic test bed for our final model.

---

#### **2. Preprocessing Pipeline**

##### **2.1 Jigsaw Dataset Preprocessing**

Before preprocessing, a thorough Exploratory Data Analysis (EDA) was conducted on the Jigsaw data. Key findings included the multi-label nature of the data, a significant imbalance among labels, and the presence of noise like URLs, HTML tags, and non-ASCII characters. These insights directly guided the following cleaning steps.

*   **Data Loading and Quality Checks:**
    *   The `train.csv` and `test.csv` files were loaded and inspected for structural integrity.
    *   Initial checks revealed a small number of rows with missing text and several hundred duplicate comments, which were flagged for removal.

*   **Text Cleaning and Normalization:**
    *   A minimal but effective preprocessing pipeline was applied to the `comment_text` column:
        1.  **Noise Removal:** URLs, HTML tags, emojis, and non-ASCII characters were removed using regular expressions.
        2.  **Whitespace Normalization:** Multiple spaces and line breaks were collapsed into a single space.
        3.  **HTML Tag/Url removal :** URLs, HTML tags, emojis, and non-ASCII characters were removed using regular expressions.
    *   The cleaned text was stored in a new `clean_text` column, preserving the original comment for reference.

*   **Final Structuring and Splitting:**
    *   The previously identified missing and duplicate entries were removed from the dataset.
    *   An `any_toxic` column was temporarily created as a binary flag (1 if any of the six toxicity labels were positive, 0 otherwise). **This column was used exclusively to perform a robust stratified 80-20 train-validation split**, ensuring the proportion of toxic to non-toxic comments was maintained across both sets.
    *   **Final Dataset Sizes:**
        *   **Train Set:** 127,397 samples
        *   **Validation Set:** 31,850 samples

*   **Saving Processed Data:**
    *   The final, processed training and validation dataframes were saved as `.csv` files.
    *   To create a lean dataset optimized for training, unnecessary columns such as the original `id`, `comment_text`, and the temporary `any_toxic` column were **dropped before saving**. The final artifacts contain only the `clean_text` and the six label columns, ensuring consistent and reproducible use in the model training milestone.

##### **2.2 HASOC Dataset Preprocessing**

The HASOC dataset required a more complex preparation process due to its fragmented nature across multiple years and file formats.

*   **Data Unification and Standardization:**
    *   The first step involved unifying HASOC datasets from 2019, 2020, and 2021 for both English and Hindi. These were spread across `.tsv`, `.xlsx`, and `.csv` files.
    *   A custom data loader was built to handle the different formats and standardize column names (e.g., mapping `tweet`, `Text`, and `comment` to a single `text` column).
    *   Essential metadata such as `year`, `language`, and data `split` were added to each record to maintain provenance.

*   **Advanced Text Cleaning:**
    *   A specialized cleaning function (`clean_hasoc_text_v2`) was applied. Unlike the Jigsaw cleaning, this function was designed to preserve semantically important social media cues:
        1.  **URL Replacement:** Replaced URLs with a generic 'URL' token instead of removing them.
        2.  **Username/Hashtag Preservation:** Removed only the `@` and `#` symbols, keeping the username and hashtag text, as they often contain critical context.
        3.  Standard noise removal (HTML tags, invisible characters, extra whitespace) was also performed.

*   **Label Augmentation for Compatibility:**
    *   The original HASOC dataset provides a single binary label ('HOF'/'NOT' for "Hate and Offensive"). To make it compatible with our multi-label Jigsaw-trained model, we augmented it using the **Gemini API**.
    *   Each cleaned comment was passed through a carefully engineered prompt, instructing the model to classify it across the same six toxicity categories used in the Jigsaw dataset.
    *   This step transformed HASOC from a binary-labeled dataset into a rich, multi-label test set that perfectly aligns with our model's output structure.

*   **Saving Processed Data:** The unified, cleaned, and augmented English and Hindi datasets were combined and saved as `hasoc_combined_augmented.csv`, ready for model evaluation.

