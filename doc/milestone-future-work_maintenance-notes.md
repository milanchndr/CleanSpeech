# **CleanSpeech: Future Work and Maintenance Notes**

This document provides practical guidance for maintaining, updating, and extending the CleanSpeech project. It outlines known limitations, suggests potential improvements, and provides a clear process for retraining the core model.

## 1. Possible Extensions and Improvements

CleanSpeech provides a strong foundation, but several avenues exist for future enhancements.

### 1.1. Model Enhancements

*   **🧪 Experiment with Different Architectures:**
    *   **Larger Models:** Fine-tune larger backbones like `Gemma3` for potentially higher accuracy, especially on nuanced text.
    *   **Efficient Models:** For applications requiring lower latency and cost, fine-tune a distilled model like `DistilBERT` or `MobileBERT`.
    *   **Alternative Models:** Explore `ModernBERT` as mentioned in the architecture justification, which might offer a better balance of performance and efficiency.

*   **⚙️ Improve Explainability (XAI):**
    *   The current word perturbation method is intuitive but computationally expensive (O(n) predictions per explanation). Explore faster, gradient-based methods like **Integrated Gradients** or **LRP (Layer-wise Relevance Propagation)**.
    *   Implement **phrase-level explanations** to capture how groups of words (e.g., "go die in a fire") contribute to toxicity, which is more meaningful than individual word scores.

### 1.2. Dataset Expansion and Multilingual Support

*   **🗣️ Activate Multilingual Capabilities (High Priority):**
    *   The `mDeBERTa-v3-base` model is inherently multilingual, but it was only fine-tuned and evaluated on **English-only data** from the Jigsaw dataset. A key opportunity for future work is to leverage its multilingual potential.
    *   **Action:** Fine-tune the model on datasets like the **Jigsaw Multilingual Toxic Comment Classification** (which includes Spanish, French, Italian, Russian, etc.) to formally add support for new languages.

*   **🇮🇳 Add Code-Mixed Language Support (Hinglish):**
    *   The **HASOC (Hate Speech and Offensive Content)** dataset was prepared and augmented in Milestone 2 for this purpose. The next step is to use this dataset to formally evaluate and potentially fine-tune the model to handle code-mixed Hindi-English ("Hinglish"), a common scenario on social media.

*   **📈 Incorporate More Diverse English Data Sources:**
    *   To improve generalization beyond Wikipedia comments, incorporate training data from:
        *   **Gaming Platforms:** (e.g., comments from Twitch or game-specific forums).
        *   **Reddit:** A diverse source of conversational styles and community-specific slang.

*   **🔄 Implement a Feedback Loop:**
    *   Add a feature to the UI allowing users to flag incorrect predictions. These flagged examples can be collected, manually reviewed, and used to create a new, high-quality dataset for periodic retraining.

### 1.3. Feature Additions

*   **🎨 Interactive Explanations:** Enhance the UI to allow users to hover over or click on a color-coded word to see its exact importance score.
*   **🎭 Tone Control for Rewriting:** Modify the Gemini prompt to allow users to select the desired tone for the rewrite (e.g., "Formal," "Casual," "More Assertive").
*   **📊 Batch Processing:** Add a feature to the Streamlit UI for uploading a CSV of comments and receiving a downloadable report.
*   **📈 Fairness and Bias Auditing:** Integrate tools like the Hugging Face `evaluate-fairness` library to systematically audit the model for biases against specific identity groups.

---

## 2. Known Limitations

Be aware of the following limitations when deploying or extending the system.

### 2.1. Performance and Scalability

*   **⚠️ High Latency of Explainability:** The word perturbation method for XAI requires `N+1` predictions for a comment with `N` words, making it unsuitable for high-throughput, real-time applications without optimization.
*   **🐌 CPU Inference is Slow:** The `mDeBERTa-v3-base` model (183M parameters) runs significantly slower on a CPU. The backend API **must be hosted on a GPU-accelerated instance** for production use.

### 2.2. Language and Context Coverage

*   **🚫 English-Only Model:** The current production model **only supports English**. It will perform poorly on other languages and code-mixed text, as confirmed by the error analysis in Milestone 5 where it failed to recognize non-English words like "pagal".
*   **💬 Lack of Conversational Context:** The model analyzes each comment in isolation. It cannot understand conversation history, making it vulnerable to misinterpreting sarcasm, in-jokes, or reclaimed slurs.
*   **📈 Evolving Slang:** The model's knowledge is static. It will fail to recognize new toxic slang or memes until it is retrained on more recent data.

### 2.3. Hardware and Cost

*   **💰 Cost of Generative Rewriting:** The rewriting feature relies on the Google Gemini API, which is a **metered, paid service**. Scaling this feature will incur significant operational costs.
*   **🖥️ Hardware Requirements:** The backend API requires a machine with a modern GPU (e.g., NVIDIA T4, A10G) and sufficient vRAM (>8 GB recommended).

---

## 3. Model Retraining and Updating Guide

To keep the model relevant and improve its performance, it should be periodically retrained.

### 3.1. When to Retrain

Consider retraining the model if:
1.  **Model Drift is Detected:** Performance metrics on a held-out test set of recent, real-world data show a significant decline.
2.  **New Datasets Become Available:** You acquire a new dataset to add multilingual support or improve coverage on diverse data sources.
3.  **New Forms of Toxicity Emerge:** The model consistently fails to identify new toxic slang or hate symbols.

### 3.2. Retraining Steps

The entire process can be reproduced using the scripts provided in the project.

1.  **Step 1: Prepare Your New Data**
    *   Ensure your new training data is in a CSV file with a `comment_text` column and the six binary label columns.
    *   Apply the same preprocessing steps documented in **Milestone 2** using the functions in `toxic_comment_classification.py`.
    *   Create new stratified `train` and `validation` splits.

2.  **Step 2: Run the Training Script**
    *   Open `toxic_comment_classification.py`.
    *   Point the data loading section to your new training and validation files.
    *   Execute the script. The best-performing model checkpoint will be saved to the `best_model/` directory.

3.  **Step 3: Evaluate and Tune Thresholds**
    *   Run the evaluation portion of `explainbility+rewrite (2).py` using the predictions from your new model.
    *   This will help you find the **new optimal F1-score thresholds** for each label. This is a critical step, as thresholds may change with a new model.
    *   Compare the final metrics to the previous model to confirm successful retraining.

4.  **Step 4: Deploy the Updated Model**
    *   Upload the new model files (from `best_model/`) to the Hugging Face Hub.
    *   Update the `MODEL_ID` in your backend API deployment.
    *   Redeploy the backend API.

---

## 4. Contacts and Maintainers

For questions, bug reports, or contributions, please use the following channels:
 
*   **Team Contact :** **Milan Chandra:** [github.com/milanchndr](https://github.com/milanchndr), **Alauddin Ansari:** [github.com/git4alauddin](https://github.com/git4alauddin),**Soumyadip Ghorai:**[github.com/sghoraiIITM](https://github.com/sghoraiIITM)    *   
    *   
    *   **Suman Ghorai:** 
*   **Bug Reports & Feature Requests:** Please open an issue on the official GitHub repository (if applicable).
