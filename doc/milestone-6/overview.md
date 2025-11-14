# **Milestone - 6**

## A. Overview

### 1. Purpose

Online toxicity—including hate speech, harassment, and threats—creates hostile digital environments that degrade user experience and harm platform integrity. Standard content moderation tools often act as opaque "black boxes," flagging or removing content without explaining their reasoning or offering users a path to improve their communication. This lack of transparency and constructive feedback fails to address the root of the problem and can lead to user frustration.

**CleanSpeech** was developed to address these gaps. It is an end-to-end intelligent moderation system designed not just to **detect** toxic content but also to **explain** its reasoning and **rewrite** harmful messages into constructive alternatives.

The primary objectives are to:
*   **Create a transparent moderation pipeline** that provides clear, word-level explanations for its decisions.
*   **Educate users** by showing them which specific parts of their language are problematic.
*   **Foster healthier online communities** by offering constructive feedback that preserves the user's original intent while removing toxicity.

### 2. Architecture Summary

CleanSpeech is built on a modular, linear architecture that processes user input through three core stages: **Detection**, **Explanation**, and **Rewriting**. Each component is designed to be independent, allowing for future upgrades and maintenance.


**Figure 1:** The end-to-end system architecture of CleanSpeech, showing the flow from user input to a structured output containing classification, explanation, and a constructive rewrite.

#### Data Flow

The system processes text through the following steps:

1.  **User Input:** A user submits text through the Streamlit frontend UI.
2.  **API Request:** The frontend sends the text to the backend API for analysis.
3.  **Text Preprocessing:** The backend cleans the input text by converting it to lowercase and removing noise such as URLs, HTML tags, and non-ASCII characters. The text is then tokenized for the model.
4.  **Toxicity Classification:** The fine-tuned **`mDeBERTa-v3-base`** model performs multi-label classification, generating probabilities for six toxicity categories (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).
5.  **Explainability Analysis:** A custom word-level perturbation analysis is performed. The system measures how the toxicity prediction changes when each word is removed, assigning an "importance score" that quantifies its contribution to the final classification.
6.  **Constructive Rewriting:** If the text is classified as toxic, the original comment and its toxicity analysis are sent to the **Google Gemini API**. A carefully engineered few-shot prompt guides the API to generate a constructive, non-toxic alternative that preserves the core message.
7.  **Response Composition:** The backend packages the classification probabilities, word-level importance scores, and the rewritten text into a structured JSON response.
8.  **UI Visualization:** The Streamlit frontend receives the JSON response and renders it through a series of interactive visualizations, including probability bars, color-coded text explanations, and attention heatmaps.

### 3. Deployed Components

CleanSpeech is composed of several interconnected components, each deployed and accessible for interaction or integration.

| Component | Technology | Status & Location | Description |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | Streamlit | **Deployed on streamlit.io** | An interactive, multi-page web application for real-time text analysis and visualization. Run with `https://cleanspeech.streamlit.app`. |
| **Backend API** | FastAPI | **Deployed on Hugging Face Spaces** | The core inference engine that exposes the model's capabilities via a REST API endpoint. **Endpoint:** `https://milanchndr-Toxic-Comment-Classifier-Explainer.hf.space/predict` |
| **Core Classifier Model** | `mDeBERTa-v3-base` | **Deployed on Hugging Face Hub** | The fine-tuned, multi-label toxicity detection model, available for download and use in other applications. **Model ID:** `milanchndr/toxicity-classifier-mdeberta` |
| **Generative Rewriter** | Google Gemini | **External API** | The text rewriting module, accessed via the Google Generative AI API. Requires a valid API key for operation. |
