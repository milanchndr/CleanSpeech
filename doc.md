# **UI Development Documentation – Streamlit Interface for CleanSpeech**

## **1. Overview**

### **Purpose**

The CleanSpeech UI provides an interactive, user-friendly interface for detecting and analyzing toxic content in text.
It serves as the frontend layer over multiple toxicity detection backends, enabling:

* Model selection
* Threshold customization
* Real-time text analysis
* Visualization of explainability outputs (probabilities, attention, word importance)

### **Architecture Summary**

The UI is implemented as a **multi-page Streamlit application** with a sidebar-driven configuration panel.
Core components:

* **app.py** — Main entry and navigation controller
* **pages/chat.py** — Chat-style toxicity detection
* **pages/explain.py** — Explainability dashboard for model outputs
* **_temp/config.py** — Page config + available model registry

Data flow:

```
User Input → Streamlit UI → Selected Backend Model → API Response → UI Visualizations
```

---

## **2. Environment Setup**

### **Dependencies**

The UI relies on the following libraries:

* **streamlit** – primary UI framework
* **pandas** – result formatting & tables
* **requests** – backend API calls
* **json** – result persistence
* **numpy** – array operations

### **Installation**

```
pip install streamlit pandas requests numpy
```

### **Python Version**

Python **3.8 or above**

### **Hardware Requirements**

Minimal – CPU-only execution is sufficient as the UI handles only interface and visualization.

---

## **3. UI Architecture**

### **Application Structure**

```
ui/
├── app.py                # Main Streamlit entry point
├── pages/
│   ├── chat.py           # Chat UI for toxicity detection
│   └── explain.py        # Explainability dashboard
└── _temp/
    └── config.py         # PAGE_CONFIG + MODEL_MAPPER
```

### **Page Configuration**

Two primary pages:

1. **Chat Page**

   * A chat-like interface
   * Allows users to enter free-text
   * Displays toxicity categories and detected levels

2. **Explain Page**

   * Detailed visual explanations (probabilities, attention maps, token importance)
   * Useful for understanding model behaviour

### **Navigation**

The app uses **Streamlit multipage architecture** with icon-based navigation:

```
st.navigation([
   st.Page("pages/chat.py", title="Chat", icon="💬"),
   st.Page("pages/explain.py", title="Explain", icon="🔍")
])
```

---

## **4. Sidebar Controls**

The sidebar functions as the global configuration panel for the entire UI.

### **Model Selection**

* Dropdown listing all available models from `MODEL_MAPPER`
* A **Submit** button ensures explicit confirmation and updates `st.session_state.model`

### **Toxicity Threshold Controls**

Six sliders, one per toxicity category:

| Category      | Default | Range       | Step |
| ------------- | ------- | ----------- | ---- |
| toxic         | 0.8     | 0.00 – 1.00 | 0.05 |
| severe_toxic  | 0.5     | 0.00 – 1.00 | 0.05 |
| obscene       | 0.7     | 0.00 – 1.00 | 0.05 |
| threat        | 0.1     | 0.00 – 1.00 | 0.05 |
| insult        | 0.5     | 0.00 – 1.00 | 0.05 |
| identity_hate | 0.4     | 0.00 – 1.00 | 0.05 |

Values update the shared session state dictionary:

```
st.session_state.toxicity_threshold
```

---

## **5. Session State Management**

The UI uses `st.session_state` to maintain consistency across pages.

| Variable             | Type          | Purpose                              |
| -------------------- | ------------- | ------------------------------------ |
| `model`              | string        | selected model backend               |
| `current_message`    | string / None | last user input text                 |
| `api_response`       | dict / None   | last model output                    |
| `toxicity_threshold` | dict          | threshold settings for each category |

These values ensure that:

* Changes in the sidebar persist across pages
* Both pages read from the same model & thresholds
* Explain page can render results from the chat page

---

## **6. User Interaction Flow**

1. User runs the UI:

   ```
   streamlit run ui/app.py
   ```
2. Sidebar loads available models and threshold sliders.
3. User selects model + thresholds and clicks **Submit**.
4. User navigates to **Chat** or **Explain** tab.
5. `chat.py` sends requests to backend API and stores responses.
6. `explain.py` reads saved output from session state and visualizes:

   * Probabilities (bar charts)
   * Word importance (colored tokens)
   * Attention heatmap
   * Cumulative impact curve

---

## **7. Configuration Dependencies**

The UI depends on `_temp/config.py`:

### **Required Objects**

* **PAGE_CONFIG**
  Streamlit page settings: title, icon, layout.

* **MODEL_MAPPER**
  Dictionary mapping model names → actual model endpoints/paths.

Example:

```python
MODEL_MAPPER = {
    "distilbert-toxic": "http://localhost:5000/predict",
    "roberta-toxic": "hf-space-endpoint"
}
```

---

## **8. Reproducibility Checklist**

To fully reproduce the Streamlit UI:

* [x] `requirements.txt` includes **streamlit**
* [x] `_temp/config.py` defines PAGE_CONFIG and MODEL_MAPPER
* [x] `pages/chat.py` and `pages/explain.py` present and functional
* [x] Python ≥ 3.8
* [x] Instructions documented:

```
cd d:\User\vscode\CleanSpeech
pip install -r requirements.txt
streamlit run ui/app.py
```

---

## **9. Running the UI Locally**

```bash
# Navigate to project
cd d:\User\vscode\CleanSpeech

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit UI
streamlit run ui/app.py
```

The application will open at:

```
http://localhost:8501
```

