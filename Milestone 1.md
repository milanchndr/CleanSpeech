# Milestone 1: Problem Definition & Literature Review  
**CleanSpeech: Toxicity Detection & Rewriting with Explainable AI**

---

## 1. Problem Definition

### 1.1 Background
Online toxicity in social media, gaming platforms, and forums creates hostile environments that harm user experience and platform reputation. Toxic behavior includes hate speech, harassment, profanity, and threats. Current AI moderation tools typically provide only binary classifications (toxic/non-toxic) without explaining why content was flagged or offering constructive alternatives.

**Key Issues:**
- 41% of U.S. adults report experiencing online harassment (Pew Research Center, 2021)  
- LGBTQ+ youth face higher rates of cyberbullying, with 15% of high school students reporting electronic bullying (Trevor Project, 2021)  
- Existing tools lack transparency in moderation decisions  
- No guidance provided to users on improving their communication  
- Poor support for non-English and code-mixed languages  

### 1.2 Problem Statement
There is a need for an intelligent moderation system that not only detects and categorizes toxic content but also explains its decisions and provides constructive alternatives, helping users communicate better while maintaining safer online spaces.

### 1.3 Project Objectives
- Develop a multi-class toxicity classifier to detect and categorise toxic comments  
- Implement severity scoring to differentiate between mild and severe toxicity  
- Provide explainable predictions by highlighting problematic words  
- Create a rewriting module that converts toxic comments into constructive alternatives  
- Build a working demo application to showcase the solution  

### 1.4 Success Criteria
- Classification accuracy >80% on standard toxicity datasets  
- Rewritten comments preserve original meaning while reducing toxicity  
- The system provides clear explanations for toxic predictions  
- Demo successfully processes real-world text examples  

---

## 2. Literature Review

### 2.1 Toxicity Detection Methods
**Early Approaches:** Traditional machine learning methods used feature engineering (n-grams, sentiment scores) with classifiers like SVM and Random Forests. These achieved moderate accuracy but struggled with context understanding and generalization.

**Transformer-Based Models:**  
Modern approaches leverage pre-trained transformer models fine-tuned on toxicity datasets. BERT-based models showed significant improvements in understanding context and nuance in toxic language.

**Key Research:**

1. **Mozafari et al. (2020)** - *"A BERT-Based Transfer Learning Approach for Hate Speech Detection in Online Social Media"*  
   [DOI link](https://doi.org/10.48550/arXiv.1910.12574)  
   - **Contribution:** Demonstrated that fine-tuning BERT on hate speech datasets achieves superior performance compared to traditional methods  
   - **Methodology:** Fine-tuned BERT-base on multiple hate speech datasets with multi-label classification  
   - **Results:** Achieved 93-95% F1-score across different toxicity categories  
   - **Relevance to our project:** Validates the use of transformer models for toxicity detection and multi-label classification approach  
   - **Limitation:** No explainability mechanism, treats all toxic content equally without severity differentiation  

2. **Mathew et al. (2021)** - *"HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection"*  
   - **Contribution:** Created a dataset with human-annotated explanations (rationales) for why content is toxic  
   - **Methodology:** 20,000 social media posts labeled with toxicity categories plus word-level annotations indicating which words contribute to toxicity  
   - **Results:** Demonstrated that models with attention mechanisms can identify toxic keywords with 70-75% agreement with human annotators  
   - **Relevance to our project:** Provides both a benchmark dataset and methodology for implementing explainability in toxicity detection  
   - **Key Insight:** Explainability improves user trust and helps content creators understand moderation decisions  

3. **He et al. (2021)** - *"DeBERTa: Decoding-enhanced BERT with Disentangled Attention"*  
   - **Contribution:** Introduced disentangled attention mechanism that separately encodes content and position information  
   - **Methodology:** Enhanced BERT architecture with disentangled attention and improved mask decoder  
   - **Results:** Achieved state-of-the-art results on SuperGLUE benchmark, outperforming BERT and RoBERTa  
   - **Relevance to our project:** DeBERTa's superior context understanding makes it ideal for detecting nuanced toxicity (sarcasm, implicit hate speech)  
   - **Application:** We will use DeBERTa-v3 as our base model for toxicity classification  

---

### 2.2 Text Rewriting and Detoxification
Limited research exists on constructive rewriting of toxic content. Most work focuses on style transfer or general text paraphrasing.

**Key Research:**

1. **Nogueira dos Santos et al. (2018)** - *"Fighting Offensive Language on Social Media with Unsupervised Text Style Transfer"*  
   - **Contribution:** Applied style transfer techniques to convert toxic text to non-toxic while preserving meaning  
   - **Methodology:** Used encoder-decoder architecture with adversarial training  
   - **Results:** Successfully reduced toxicity but struggled with semantic preservation (often generated generic neutral text)  
   - **Relevance to our project:** Highlights the challenge of maintaining meaning during rewriting  
   - **Our Approach:** Use large language models (Gemini) with carefully designed prompts to better preserve context  

2. **Madaan et al. (2020)** - *"Politeness Transfer: A Tag and Generate Approach"*  
   - **Contribution:** Developed a method to convert impolite text to polite alternatives  
   - **Methodology:** Sequence-to-sequence model with politeness markers as control codes  
   - **Results:** 85% semantic similarity between original and rewritten text  
   - **Limitation:** Required parallel corpus (paired examples of impolite/polite text)  
   - **Relevance to our project:** Demonstrates feasibility of preserving meaning while changing tone  

---

### 2.3 Explainable AI in NLP
- **Attention Mechanisms:** Attention weights in transformers can highlight which words influenced predictions. However, research shows attention alone may not always be reliable for explanations.  
- **Post-hoc Methods:** SHAP (Shapley Additive Explanations) and LIME provide model-agnostic explanations by measuring feature importance. SHAP is particularly suited for transformer models.

**Key Research:**

1. **Lundberg & Lee (2017)** - *"A Unified Approach to Interpreting Model Predictions"*  
   - **Contribution:** Introduced SHAP values based on game theory for model interpretability  
   - **Methodology:** Uses Shapley values to quantify each feature's contribution to predictions  
   - **Results:** Provides consistent and locally accurate explanations across different model types  
   - **Relevance to our project:** We will use SHAP to highlight toxic words and explain classification decisions  
   - **Implementation:** SHAP values will show how much each word increases/decreases toxicity score  

---

## 3. Existing Solutions and Baselines

### 3.1 Commercial Solutions
- **Perspective API (Google Jigsaw):** Provides toxicity scores (0-1) for text. Used by platforms like New York Times, Wikipedia. Limitation: No explanations, no constructive feedback.  
- **Azure Content Moderator:** Multi-category classification with severity levels. Limitation: Generic categories, not customizable.  
- **OpenAI Moderation API:** Fast classification across 11 categories. Limitation: No explainability or rewriting features.  

### 3.2 Academic Baselines
- **Jigsaw Toxic Comment Dataset:** 160,000+ Wikipedia comments with 6 toxicity labels. Competition winners achieved high performance using ensemble models (BERT, RoBERTa variants). Standard benchmark for toxicity detection research.  
- **HateXplain Dataset:** 20,000 posts with explanations. Best models: ~93% F1-score with 70-75% explainability agreement with human annotators.  

---

## 4. Research Gaps and Our Contribution

### 4.1 Identified Gaps
- No End-to-End Solution: Most systems only detect toxicity without helping users improve their communication  
- Limited Explainability: Commercial tools provide scores without explaining which words are problematic  
- No Constructive Feedback: Content gets removed/flagged but users don't learn how to communicate better  
- Poor Multilingual Support: Most models struggle with code-mixed languages like Hinglish  

### 4.2 Our Contribution
- **Integrated Pipeline:** Detection → Explanation → Rewriting in one system  
- **Explainable Predictions:** Highlight problematic words and categorize toxicity types  
- **Constructive Rewriting:** Generate polite alternatives that preserve user intent  
- **Practical Demo:** User-friendly interface for real-world testing  

---

## 5. Proposed Methodology Overview

### 5.1 Architecture
```

Input Text
↓
[1. Toxicity Detection]
DeBERTa Classifier
(Multi-label + Severity)
↓
[2. Explainability]
SHAP Analysis
(Word-level attribution)
↓
[3. Rewriting]
Gemini API
(Constructive alternative)
↓
Output (Label + Explanation + Rewrite)

```

### 5.2 Datasets
- **Primary:** Jigsaw Toxic Comment Classification (training)  
- **Validation:** HateXplain (explainability validation)  
- **Testing:** HASOC Hindi-English (generalization)  

### 5.3 Evaluation Metrics
- **Classification:** F1-score, ROC-AUC per category  
- **Explainability:** Agreement with human annotations  
- **Rewriting:** Semantic similarity (BERTScore), toxicity reduction rate  

---

## 6. Expected Outcomes
- A working multi-class toxicity classifier with >80% accuracy  
- Explainable predictions highlighting toxic words  
- Rewriting module that maintains meaning while reducing toxicity  
- Interactive demo for real-time testing  
- Comprehensive evaluation against existing baselines  

---

## 8. Key References

### Toxicity Detection
- Mozafari, M., Farahbakhsh, R., & Crespi, N. (2020). *A BERT-based transfer learning approach for hate speech detection in online social media*. Complex Networks and Their Applications VIII.  
- Mathew, B., et al. (2021). *HateXplain: A benchmark dataset for explainable hate speech detection*. AAAI Conference on Artificial Intelligence.  
- He, P., et al. (2021). *DeBERTa: Decoding-enhanced BERT with disentangled attention*. ICLR.  

### Text Rewriting
- Nogueira dos Santos, C., et al. (2018). *Fighting offensive language on social media with unsupervised text style transfer*. ACL.  
- Madaan, A., et al. (2020). *Politeness transfer: A tag and generate approach*. ACL.  

### Explainability
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NeurIPS.  

### Datasets
- Jigsaw/Conversation AI. (2018). *Toxic Comment Classification Challenge*. Kaggle.  
- Wulczyn, E., Thain, N., & Dixon, L. (2017). *Ex machina: Personal attacks seen at scale*. WWW.  


