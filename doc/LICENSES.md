# **Licensing & Attribution**

This document outlines the licenses for the source code of the CleanSpeech project, as well as the licenses and attribution for the datasets, pre-trained models, and external APIs it relies on. We are grateful to the creators and maintainers of these resources for making their work available.

---

## 1. Project Code License

The source code for the **CleanSpeech** project is licensed under the MIT License. You are free to use, modify, and distribute the code for any purpose, provided you include the original copyright and license notice in any copy of the software.

```text
MIT License

Copyright (c) 2024 [DS LAB TERM3 2025 TEAM 10)]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Models, APIs, and External Libraries

### 2.1 Core Classifier Model: mDeBERTa-v3

The fine-tuned toxicity classifier is based on the `microsoft/mdeberta-v3-base` model.

*   **Model:** `microsoft/mdeberta-v3-base`
*   **License:** MIT License
*   **Source:** [Hugging Face Model Hub](https://huggingface.co/microsoft/mdeberta-v3-base)
*   **Citation:**
    ```bibtex
    @inproceedings{
        he2021deberta,
        title={{DEBERTA}: Decoding-enhanced {BERT} with Disentangled Attention},
        author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
        booktitle={International Conference on Learning Representations},
        year={2021},
        url={https://openreview.net/forum?id=XPZIaotutsD}
    }
    ```

### 2.2 Rewriting Module: Google Gemini API

The constructive text rewriting is performed using the Google Gemini family of models.

*   **Service:** Google Gemini API
*   **Terms of Use:** Use of the Gemini API is subject to the [Google Cloud Platform Terms of Service](https://cloud.google.com/terms) and the [Generative AI Service Specific Terms](https://cloud.google.com/terms/service-terms/ai-generative-service).
*   **Source:** [Google AI for Developers](https://ai.google.dev/)

---

## 3. Datasets

### 3.1 Jigsaw Toxic Comment Classification Challenge

This dataset was the primary corpus for training and validating the toxicity detection model.

*   **Dataset:** Jigsaw Toxic Comment Classification Challenge
*   **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
*   **Source:** [Kaggle Competition Page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
*   **Citation:**
    ```bibtex
    @inproceedings{wulczyn2017ex,
      title={Ex machina: Personal attacks seen at scale},
      author={Wulczyn, Ellery and Thain, Nithum and Dixon, Lucas},
      booktitle={Proceedings of the 26th International Conference on World Wide Web},
      pages={1391--1399},
      year={2017}
    }
    ```

