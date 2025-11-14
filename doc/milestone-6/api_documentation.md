# **CleanSpeech Toxicity Classifier API Documentation**

## 1. Overview

Welcome to the CleanSpeech API. This service provides real-time, multi-label toxicity classification and explainability for text content. You can use this API to detect six different types of toxicity and understand *why* the model made a specific prediction by analyzing word-level contributions.

The API is powered by a fine-tuned `mDeBERTa-v3-base` model, which offers strong contextual understanding and multilingual capabilities.

## 2. Base URL

All API endpoints are relative to the following base URL, hosted on Hugging Face Spaces:

```
https://milanchndr-Toxic-Comment-Classifier-Explainer.hf.space
```

### Authentication

The API is public and does not require an authentication token or API key for access.

---

## 3. Endpoint: Analyze Text

This is the primary endpoint for submitting text and receiving a full toxicity analysis.

### `POST /predict`

Analyzes a given text string and returns a detailed JSON object containing toxicity probabilities, token-level explanations, and attention scores.

#### Request Body

The request body must be a JSON object with a single key, `text`.

**Format:**
```json
{
  "text": "Your text to be analyzed goes here."
}
```

**Parameters:**

| Parameter | Type   | Required | Description                                  |
| :-------- | :----- | :------- | :------------------------------------------- |
| `text`    | string | Yes      | The input text string you want to classify. |

#### Response Body

The API returns a detailed JSON object. On success, the HTTP status code will be `200 OK`.

**Example Response:**
```json
{
    "input_text": "You are an idiot!",
    "probabilities": {
        "toxic": 0.981,
        "severe_toxic": 0.152,
        "obscene": 0.853,
        "threat": 0.011,
        "insult": 0.954,
        "identity_hate": 0.057
    },
    "tokens": [
        " ",
        " you",
        " are",
        " an",
        " idiot",
        "!"
    ],
    "attention": {
        "tokens": [
            " ", " you", " are", " an", " idiot", "!"
        ],
        "matrix": [
            [0.1, 0.2, ...],
            [0.3, 0.4, ...],
            ...
        ]
    },
    "word_importance": {
        "label": "toxic",
        "tokens": [
            "you",
            "are",
            "an",
            "idiot"
        ],
        "importance_scores": [
            0.012,
            -0.005,
            0.001,
            0.758
        ],
        "base_value": 0.045,
        "prediction": 0.981
    }
}
```

#### Response Field Descriptions

| Key                 | Type          | Description                                                                                                                                                                                 |
| :------------------ | :------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `input_text`        | string        | The original text that was submitted in the request.                                                                                                                                        |
| `probabilities`     | object        | A dictionary mapping each of the six toxicity labels to its predicted probability score (a float between 0.0 and 1.0).                                                                        |
| `tokens`            | array         | The input text as processed by the model's tokenizer. Note the leading space ` ` for subword tokenization.                                                                                   |
| `attention`         | object        | Contains data from the model's last self-attention layer. Useful for advanced analysis.                                                                                                     |
| ↳ `tokens`          | array         | The tokens corresponding to the attention matrix dimensions.                                                                                                                                |
| ↳ `matrix`          | 2D array      | An `NxN` matrix of attention scores, where `N` is the number of tokens. Each value indicates how much a token attends to other tokens.                                                      |
| `word_importance`   | object        | The core explainability output, based on a word perturbation method.                                                                                                                        |
| ↳ `label`           | string        | The name of the label being explained (typically the one with the highest probability).                                                                                                     |
| ↳ `tokens`          | array         | The words from the cleaned input text.                                                                                                                                                      |
| ↳ `importance_scores`| array         | An array of floats, one for each word. A **positive** score means the word increases toxicity. A **negative** score means it decreases toxicity. The magnitude indicates the contribution. |
| ↳ `base_value`      | float         | The model's baseline prediction for an empty input. This is the starting point for the explanation.                                                                                        |
| ↳ `prediction`      | float         | The final prediction score for the explained `label`. The sum of `base_value` and all `importance_scores` should approximate this value.                                                   |

---

## 4. Example Usage

You can easily test the API using `curl` from your terminal.

### Example `curl` Request

```bash
curl -X POST "https://milanchndr-Toxic-Comment-Classifier-Explainer.hf.space/predict" \
-H "Content-Type: application/json" \
-d '{"text": "You are an idiot and I hate you!"}'
```

This command sends the text "You are an idiot and I hate you!" to the `/predict` endpoint and will return the full JSON analysis as described above.

## 5. Status Codes

The API uses standard HTTP status codes to indicate the success or failure of a request.

| Code | Meaning                 | Description                                                                 |
| :--- | :---------------------- | :-------------------------------------------------------------------------- |
| `200`| **OK**                  | The request was successful, and the analysis is in the response body.       |
| `422`| **Unprocessable Entity**| The request body is malformed. This usually means the JSON is invalid or the `text` key is missing. |
| `500`| **Internal Server Error** | An unexpected error occurred on the server while processing the request.    |
