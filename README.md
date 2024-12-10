# AI vs Human Text Detection using Intrinsic Dimension

## Project Overview
This project aims to detect whether a given text has been generated by an AI model or written by a human. The detection leverages the concept of intrinsic dimension through two primary methods:

1. **N-gram Rank Analysis**
   - This method analyzes the ranks of n-grams (2-grams, 3-grams, 4-grams) in the text.
   - AI-generated texts often exhibit lower-ranked n-grams due to the constraints of the model's `lm_head` embedding dimension and vocabulary size, which limit the model’s capacity to explore diverse combinations of tokens.
   - In contrast, human-written text does not face these constraints, leading to higher-ranked n-grams.

2. **Embedding-based Dimensionality Analysis** (Work in Progress)
   - This approach uses embedding models (e.g., BERT, GPT) to analyze the text's intrinsic dimensionality.
   - By performing Principal Component Analysis (PCA) on the embeddings, we measure the number of dimensions needed to explain a high variance threshold (e.g., 95%).
   - AI-generated texts typically exhibit lower intrinsic dimensions compared to human-written texts due to the structured nature of their generation.

## Dataset Generation
To create synthetic datasets for training and evaluation, we used the CNN/Daily Mail dataset. The process involved:

1. **Text Truncation:** Randomly truncating human-written articles.
2. **Completion:** Asking various language models to generate completions for the truncated articles.
3. **Labeling:** Retaining and labeling these completions as either human (0) or AI (1).

Different models were used for data generation, and the datasets are named accordingly. You can explore the datasets [here on Hugging Face](https://huggingface.co/collections/zcamz/ai-vs-human-6754d445b3826df8fd547c0e).

## Results Summary
Below is a quick overview of the evaluation results for the n-gram method on various datasets:

| Dataset Name                               | Accuracy | F1-Score (Class 0) | F1-Score (Class 1) |
|-------------------------------------------|----------|--------------------|--------------------|
| `zcamz/ai-vs-human-google-gemma-2-2b-it`  | 71%      | 72%                | 69%                |
| `zcamz/ai-vs-human-Qwen-Qwen2.5-1.5B`     | 69%      | 63%                | 73%                |
| `zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-360M` | 89% | 89%                | 89%                |
| `zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-1.7B` | 75% | 75%                | 76%                |
| `zcamz/ai-vs-human-meta-llama-Llama-3.2-1B` | 64% | 65%                | 63%                |

### Key Insights
- The `HuggingFaceTB-SmolLM2-360M` dataset achieved the highest accuracy of 89%, showcasing the potential of the n-gram method on smaller model outputs.
- Performance varies across datasets, influenced by the size and characteristics of the language models used for text generation.

### Detailed Results
<details>
<summary>Click to expand results for `zcamz/ai-vs-human-google-gemma-2-2b-it`</summary>

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.69      0.76      0.72      1000
           1       0.73      0.65      0.69      1000

    accuracy                           0.71      2000
   macro avg       0.71      0.71      0.71      2000
weighted avg       0.71      0.71      0.71      2000
```
**Confusion Matrix:**
```
                 Predicted
                 Human   AI
Actual Human     759      241
Actual AI        347      653
```

</details>

<details>
<summary>Click to expand results for `zcamz/ai-vs-human-Qwen-Qwen2.5-1.5B`</summary>

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.77      0.54      0.63      1000
           1       0.64      0.83      0.73      1000

    accuracy                           0.69      2000
   macro avg       0.70      0.69      0.68      2000
weighted avg       0.70      0.69      0.68      2000
```
**Confusion Matrix:**
```
                 Predicted
                 Human   AI
Actual Human     539      461
Actual AI        165      835
```

</details>

<details>
<summary>Click to expand results for `zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-360M`</summary>

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.90      0.87      0.89      1000
           1       0.88      0.91      0.89      1000

    accuracy                           0.89      2000
   macro avg       0.89      0.89      0.89      2000
weighted avg       0.89      0.89      0.89      2000
```
**Confusion Matrix:**
```
                 Predicted
                 Human   AI
Actual Human     874      126
Actual AI         92      908
```

</details>

<details>
<summary>Click to expand results for `zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-1.7B`</summary>

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.77      0.73      0.75      1000
           1       0.74      0.78      0.76      1000

    accuracy                           0.75      2000
   macro avg       0.75      0.75      0.75      2000
weighted avg       0.75      0.75      0.75      2000
```
**Confusion Matrix:**
```
                 Predicted
                 Human   AI
Actual Human     731      269
Actual AI        223      777
```

</details>

<details>
<summary>Click to expand results for `zcamz/ai-vs-human-meta-llama-Llama-3.2-1B`</summary>

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.63      0.66      0.65      1000
           1       0.64      0.62      0.63      1000

    accuracy                           0.64      2000
   macro avg       0.64      0.64      0.64      2000
weighted avg       0.64      0.64      0.64      2000
```
**Confusion Matrix:**
```
                 Predicted
                 Human   AI
Actual Human     658      342
Actual AI        382      618
```

</details>

## Running the Project
### N-gram Analysis
To run the n-gram analysis:
```bash
uv sync
source .venv/bin/activate
python src/ngram.py --dataset_path [DATASETPATH]
```

### Generating Data
1. Create a config file in the `config` directory for your chosen model.
2. Add a line in the `Makefile`:
   ```bash
   python3 ./src/dataset/generate.py ./config/[MODEL_NAME_CONFIG].yaml
   ```
3. Run the following commands:
   ```bash
   make generate_[MODEL_NAME]
   make generate_[MODEL_name]
   ```

## Embedding-based Dimensionality Analysis
This method is under development. Future work will focus on:

- Analyzing embeddings from different models using PCA.
- Comparing the number of dimensions required for AI vs. human text to explain 95% variance.
- Evaluating this method on synthetic datasets.

## Future Directions
- [x] Refine the n-gram analysis.
- [ ] Complete and validate the embedding-based dimensionality method.
- [ ] Expand datasets to include more diverse models and text styles.

---
**Hugging Face Datasets Collection:** [Link](https://huggingface.co/collections/zcamz/ai-vs-human-6754d445b3826df8fd547c0e)

