# AI vs Human Text Detection using Intrinsic Dimension

## Project Overview
This project aims to detect whether a given text has been generated by an AI model or written by a human. The detection leverages the concept of intrinsic dimension through three primary methods:

1. **N-gram Rank Analysis**
   - This method analyzes the ranks of n-grams (2-grams, 3-grams, 4-grams) in the text.
   - AI-generated texts often exhibit lower-ranked n-grams due to the constraints of the model's `lm_head` embedding dimension and vocabulary size, which limit the model's capacity to explore diverse combinations of tokens.
   - In contrast, human-written text does not face these constraints, leading to higher-ranked n-grams.

2. **Embedding-based Dimensionality Analysis (PCA)**
   - This approach uses embedding models (e.g., BERT, GPT) to analyze the text's intrinsic dimensionality.
   - By performing Principal Component Analysis (PCA) on the embeddings, we measure the number of dimensions needed to explain a high variance threshold (95%).
   - AI-generated texts typically exhibit lower intrinsic dimensions compared to human-written texts due to the structured nature of their generation.

3. **RoBERTa-based Classification**
   - A fine-tuned RoBERTa model trained to distinguish between AI and human-written text.
   - Provides high precision for AI text detection and high recall for human text identification.

## Dataset Generation
To create synthetic datasets for training and evaluation, we used the CNN/Daily Mail dataset. The process involved:

1. **Text Truncation:** Randomly truncating human-written articles.
2. **Completion:** Asking various language models to generate completions for the truncated articles.
3. **Labeling:** Retaining and labeling these completions as either human (0) or AI (1).

Different models were used for data generation, and the datasets are named accordingly. You can explore the datasets [here on Hugging Face](https://huggingface.co/collections/zcamz/ai-vs-human-6754d445b3826df8fd547c0e).
## Project Structure

```plaintext
├── config/                 # Model configuration files for text generation
├── prompts/               # System prompts used for text generation
└── src/
    ├── pca.py            # PCA-based intrinsic dimension analysis and evaluation
    ├── ngram.py          # N-gram rank analysis and evaluation
    ├── pretrained.py     # RoBERTa-based classification
    └── dataset/
        ├── generate.py   # Dataset generation pipeline
        └── push_hf.py    # Utility to push datasets to Hugging Face
```

Each component serves a specific purpose:
- `config/`: Contains YAML configuration files for different models used in text generation
- `prompts/`: Stores system prompts that guide the AI models during text generation
- `src/`: Main source code directory
  - `pca.py`: Implements PCA-based intrinsic dimensionality analysis
  - `ngram.py`: Implements n-gram rank analysis method
  - `pretrained.py`: Handles RoBERTa model fine-tuning and evaluation
  - `dataset/`: Dataset management
    - `generate.py`: Orchestrates the dataset generation pipeline
    - `push_hf.py`: Handles dataset upload to Hugging Face Hub

## Results Summary

### N-gram Analysis
| Dataset Name                               | Accuracy | F1-Score (Class 0) | F1-Score (Class 1) |
|-------------------------------------------|----------|--------------------|--------------------|
| `zcamz/ai-vs-human-google-gemma-2-2b-it`  | 71%      | 72%                | 69%                |
| `zcamz/ai-vs-human-Qwen-Qwen2.5-1.5B`     | 69%      | 63%                | 73%                |
| `zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-360M` | 89% | 89%                | 89%                |
| `zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-1.7B` | 75% | 75%                | 76%                |
| `zcamz/ai-vs-human-meta-llama-Llama-3.2-1B` | 64% | 65%                | 63%                |

### Zero-shot Domain Transfer Evaluation
When validating on a new domain dataset (OpenWebText) generated by a larger model (Llama-3.1-8B):
- Accuracy: 57%
- F1-Score (Human): 0.61
- F1-Score (AI): 0.51

This evaluation demonstrates the method's generalization capabilities across domains and models not seen during rank computation.

### PCA-based Analysis
The PCA analysis was performed with the following parameters:
- Minimum tokens per sample: 150
- Number of components: 150
- Variance threshold: 95%

Results across datasets:
| Dataset | Accuracy | Human Precision | AI Precision |
|---------|----------|-----------------|--------------|
| Qwen-2.5-1.5B | 62.4% | 63.4% | 61.4% |
| SmolLM2-360M | 64.6% | 81.3% | 46.8% |
| SmolLM2-1.7B | 67.4% | 69.1% | 65.6% |
| Gemma-2-2b | 65.3% | 66.3% | 64.5% |
| Llama-3.2-1B | 58.1% | 60.0% | 56.3% |

### RoBERTa Fine-tuned Model
Results on SmolLM2-1.7B dataset:
- Accuracy: 76%
- F1-Score (Human): 0.75
- F1-Score (AI): 0.78
- Notable high recall for human text (98%) and high precision for AI text (99%)

### Key Insights
- The `HuggingFaceTB-SmolLM2-360M` dataset achieved the highest accuracy of 89% with the n-gram method.
- PCA-based analysis shows consistent performance across different models, with accuracies ranging from 58% to 67%.
- RoBERTa fine-tuning achieves strong results with particularly high precision for AI text detection.
- Zero-shot transfer to new domains remains challenging but shows promising results.
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
- [x] Complete and validate the embedding-based dimensionality method.
- [x] Expand datasets to include more diverse models and text styles.

---
**Hugging Face Datasets Collection:** [Link](https://huggingface.co/collections/zcamz/ai-vs-human-6754d445b3826df8fd547c0e)

