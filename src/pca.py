from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import nltk
from embed import (
    word2vec_emb,
    token_transformer_emb,
    sentence_transformer_emb
)


def intrinsic_dimension(texts, embedding_model, embedding_type, threshold=0.95, elements=100):
    # Compute embeddings based on the specified embedding type
    texts = texts[:elements]
    if embedding_type == "word2vec":
        embeddings = word2vec_emb(texts, embedding_model)
        n_components = 25
    elif embedding_type == "token_transformer":
        embeddings = token_transformer_emb(texts, embedding_model)
        n_components = 200
    elif embedding_type == "sentence_transformer":
        embeddings = sentence_transformer_emb(texts, embedding_model)
        n_components = 768
    else:
        raise ValueError("Invalid embedding type. Choose 'word2vec' or 'sentence_transformer'.")

    threshold = 0.95
    dimensions = []
    for samples in tqdm(embeddings):
        samples = np.array(samples)
        prev = len(dimensions)
        pca = PCA(n_components=n_components)
        pca.fit(samples)
        explained_variance = pca.explained_variance_ratio_
        cumul_var = 0
        dim = 0
        for variance in explained_variance:
            cumul_var += variance
            dim += 1

            if cumul_var > threshold:
                dimensions.append(dim)
                break

        if prev == len(dimensions):
            dimensions.append(n_components)


    dimensions = np.array(dimensions)
    return np.mean(dimensions)


if __name__ == "__main__":
    nltk.download("punkt")
    dataset_path = "ilyasoulk/ai-vs-human"
    word_transformer_path = "bert-base-uncased"

    ds = load_dataset(dataset_path, split="train")
    ai_ds = ds["ai"]
    hm_ds = ds["human"]

    # intrinsic_dimension(human_dataset_path, sentence_transformers_path, "sentence_transformer", elements=100)
    avg_dim_ai_bert = intrinsic_dimension(ai_ds, word_transformer_path, "token_transformer")
    avg_dim_human_bert = intrinsic_dimension(hm_ds, word_transformer_path, "token_transformer")


    print("Results")
    print(f"Average dimension using bert on synthetic dataset {dataset_path.split('/')[-1]} : {avg_dim_ai_bert}")
    print(f"Average dimension using bert on human dataset {dataset_path.split('/')[-1]}: {avg_dim_human_bert}")

# Results
# Average dimension using bert on synthetic dataset ai-vs-human : 153.73
# Average dimension using bert on human dataset ai-vs-human: 196.17
