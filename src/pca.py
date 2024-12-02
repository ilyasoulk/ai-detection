from datasets import load_dataset
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
        n_components = 512
    elif embedding_type == "sentence_transformer":
        embeddings = sentence_transformer_emb(texts, embedding_model)
        n_components = 768
    else:
        raise ValueError("Invalid embedding type. Choose 'word2vec' or 'sentence_transformer'.")

    threshold = 0.95
    dimensions = []
    for samples in embeddings:
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
    human_dataset_path = "abisee/cnn_dailymail"
    ai_dataset_path = "HuggingFaceTB/cosmopedia-100k"
    word2vec_path = "glove-twitter-25"
    sentence_transformers_path = "multi-qa-mpnet-base-dot-v1"
    word_transformer_path = "bert-base-uncased"

    ai_ds = load_dataset(ai_dataset_path, split="train")
    hm_ds = load_dataset(human_dataset_path, '3.0.0', split="train")
    ai_ds = ai_ds["text"]
    hm_ds = hm_ds["article"]

    # intrinsic_dimension(human_dataset_path, sentence_transformers_path, "sentence_transformer", elements=100)
    avg_dim_ai_bert = intrinsic_dimension(ai_ds, word_transformer_path, "token_transformer")
    avg_dim_human_bert = intrinsic_dimension(hm_ds, word_transformer_path, "token_transformer")
    avg_dim_human_word2vec = intrinsic_dimension(ai_ds, word2vec_path, "word2vec")
    avg_dim_ai_word2vec = intrinsic_dimension(hm_ds, word2vec_path, "word2vec")


    print("Results")
    print(f"Average dimension using bert on synthetic dataset {ai_dataset_path.split('/')[-1]} : {avg_dim_ai_bert}")
    print(f"Average dimension using bert on human dataset {human_dataset_path.split('/')[-1]}: {avg_dim_human_bert}")
    print(f"Average dimension using word2vec on synthetic dataset : {avg_dim_ai_word2vec}")
    print(f"Average dimension using word2vec on human dataset : {avg_dim_human_word2vec}")

# Results

# Average dimension using bert on synthetic dataset cosmopedia-100k : 208.24
# Average dimension using bert on human dataset imdb : 169.36
# Average dimension using word2vec on synthetic dataset cosmopedia-100k : 19.2
# Average dimension using word2vec on human dataset imdb : 17.55

# Average dimension using bert on synthetic dataset cosmopedia-100k : 208.24
# Average dimension using bert on human dataset cnn_dailymail: 213.11
# Average dimension using word2vec on synthetic dataset cosmopedia-100k : 18.69
# Average dimension using word2vec on human dataset cnn_dailymail : 19.2
