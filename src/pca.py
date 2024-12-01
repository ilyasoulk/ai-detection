from datasets import load_dataset
import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer
import nltk


def preprocess_text(text):
    tokens = text.split()
    return tokens


def word2vec_emb(texts, embedding_model):
    model = api.load(embedding_model)
    res = []
    for text in texts:
        tokens = preprocess_text(text)
        vectors = torch.Tensor([model[token] for token in tokens if token in model])
        res.append(vectors)
    return res


def token_transformer_emb(texts, embedding_model):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    embedder = AutoModel.from_pretrained(embedding_model)

    tokens = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = embedder(**tokens)

    embeddings = outputs.last_hidden_state

    return embeddings


def sentence_transformer_emb(texts, embedding_model):
    split_texts = [nltk.sent_tokenize(text) for text in texts]

    embedder = SentenceTransformer(embedding_model)
    
    grouped_embeddings = [
        embedder.encode(sentences)
        for sentences in split_texts
    ]

    return grouped_embeddings


def intrinsic_dimension(dataset_path, embedding_model, embedding_type, threshold=0.95, elements=100):
    ds = load_dataset(dataset_path, split="train")
    texts = ds["text"][:elements]

    # Compute embeddings based on the specified embedding type
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
    human_dataset_path = "imdb"
    ai_dataset_path = "HuggingFaceTB/cosmopedia-100k"
    word2vec_path = "glove-twitter-25"
    sentence_transformers_path = "multi-qa-mpnet-base-dot-v1"
    word_transformer_path = "bert-base-uncased"
    # intrinsic_dimension(human_dataset_path, sentence_transformers_path, "sentence_transformer", elements=100)
    avg_dim_ai_bert = intrinsic_dimension(ai_dataset_path, word_transformer_path, "token_transformer")
    avg_dim_human_bert = intrinsic_dimension(human_dataset_path, word_transformer_path, "token_transformer")
    avg_dim_human_word2vec = intrinsic_dimension(human_dataset_path, word2vec_path, "word2vec")
    avg_dim_ai_word2vec = intrinsic_dimension(ai_dataset_path, word2vec_path, "word2vec")


    print("Results")
    print(f"Average dimension using bert on synthetic dataset : {avg_dim_ai_bert}")
    print(f"Average dimension using bert on human dataset : {avg_dim_human_bert}")
    print(f"Average dimension using word2vec on synthetic dataset : {avg_dim_ai_word2vec}")
    print(f"Average dimension using word2vec on human dataset : {avg_dim_human_word2vec}")

# Results
# Average dimension using bert on synthetic dataset : 208.24
# Average dimension using bert on human dataset : 169.36
# Average dimension using word2vec on synthetic dataset : 19.2
# Average dimension using word2vec on human dataset : 17.55
