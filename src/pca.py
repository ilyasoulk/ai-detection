from datasets import load_dataset
import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA


def preprocess_text(text):
    tokens = text.split()
    return tokens


def embed_text(text, model):
    tokens = preprocess_text(text)
    vectors = [model[token] for token in tokens if token in model]
    return np.array(vectors)


def intrinsic_dimension(dataset_path, embed, threshold=0.95):
    dataset = load_dataset(dataset_path)
    threshold = 0.95
    dimensions = []
    for sample in dataset['train'].select(range(10)):
        vectors = embed_text(sample["text"], embed)
        prev = len(dimensions)
        pca = PCA(n_components=25)
        pca.fit(vectors)
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
            dimensions.append(25)


    dimensions = np.array(dimensions)
    return np.mean(dimensions)


if __name__ == "__main__":
    human_dataset_path = "imdb"
    ai_dataset_path = "HuggingFaceTB/cosmopedia-100k"
    word2vec_path = "glove-twitter-25"
    model = api.load(word2vec_path)
    i_dim_human = intrinsic_dimension(human_dataset_path, model)
    i_dim_ai = intrinsic_dimension(ai_dataset_path, model)
    print(i_dim_human, i_dim_ai)

