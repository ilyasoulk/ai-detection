from datasets import load_dataset
from gensim.models import KeyedVectors
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
    for sample in dataset['train'].select(range(1000)):
        vectors = embed_text(sample["text"], embed)
        prev = len(dimensions)
        for i in range(2, 25):
            pca = PCA(n_components=i)
            pca.fit(vectors)
            explained_variance = np.sum(pca.explained_variance_ratio_)
            if explained_variance > threshold:
                dimensions.append(i)
                break

        if prev == len(dimensions):
            dimensions.append(25)


    dimensions = np.array(dimensions)
    return np.mean(dimensions)


if __name__ == "__main__":
    model = api.load("glove-twitter-25")
    i_dim_human = intrinsic_dimension("imdb", model)
    i_dim_ai = intrinsic_dimension("HuggingFaceTB/cosmopedia-100k", model)
    print(i_dim_human, i_dim_ai)

