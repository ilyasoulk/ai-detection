# The goal is to compute the rank of the ngram matrix of each sample
# We will try to see if there is a relation between AI/Human generated text and n_gram rank
from sklearn.feature_extraction.text import CountVectorizer
from nltk import sent_tokenize
import nltk
import numpy as np
from datasets import load_dataset


def get_ngram_rank(text, n=2):
    corpus = sent_tokenize(text)
    len_corpus = len(corpus) 
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngram_matrix = vectorizer.fit_transform(corpus)
    ngram_matrix = ngram_matrix.toarray()
    rank = np.linalg.matrix_rank(ngram_matrix)
    return rank, len_corpus


def main():
    nltk.download("punkt")
    human_dataset_path = "abisee/cnn_dailymail"
    ai_dataset_path = "HuggingFaceTB/cosmopedia-100k"
    n_elements = 100


    ai_ds = load_dataset(ai_dataset_path, split="train")
    hm_ds = load_dataset(human_dataset_path, '3.0.0', split="train")
    ai_ds = ai_ds["text"][:n_elements]
    hm_ds = hm_ds["article"][:n_elements]

    ai_ranks, total_len_ai = 0, 0
    hm_ranks, total_len_hm = 0, 0
    for hm, ai in zip(hm_ds, ai_ds):
        ai_rank, len_ai = get_ngram_rank(ai)
        hm_rank, len_hm = get_ngram_rank(hm)

        ai_ranks += ai_rank
        hm_ranks += hm_rank
        total_len_ai += len_ai
        total_len_hm += len_hm

    print(f"Average rank of ngram matrix for AI dataset : {ai_dataset_path.split('/')[-1]} = {ai_ranks / n_elements}")
    print(f"Average length of {ai_dataset_path.split('/')[-1]} : {total_len_ai / n_elements}")
    print(f"Average length of {human_dataset_path.split('/')[-1]} : {total_len_hm / n_elements}")
    print(f"Average rank of ngram matrix for Human dataset : {human_dataset_path.split('/')[-1]} = {hm_ranks / n_elements}")


# Average rank of ngram matrix for AI dataset : cosmopedia-100k = 28.66
# Average length of cosmopedia-100k : 30.84
# Average length of cnn_dailymail : 32.4
# Average rank of ngram matrix for Human dataset : cnn_dailymail = 31.98


if __name__ == "__main__":
    main()
