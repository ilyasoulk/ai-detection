# The goal is to compute the rank of the ngram matrix of each sample
# We will try to see if there is a relation between AI/Human generated text and n_gram rank
from sklearn.feature_extraction.text import CountVectorizer
from nltk import sent_tokenize
import nltk
import numpy as np
from datasets import load_dataset

def truncate_to_n_sentences(text, n_sentences=10):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:n_sentences])


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
    dataset_path = "ilyasoulk/ai-vs-human"
    n_elements = 100


    ds = load_dataset(dataset_path, split="train")
    ai_ds = ds["ai"][:n_elements]
    hm_ds = ds["human"][:n_elements]

    ai_ranks, total_len_ai = 0, 0
    hm_ranks, total_len_hm = 0, 0
    n_sentences = 10
    for hm, ai in zip(hm_ds, ai_ds):
        ai_truncated = truncate_to_n_sentences(ai, n_sentences)
        hm_truncated = truncate_to_n_sentences(hm, n_sentences)

        ai_rank, len_ai = get_ngram_rank(ai_truncated)
        hm_rank, len_hm = get_ngram_rank(hm_truncated)

        ai_ranks += ai_rank
        hm_ranks += hm_rank
        total_len_ai += len_ai
        total_len_hm += len_hm

    print(f"Average rank of ngram matrix for AI dataset : {dataset_path.split('/')[-1]} = {ai_ranks / n_elements}")
    print(f"Average rank of ngram matrix for Human dataset : {dataset_path.split('/')[-1]} = {hm_ranks / n_elements}")
    print(f"Average length of {dataset_path.split('/')[-1]} : {total_len_ai / n_elements}")
    print(f"Average length of {dataset_path.split('/')[-1]} : {total_len_hm / n_elements}")


# Average rank of ngram matrix for AI dataset : ai-vs-human = 8.49
# Average rank of ngram matrix for Human dataset : ai-vs-human = 9.86
# Average length of ai-vs-human : 8.55
# Average length of ai-vs-human : 9.95


if __name__ == "__main__":
    main()
