# The goal is to compute the rank of the ngram matrix of each sample
# We will try to see if there is a relation between AI/Human generated text and n_gram rank
from sklearn.feature_extraction.text import CountVectorizer
from nltk import sent_tokenize
import nltk
import numpy as np
from datasets import load_dataset
import argparse

import pandas as pd
from tqdm import tqdm
from utils.utils import split_dataset_random, transform_paired_dataset, evaluate


class NgramAnalyzer:
    def __init__(self, n_range=[2, 3, 4], threshold=1e-10):
        self.n_range = n_range
        self.threshold = threshold
        self.stats = None
        self.features = None

    def create_ngram_matrix(self, text, n):
        corpus = sent_tokenize(text)
        try:
            vectorizer = CountVectorizer(ngram_range=(n, n))
            ngram_matrix = vectorizer.fit_transform(corpus)
            return ngram_matrix.toarray()
        except Exception as e:
            raise ValueError(f"Ngram failed {e}")

    def rank_with_svd(self, s):
        return np.sum(s > self.threshold * s[0])

    def analyze_text(self, text):
        results = {}
        for n in self.n_range:
            try:
                ngram_matrix = self.create_ngram_matrix(text, n)
                s = np.linalg.svd(ngram_matrix, compute_uv=False)

                results[f"rank_n{n}"] = self.rank_with_svd(s)
                results[f"largest_sv_n{n}"] = s[0]
                results[f"sv_ratio_n{n}"] = s[0] / s[-1] if len(s) > 1 else 0
            finally:
                continue

        return results

    def fit(self, X, y):
        """
        X: array-like of shape (n_samples,) containing text samples
        y: array-like of shape (n_samples,) containing labels (0 for human, 1 for AI)
        """
        all_results = []

        for text, label in tqdm(zip(X, y), total=len(X)):
            results = self.analyze_text(text)
            results["type"] = "ai" if label == 1 else "human"
            results["text_length"] = len(text)
            all_results.append(results)

        df = pd.DataFrame(all_results)

        # Compute statistics for prediction
        stats = {}
        features = [f"rank_n{n}" for n in self.n_range] + [
            f"largest_sv_n{n}" for n in self.n_range
        ]

        for type_ in ["ai", "human"]:
            stats[type_] = {
                "mean": df[df["type"] == type_][features].mean(),
                "std": df[df["type"] == type_][features].std(),
            }

        self.stats = stats
        self.features = features

        return self

    def predict(self, X):
        """
        X: array-like of shape (n_samples,) containing text samples
        Returns: array of predictions (0 for human, 1 for AI)
        """
        if self.stats is None:
            raise ValueError("Must call fit before predict")

        predictions = []
        for text in tqdm(X):
            predictions.append(self.predict_text(text))

        return np.array(predictions)

    def predict_text(self, text):
        if self.stats is None:
            raise ValueError("Must compute training stats before prediction")

        results = self.analyze_text(text)
        results["text_length"] = len(text)

        scores = {"ai": 0, "human": 0}
        for type_ in ["ai", "human"]:
            if self.features is not None:
                for feature in self.features:
                    if feature in results:
                        z_score = (
                            abs(results[feature] - self.stats[type_]["mean"][feature])
                            / self.stats[type_]["std"][feature]
                        )
                        scores[type_] += z_score

        return 1 if scores["ai"] < scores["human"] else 0


if __name__ == "__main__":
    import nltk

    nltk.download("punkt_tab")

    parser = argparse.ArgumentParser(description="Load config for dataset processing")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset or huggingface id",
    )
    parser.add_argument("--testset_path", type=str, help="Dataset to run the test on")

    args = parser.parse_args()
    # Load dataset
    dataset = load_dataset(args.dataset_path, split="train")

    # Split into train and validation splits
    if args.testset_path is None:
        dataset = split_dataset_random(dataset)

        # Transform into text, class dataset
        train_set = transform_paired_dataset(dataset["train"])
        test_set = transform_paired_dataset(dataset["validation"])

    else:
        train_set = transform_paired_dataset(dataset)

        test_set = load_dataset(args.testset_path, split="train")
        test_set = transform_paired_dataset(test_set)

    # Initialize analyzer
    analyzer = NgramAnalyzer()

    # Fit the analyzer
    analyzer.fit(train_set["text"], train_set["class"])

    # Make predictions
    predictions = analyzer.predict(test_set["text"])
    gt = test_set["class"]

    # Evaluate the model
    evaluate(gt, predictions)
