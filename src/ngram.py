# The goal is to compute the rank of the ngram matrix of each sample
# We will try to see if there is a relation between AI/Human generated text and n_gram rank
from sklearn.feature_extraction.text import CountVectorizer
from nltk import sent_tokenize
import nltk
import numpy as np
from datasets import load_dataset
import argparse

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm


class TextAnalyzer:
    def __init__(self, n_range=[2, 3, 4], threshold=1e-10):
        self.n_range = n_range
        self.threshold = threshold
        self.stats = None
        self.features = None

    def create_ngram_matrix(self, text, n):
        corpus = sent_tokenize(text)
        vectorizer = CountVectorizer(ngram_range=(n, n))
        ngram_matrix = vectorizer.fit_transform(corpus)
        return ngram_matrix.toarray()

    def rank_with_svd(self, s):
        return np.sum(s > self.threshold * s[0])

    def analyze_text(self, text):
        results = {}
        for n in self.n_range:
            ngram_matrix = self.create_ngram_matrix(text, n)
            s = np.linalg.svd(ngram_matrix, compute_uv=False)

            results[f"rank_n{n}"] = self.rank_with_svd(s)
            results[f"largest_sv_n{n}"] = s[0]
            results[f"sv_ratio_n{n}"] = s[0] / s[-1] if len(s) > 1 else 0

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
                    z_score = (
                        abs(results[feature] - self.stats[type_]["mean"][feature])
                        / self.stats[type_]["std"][feature]
                    )
                    scores[type_] += z_score

        return 1 if scores["ai"] < scores["human"] else 0


def split_dataset_random(dataset, seed=42):
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    return {"train": split["train"], "validation": split["test"]}


def transform_paired_dataset(dataset):
    """
    Transform a dataset with 'ai' and 'human' columns into a format with 'text' and 'class' columns
    Returns a dataset with twice as many rows, where:
    - 'text' contains all texts (both AI and human)
    - 'class' contains 1 for AI-generated text and 0 for human-written text
    """
    texts = []
    labels = []

    # Add AI texts with label 1
    texts.extend(dataset["ai"])
    labels.extend([1] * len(dataset["ai"]))

    # Add human texts with label 0
    texts.extend(dataset["human"])
    labels.extend([0] * len(dataset["human"]))

    return {"text": texts, "class": labels}


if __name__ == "__main__":
    import nltk

    nltk.download("punkt")

    parser = argparse.ArgumentParser(description="Load config for dataset processing")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset or huggingface id",
    )

    args = parser.parse_args()
    # Load dataset
    dataset = load_dataset(args.dataset_path, split="train")

    # Split into train and validation splits
    dataset = split_dataset_random(dataset)

    # Transform into text, class dataset
    train_set = transform_paired_dataset(dataset["train"])
    val_set = transform_paired_dataset(dataset["validation"])

    # Initialize analyzer
    analyzer = TextAnalyzer()

    # Fit the analyzer
    analyzer.fit(train_set["text"], train_set["class"])

    # Make predictions
    predictions = analyzer.predict(val_set["text"])
    gt = val_set["class"]

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(gt, predictions))

    # Print confusion matrix
    cm = confusion_matrix(gt, predictions)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Human   AI")
    print(f"Actual Human  {cm[0][0]:6d} {cm[0][1]:8d}")
    print(f"Actual AI    {cm[1][0]:6d} {cm[1][1]:8d}")
