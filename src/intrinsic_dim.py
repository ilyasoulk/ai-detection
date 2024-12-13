from sklearn.decomposition import PCA
import torch
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils.utils import split_dataset_random, transform_paired_dataset
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


class IntrinsicDimAnalyzer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.stats = None
        self.features = None
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.embedder = AutoModel.from_pretrained(embedding_model, device_map="cuda")
        self.batch_size = 10
        self.min_tokens = 128

    def embed(self, texts):
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        tokens = {k: v.to(self.embedder.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.embedder(**tokens)

        embeddings = outputs.last_hidden_state

        return embeddings

    def compute_intrinsic_dimensions(self, embeddings: torch.Tensor) -> list: ...

    def fit(self, X, y):
        """
        X: array-like of shape (n_samples,) containing text samples
        y: array-like of shape (n_samples,) containing labels (0 for human, 1 for AI)
        """
        all_results = []
        X_filered = []
        tokens = self.tokenizer(
            X,
            max_length=self.min_tokens,
        )

        for text, tkn in zip(X, tokens["input_ids"]):
            if len(tkn) < self.min_tokens:
                continue
            X_filered.append(text)

        X = X_filered
        batch_size = self.batch_size
        for start in tqdm(range(0, len(X), batch_size), desc="Fit texts"):
            end = start + batch_size if start + batch_size < len(X) else len(X)
            batch = X[start:end]
            embeddings = self.embed(batch)
            results = self.compute_intrinsic_dimensions(embeddings)
            for i, r in enumerate(results):
                r["type"] = "ai" if y[start + i] == 1 else "human"
                r["text_length"] = len(X[start + i])
            all_results.extend(results)

        if not all_results:
            raise ValueError("No valid texts found with sufficient characters")

        df = pd.DataFrame(all_results)

        # Compute statistics for prediction
        stats = {}

        for type_ in ["ai", "human"]:
            stats[type_] = {
                "mean": df[df["type"] == type_][self.features].mean(),
                "std": df[df["type"] == type_][self.features].std(),
            }

        self.stats = stats

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

    def predict_text(self, result: dict):
        if self.stats is None:
            raise ValueError("Must compute training stats before prediction")

        scores = {"ai": 0, "human": 0}
        for type_ in ["ai", "human"]:
            if self.features is not None:
                for feature in self.features:
                    if feature in result:
                        z_score = (
                            abs(result[feature] - self.stats[type_]["mean"][feature])
                            / self.stats[type_]["std"][feature]
                        )
                        scores[type_] += z_score
        return 1 if scores["ai"] < scores["human"] else 0  # 1 for AI, 0 for human

    def evaluate(self, X, y):
        """
        Evaluate the model on test data, skipping samples with insufficient tokens

        X: array-like of shape (n_samples,) containing text samples
        y: array-like of shape (n_samples,) containing ground truth labels
        Returns: Dictionary containing evaluation metrics
        """
        predictions = []
        ground_truth = []
        skipped = 0
        X_filered = []
        tokens = self.tokenizer(
            X,
            max_length=self.min_tokens,
        )

        for text, tkn in zip(X, tokens["input_ids"]):
            if len(tkn) < self.min_tokens:
                continue
            X_filered.append(text)

        X = X_filered  

        batch_size = self.batch_size
        for start in tqdm(range(0, len(X), batch_size), desc="Evaluate texts"):
            end = start + batch_size if start + batch_size < len(X) else len(X)
            batch = X[start:end]
            embeddings = self.embed(batch)
            results = self.compute_intrinsic_dimensions(embeddings)
            for i, r in enumerate(results):
                r["text_length"] = len(X[start + i])
                predictions.append(self.predict_text(r))
                ground_truth.append(y[start + i])

        if not predictions:
            raise ValueError("No valid texts found with sufficient characters")

        # Convert to numpy arrays for sklearn metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)

        # Calculate metrics
        report = classification_report(ground_truth, predictions, output_dict=True)
        cm = confusion_matrix(ground_truth, predictions)

        return {
            "classification_report": report,
            "confusion_matrix": cm,
            "skipped_samples": skipped,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for dataset processing")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset or huggingface id",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Name or path of the embedding model",
    )

    parser.add_argument(
        "--intrinsic_dim_method",
        type=str,
        default="phd",
        help="Method to compute intrinsic dimensions",
    )

    parser.add_argument(
        "--subset_size",
        type=int,
        default=512,
        help="Number of samples to use for training and validation",
    )

    args = parser.parse_args()
    print(args)

    from pca import PCAAnalyzer
    from phd import PHDAnalyzer

    analyzers = {
        "pca": PCAAnalyzer,
        "phd": PHDAnalyzer,
    }

    # Load dataset
    dataset = load_dataset(args.dataset_path, split="train")

    subset_size = args.subset_size
    dataset = dataset.select(range(subset_size))

    # Split into train and validation splits
    dataset = split_dataset_random(dataset)

    # Transform into text, class dataset
    train_set = transform_paired_dataset(dataset["train"])
    val_set = transform_paired_dataset(dataset["validation"])

    # Initialize analyzer
    analyzer: IntrinsicDimAnalyzer = analyzers[args.intrinsic_dim_method](
        args.embedding_model
    )

    # Fit the "model"
    analyzer.fit(train_set["text"], train_set["class"])

    # Evaluation
    results = analyzer.evaluate(val_set["text"], val_set["class"])
    print("\nClassification Report:")
    print(results["classification_report"])
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Human   AI")
    print(
        f"Actual Human  {results['confusion_matrix'][0][0]:6d} {results['confusion_matrix'][0][1]:8d}"
    )
    print(
        f"Actual AI    {results['confusion_matrix'][1][0]:6d} {results['confusion_matrix'][1][1]:8d}"
    )
