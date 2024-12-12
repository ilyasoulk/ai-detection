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


class PCAAnalyzer:
    def __init__(self, embedding_model, min_tokens=150, thresholds=[0.8, 0.9, 0.95]):
        self.embedding_model = embedding_model
        self.thresholds = thresholds
        self.stats = None
        self.features = None
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.embedder = AutoModel.from_pretrained(embedding_model)
        self.min_tokens = min_tokens

    def find_min_tokens(self, texts):
        """Compute the minimum number of tokens across all texts"""
        token_lengths = []
        for text in tqdm(texts, desc="Computing min tokens"):
            tokens = self.tokenizer.encode(text)
            token_lengths.append(len(tokens))
        self.min_tokens = min(token_lengths)
        print(f"Minimum number of tokens found: {self.min_tokens}")
        return self.min_tokens

    def embed(self, texts):
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.embedder(**tokens)

        embeddings = outputs.last_hidden_state

        return embeddings

    def compute_intrinsic_dimensions(self, text):
        # Check if text has enough tokens
        tokens = self.tokenizer.encode(text)
        if len(tokens) < self.min_tokens:
            raise ValueError(f"Text has fewer than {self.min_tokens} tokens")

        # Get embeddings and truncate to min_tokens
        embeddings = self.embed([text])[0]
        embeddings = np.array(embeddings[: self.min_tokens])

        # Compute PCA with fixed n_components
        pca = PCA(n_components=self.min_tokens)
        pca.fit(embeddings)
        explained_variance = pca.explained_variance_ratio_

        results = {}
        for threshold in self.thresholds:
            cumul_var = 0
            dim = 0
            for variance in explained_variance:
                cumul_var += variance
                dim += 1
                if cumul_var > threshold:
                    break

            results[f"dim_t{threshold}"] = dim
            results[f"var_t{threshold}"] = cumul_var

        return results

    def fit(self, X, y):
        """
        X: array-like of shape (n_samples,) containing text samples
        y: array-like of shape (n_samples,) containing labels (0 for human, 1 for AI)
        """
        all_results = []
        skipped = 0

        for text, label in tqdm(zip(X, y), total=len(X), desc="Processing texts"):
            try:
                results = self.compute_intrinsic_dimensions(text)
                results["type"] = "ai" if label == 1 else "human"
                results["text_length"] = len(text)
                all_results.append(results)
            except ValueError as e:
                skipped += 1
                continue
            except Exception as e:
                print(f"Error processing text: {e}")
                skipped += 1
                continue

        print(f"Skipped {skipped} texts with fewer than {self.min_tokens} tokens")
        if not all_results:
            raise ValueError("No valid texts found with sufficient tokens")

        df = pd.DataFrame(all_results)

        # Compute statistics for prediction
        stats = {}
        features = [f"dim_t{t}" for t in self.thresholds] + [
            f"var_t{t}" for t in self.thresholds
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

        results = self.compute_intrinsic_dimensions(text)
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

        for text, label in tqdm(zip(X, y), total=len(X), desc="Evaluating texts"):
            try:
                pred = self.predict_text(text)
                predictions.append(pred)
                ground_truth.append(label)
            except ValueError as e:
                skipped += 1
                continue
            except Exception as e:
                print(f"Error processing text: {e}")
                skipped += 1
                continue

        print(f"Skipped {skipped} texts with fewer than {self.min_tokens} tokens")

        if not predictions:
            raise ValueError("No valid texts found with sufficient tokens")

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

    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_path, split="train")

    # Split into train and validation splits
    dataset = split_dataset_random(dataset)

    # Transform into text, class dataset
    train_set = transform_paired_dataset(dataset["train"])
    val_set = transform_paired_dataset(dataset["validation"])

    # Initialize analyzer
    analyzer = PCAAnalyzer(args.embedding_model)

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
    print(f"\nSkipped {results['skipped_samples']} samples due to insufficient tokens")
