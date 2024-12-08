# The goal is to compute the rank of the ngram matrix of each sample
# We will try to see if there is a relation between AI/Human generated text and n_gram rank
from sklearn.feature_extraction.text import CountVectorizer
from nltk import sent_tokenize
import nltk
import numpy as np
from datasets import load_dataset
import argparse

from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from nltk import sent_tokenize
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

class TextAnalyzer:
    def __init__(self, n_range=[2, 3, 4], threshold=1e-10):
        self.n_range = n_range
        self.threshold = threshold
        self.stats = None
        self.features = None
        nltk.download('punkt')

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
            
            results[f'rank_n{n}'] = self.rank_with_svd(s)
            results[f'largest_sv_n{n}'] = s[0]
            results[f'sv_ratio_n{n}'] = s[0]/s[-1] if len(s) > 1 else 0
        
        return results

    def process_dataset(self, dataset):
        all_results = []
        
        for sample in tqdm(dataset):
            ai_results = self.analyze_text(sample["ai"])
            ai_results['type'] = 'ai'
            ai_results['text_length'] = len(sample["ai"])
            
            human_results = self.analyze_text(sample["human"])
            human_results['type'] = 'human'
            human_results['text_length'] = len(sample["human"])
            
            all_results.extend([ai_results, human_results])
            
        return pd.DataFrame(all_results)

    def visualize_results(self, df):
        plt.figure(figsize=(12,6))
        sns.boxplot(data=df.melt(id_vars=['type'], 
                              value_vars=[f'rank_n{n}' for n in self.n_range]),
                  x='variable', y='value', hue='type')
        plt.title('Rank Distribution by N-gram Size')
        plt.show()

    def compute_training_stats(self, df):
        stats = {}
        features = [f'rank_n{n}' for n in self.n_range] + \
                  [f'largest_sv_n{n}' for n in self.n_range]
        
        for type_ in ['ai', 'human']:
            stats[type_] = {
                'mean': df[df['type'] == type_][features].mean(),
                'std': df[df['type'] == type_][features].std()
            }
        
        self.stats = stats
        self.features = features

    def predict_text(self, text):
        if self.stats is None:
            raise ValueError("Must compute training stats before prediction")
            
        results = self.analyze_text(text)
        results['text_length'] = len(text)
        
        scores = {'ai': 0, 'human': 0}
        for type_ in ['ai', 'human']:
            for feature in self.features:
                z_score = abs(results[feature] - self.stats[type_]['mean'][feature]) / self.stats[type_]['std'][feature]
                scores[type_] += z_score
                
        return 0 if scores['ai'] < scores['human'] else 1

    def evaluate_test_set(self, test_dataset):
        predictions = []
        true_labels = []
        
        for sample in tqdm(test_dataset):
            ai_pred = self.predict_text(sample['ai'])
            human_pred = self.predict_text(sample['human'])
            
            predictions.extend([ai_pred, human_pred])
            true_labels.extend([0, 1])
        
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        cm = confusion_matrix(true_labels, predictions)
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 AI     Human")
        print(f"Actual AI     {cm[0][0]:6d} {cm[0][1]:8d}")
        print(f"Actual Human  {cm[1][0]:6d} {cm[1][1]:8d}")
        
        return predictions



def split_dataset_random(dataset, seed=42):
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    return {
        'train': split['train'],
        'validation': split['test']
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for dataset processing")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset or huggingface id"
    )

    args = parser.parse_args()
    dataset_path = args.dataset_path

    ds = load_dataset(dataset_path, split="train")
    ds = split_dataset_random(ds)

    # Initialize analyzer
    analyzer = TextAnalyzer()

    # Process training data
    train_df = analyzer.process_dataset(ds["train"])
    train_df.to_csv('analysis_results.csv', index=False)

    # Visualize results
    analyzer.visualize_results(train_df)

    # Compute stats for prediction
    analyzer.compute_training_stats(train_df)

    # Evaluate on test set
    predictions = analyzer.evaluate_test_set(ds["validation"])
    # 70% accuracy
