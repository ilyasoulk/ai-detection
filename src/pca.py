from sklearn.decomposition import PCA
import torch
from intrinsic_dim import IntrinsicDimAnalyzer
from concurrent.futures import ThreadPoolExecutor
import os


class PCAAnalyzer(IntrinsicDimAnalyzer):
    def __init__(self, embedding_model, thresholds=[0.8, 0.9, 0.95]):
        super().__init__(embedding_model=embedding_model)
        self.thresholds = thresholds
        self.features = [f"dim_t{t}" for t in self.thresholds] + [
            f"var_t{t}" for t in self.thresholds
        ]

    def compute_intrinsic_dimensions(self, embeddings: torch.Tensor):
        embeddings = embeddings.cpu().numpy()
        pca_n_components = min(embeddings.shape[1:])

        def process_single_embedding(embedding):
            pca = PCA(n_components=pca_n_components)
            pca.fit(embedding)
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

        max_workers = min(len(embeddings), os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_results = list(executor.map(process_single_embedding, embeddings))

        return all_results
