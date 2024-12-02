from pandas.core.computation.parsing import token
from torch import maximum
import skdim
from datasets import load_dataset
from embed import token_transformer_emb

def maximum_likelihood_estimator(embeddings):
    mle = skdim.id.MLE().fit(embeddings)
    return mle.dimension_


def main():
    human_dataset_path = "abisee/cnn_dailymail"
    ai_dataset_path = "HuggingFaceTB/cosmopedia-100k"
    embedding_model = "bert-base-uncased"
    n_elements = 100


    ai_ds = load_dataset(ai_dataset_path, split="train")
    hm_ds = load_dataset(human_dataset_path, '3.0.0', split="train")

    ai_ds = ai_ds["text"][:n_elements]
    hm_ds = hm_ds["article"][:n_elements]

    ai_emb = token_transformer_emb(ai_ds, embedding_model)
    hm_emb = token_transformer_emb(hm_ds, embedding_model)
    
    
    ai_dims = 0.0
    hm_dims = 0.0
    for ai, hm in zip(ai_emb, hm_emb):
        ai_dim = maximum_likelihood_estimator(ai)
        hm_dim = maximum_likelihood_estimator(hm)

        ai_dims += ai_dim
        hm_dims += hm_dim

    print(f"Average IN for AI dataset : {ai_dataset_path.split('/')[-1]} with MLE = {ai_dims / n_elements}")
    print(f"Average IN for Human dataset : {human_dataset_path.split('/')[-1]} with MLE = {hm_dims / n_elements}")

    # Average IN for AI dataset : cosmopedia-100k with MLE = 6.125461053326112
    # Average IN for Human dataset : cnn_dailymail with MLE = 6.394587059911538


if __name__ == "__main__":
    main()

    
