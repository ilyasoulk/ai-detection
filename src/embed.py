import nltk
import torch
import gensim.downloader as api
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

def preprocess_text(text):
    tokens = text.split()
    return tokens


def word2vec_emb(texts, embedding_model):
    model = api.load(embedding_model)
    res = []
    for text in texts:
        tokens = preprocess_text(text)
        vectors = torch.Tensor([model[token] for token in tokens if token in model])
        res.append(vectors)
    return res


def token_transformer_emb(texts, embedding_model, max_tokens=200):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    embedder = AutoModel.from_pretrained(embedding_model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokens = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    max_tokens = min(max_tokens, len(tokens["input_ids"]))
    tokens["input_ids"] = tokens["input_ids"][:max_tokens]

    with torch.no_grad():
        outputs = embedder(**tokens)

    embeddings = outputs.last_hidden_state

    return embeddings


def sentence_transformer_emb(texts, embedding_model):
    split_texts = [nltk.sent_tokenize(text) for text in texts]

    embedder = SentenceTransformer(embedding_model)
    
    grouped_embeddings = [
        embedder.encode(sentences)
        for sentences in split_texts
    ]

    return grouped_embeddings

