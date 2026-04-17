import numpy as np
import pandas as pd
from utils import *
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_model(MODEL_NAME):
    print(f"Loading {MODEL_NAME}.")
    model = SentenceTransformer(MODEL_NAME)
    print("model loaded.")

def compute_similarities(model, corpus):
    output_dims = model.get_embedding_dimension()
    sentences_filter = corpus["embeddings"] == np.zeros((output_dims))
    new_sentences = corpus[ sentences_filter ]
    corpus[sentences_filter]["embeddings"] = list(model.encode(list(new_sentences["text"])))


load_model(MODEL_NAME)
corpus = get_corpus(CORPUS_PATH, PARQUET_PATH)
compute_similarities(model, corpus)
print(corpus)
