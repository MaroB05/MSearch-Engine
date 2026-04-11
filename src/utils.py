import os
#import re
import nltk
import pandas as pd
from string import punctuation
from nltk.stem import PorterStemmer

CORPUS_PATH = "corpus/corpus.json"
PARQUET_PATH = "corpus/corpus.parquet"

stemmer = PorterStemmer()

def load_corpus(corpus_path):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    data = pd.read_json(corpus_path)
    data = data.set_index("id")
    return data

def save_corpus(data, parquet_path):
    data.to_parquet(parquet_path)
    return data

def load_cached(parquet_path):
    return pd.read_parquet(parquet_path)

def get_corpus(corpus_path, parquet_path):
    if os.path.exists(parquet_path) and os.path.getmtime(corpus_path) < os.path.getmtime(parquet_path):
        corpus = load_cached(parquet_path)
        print("Loaded from cache.")
    else:
        corpus = load_corpus(corpus_path)
        save_corpus(corpus, parquet_path)
        print("Saved data to cache.")
    return corpus


def tokenize(text, stem=False):
    text = text.lower()
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [t for t in tokens if t not in punctuation]
    #text = re.sub(r"[^\w\s]", " ", text)
    #tokens = text.split()
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def extract_bag_of_words(tokens):
    return set(tokens)

