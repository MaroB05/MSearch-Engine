from utils import *

corpus = get_corpus(CORPUS_PATH, PARQUET_PATH)

corpus["tokens"] = corpus["text"].apply(lambda x: tokenize(x))
tokens = set(corpus["tokens"].sum())

inverted_index = dict.fromkeys(tokens, {})
corpus_metadata = {"avg_doc_length":0, "doc_lengths":dict.fromkeys(corpus.index))}

for i in corpus.index:
    doc = corpus.loc[i]
    corpus_metadata["doc_lengths"][i] = len(doc["text"])
    corpus_metadata["avg_doc_length"] += len(doc["text"])
    for token in set(doc["tokens"]):
        inverted_index[token][i] = doc["text"].count(token)

corpus_metadata["avg_doc_length"] /= len(corpus)


