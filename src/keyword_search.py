from utils import *


def build_index_and_corpus_metdata(corpus):
    tokens = set(corpus["tokens"].sum())

    inverted_index = {token: {} for token in tokens}
    corpus_metadata = {"avg_doc_length":0, "doc_lengths":dict.fromkeys(corpus.index)}

    for i in corpus.index:
        doc = corpus.loc[i]
        corpus_metadata["doc_lengths"][i] = len(doc["tokens"])
        corpus_metadata["avg_doc_length"] += len(doc["tokens"])
        for token in set(doc["tokens"]):
            inverted_index[token][i] = doc["tokens"].count(token)

    corpus_metadata["avg_doc_length"] /= len(corpus)
    return (inverted_index, corpus_metadata)

if __name__ == "__main__":

    corpus = get_corpus(CORPUS_PATH, PARQUET_PATH)

    corpus["tokens"] = corpus["text"].apply(lambda x: tokenize(x))
    inverted_index, corpus_metadata = build_index_and_corpus_metdata(corpus)
