from utils import *
from math import log
from collections import Counter


def freq(term, doc_id, inverted_index):
    term_index = inverted_index.get(term)
    if not term_index:
        return 0
    return term_index.get(doc_id, 0)

def df(term, inverted_index):
    term_index = inverted_index.get(term)
    if not term_index:
        return 0
    return len(term_index)

def IDF(term, inverted_index, docs_num):
    return log((docs_num - df(term, inverted_index) + 0.5) / (df(term, inverted_index) + 0.5) + 1)

def BM25(query, doc_id, inverted_index, corpus_metadata, k1=1.5, b=0.75):
    query_tokens = tokenize(query)
    score = 0
    val = 0
    for term in query_tokens:
        val = IDF(term, inverted_index, corpus_metadata["num_of_docs"]) * freq(term, doc_id, inverted_index) * (k1 + 1)
        val /= freq(term, doc_id, inverted_index) + k1 * (1 - b + b * corpus_metadata["doc_lengths"][doc_id]/corpus_metadata["avg_doc_length"])
        score += val
        
    return score

def search(query, corpus, inverted_index, corpus_metadata, top_k=5):
    query_tokens = tokenize(query)
    rank = {}
    doc_ids = set()
    for term in query_tokens:
        indices = inverted_index.get(term)
        if indices:
            doc_ids = doc_ids.union(indices)

    for doc_id in doc_ids:
        rank[doc_id] = BM25(query, doc_id, inverted_index, corpus_metadata, k1=1.5, b=0.75)

    return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]

def build_index_and_corpus_metadata(corpus):
    corpus["tokens"] = corpus["text"].apply(lambda x: tokenize(x))
    tokens = set(corpus["tokens"].sum())

    inverted_index = {token: {} for token in tokens}
    corpus_metadata = {"avg_doc_length": 0, "num_of_docs":len(corpus), "doc_lengths": {}}

    for i in corpus.index:
        doc = corpus.loc[i]
        corpus_metadata["doc_lengths"][i] = len(doc["tokens"])
        corpus_metadata["avg_doc_length"] += len(doc["tokens"])
        token_counts = Counter(doc["tokens"])
        for token, count in token_counts.items():
            inverted_index[token][i] = count

    corpus_metadata["avg_doc_length"] /= len(corpus)
    return (inverted_index, corpus_metadata)

if __name__ == "__main__":

    corpus = get_corpus(CORPUS_PATH, PARQUET_PATH)
    inverted_index, corpus_metadata = build_index_and_corpus_metadata(corpus)
    query = "how do I make lookups"
    rank = search(query, corpus, inverted_index, corpus_metadata, top_k=5)
    print(rank)

    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(text) for text in corpus["text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    print(scores)
