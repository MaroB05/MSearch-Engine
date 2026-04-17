from utils import *

def filter_corpus(corpus, ids=None, tags=None, authors=None, year_from=None, year_to=None):
    result = corpus.copy()

    if ids:
        result = result[result.index.isin(ids)]
    if tags:
        result = result[result["tags"].apply(lambda doc_tags: all(t in doc_tags for t in tags))]
    if authors:
        result = result[result["author"].isin(authors)]
    if year_from is not None:
        result = result[result["year"] >= year_from]
    if year_to is not None:
        result = result[result["year"] <= year_to]

    return result

if __name__ == "__main__":

    data = get_corpus(CORPUS_PATH, PARQUET_PATH)

    # test cases
    print("\n--- author alice ---")
    print(filter_corpus(data, authors=["alice"]))

    print("\n--- tags [ml, databases] ---")
    print(filter_corpus(data, tags=["ml", "databases"]))

    print("\n--- year_from 2023, author dave ---")
    print(filter_corpus(data, year_from=2023, authors=["dave"]))

    print("\n--- no matches ---")
    print(filter_corpus(data, authors=["zara"]))
