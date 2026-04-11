import pandas as pd

CORPUS_PATH = "corpus/corpus.json"
PARQUET_PATH = "corpus/corpus.parquet"

def load_corpus(path):
    data = pd.read_json(path)
    data = data.set_index("id")
    return data

def prepare_and_save(corpus_path, parquet_path):
    data = load_corpus(corpus_path)
    data.to_parquet(parquet_path)
    print(f"Saved {len(data)} documents to {parquet_path}")
    return data

def load_prepared(parquet_path):
    return pd.read_parquet(parquet_path)

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
    import os
    if os.path.exists(PARQUET_PATH):
        data = load_prepared(PARQUET_PATH)
        print("Loaded from cache.")
    else:
        data = prepare_and_save(CORPUS_PATH, PARQUET_PATH)

    # test cases
    print("\n--- author alice ---")
    print(filter_corpus(data, authors=["alice"]))

    print("\n--- tags [ml, databases] ---")
    print(filter_corpus(data, tags=["ml", "databases"]))

    print("\n--- year_from 2023, author dave ---")
    print(filter_corpus(data, year_from=2023, authors=["dave"]))

    print("\n--- no matches ---")
    print(filter_corpus(data, authors=["zara"]))
