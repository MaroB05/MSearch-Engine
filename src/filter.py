import json
corpus_path = "corpus/corpus.json"

def load_corpus(corpus_path):
    with open(corpus_path) as corpus:
        original_data = json.load(corpus)

    return original_data

def create_table_and_index(original_data):
    index = dict()
    index["author"] = dict()
    index["tags"] = dict()
    corpus_map = dict()
    for element in original_data:
        corpus_map[element["id"]] = dict(element["title"]=element["text"])
        index["author"][element["author"]] =

