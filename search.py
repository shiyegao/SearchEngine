import os.path
import pickle
from collections import defaultdict
import math
from nltk.stem import PorterStemmer, LancasterStemmer
from create_index import tokenize
import gzip
import json


def index_elimination(query, inverted_index):
    """ Performs index elimination pruning """

    pruned_docs = []

    for word in query:
        # if word in inverted_index:
        try:
            for doc_id in inverted_index[word]:
                if doc_id not in pruned_docs:
                    pruned_docs.append(doc_id)
                    # yield doc_id
        except:
            continue

    return pruned_docs


def calculate_score(query, doc_id, inverted_index, documents, doc_freq):
    """ Calculates score as angle between query vector and vector represented by doc_id """

    score = 0

    # for word in query:
    #     if doc_id in inverted_index[word]:
    #         score += inverted_index[word][doc_id] * query[word]

    for word in query:
        try:
            score += (inverted_index[word][doc_id] / documents[doc_id]['word_count']) * math.log(len(documents) / doc_freq[word])
        except:
            continue
    return score


def search(query_terms, inverted_index, doc_freq, documents, maxnum=10):
    scores = defaultdict(float)

    eliminated_docs = index_elimination(query_terms, inverted_index)

    for doc_id in eliminated_docs:
        scores[doc_id] = calculate_score(query_terms, doc_id, inverted_index, documents, doc_freq)

    # 根据得分排序
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:maxnum]

    # 返回查询结果
    results = []
    for doc_id, score in sorted_results:
        document = documents[doc_id]

        file_id = doc_id[: 5]
        num = doc_id.replace(file_id, '')

        file_name = 'c4-train.' + file_id + '-of-00512.json.gz'
        file_path = os.path.join('c4/realnewslike', file_name)
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            for tmp, line in enumerate(file):
                if tmp != int(num):
                    continue
                data = json.loads(line)
                content = data['text'].lower()
                break

        result = {
            'url': document['url'],
            'score': score,
            'content': content
        }
        results.append(result)

    return results
#
#
# if __name__ == '__main__':
#
#     DIRECTORY = 'test_result'
#     with open('test_result/inverted_index.pkl', 'rb') as f:
#         invert_index = pickle.load(f)
#
#     with open('test_result/doc_freq.pkl', 'rb') as f:
#         doc_freq = pickle.load(f)
#
#     with open('test_result/document.pkl', 'rb') as f:
#         document = pickle.load(f)
#
#     stemmer = PorterStemmer()
#     query = 'The overwhelming majority of Greek workers'
#     query_terms = tokenize(query, stemmer)
#     print(search(query_terms, invert_index, doc_freq, document, maxnum=10))
