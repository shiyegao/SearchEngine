import os
import math
import json
import gzip
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string
from nltk.stem import PorterStemmer, LancasterStemmer
import re
import time


NUMBER_DOCUMENTS_PER_FILE = 26952


def tokenize(textwords, stemmer):
    unwanted_chars = re.compile(r'[-_—,"\d]')
    tokens = word_tokenize(textwords)
    res = []
    for token in tokens:
        token = unwanted_chars.sub('', token)
        token = token.strip(string.punctuation)
        token = stemmer.stem(token)
        if token and len(token) > 1 and token not in stopwords.words('english'):
            res.append(token)
    return res


def save_index(inverted_index, doc_freq, document,  output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'inverted_index.pkl'), 'wb') as file:
        pickle.dump(inverted_index, file)
    with open(os.path.join(output_dir, 'doc_freq.pkl'), 'wb') as file:
        pickle.dump(doc_freq, file)
    with open(os.path.join(output_dir, 'document.pkl'), 'wb') as  file:
        pickle.dump(document, file)


def searchengine_build(directory, percentage_use, save_dir):
    # 初始化
    inverted_index = defaultdict(dict)
    doc_freq = defaultdict(int)
    document = defaultdict(dict)
    stemmer = PorterStemmer()

    upper_bound = int(percentage_use * NUMBER_DOCUMENTS_PER_FILE)
    t1 = time.time()

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if 'validation' in file_path:
            continue

        file_id = file_path.split('-')[1].split('.')[-1]

        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            for num, line in enumerate(file):
                data = json.loads(line)
                doc_id = file_id + str(num)
                document[doc_id]['url'] = data['url']
                document[doc_id]['time'] = data['timestamp']
                document[doc_id]['word_count'] = 0

                content = data['text'].lower()

                tokens = tokenize(content, stemmer)

                for token in tokens:
                    try:
                        inverted_index[token][doc_id] += 1
                    except:
                        inverted_index[token][doc_id] = 1
                    doc_freq[token] += 1
                    document[doc_id]['word_count'] += 1

                if num >= upper_bound:
                    break
        # print(len(inverted_index.keys()), len(doc_freq.keys()))
    t2 = time.time()
    save_index(inverted_index, doc_freq, document, output_dir=save_dir)
    print('Successfully saved to %s, time cost: %.1f' % (save_dir, t2-t1))

