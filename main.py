from create_index import searchengine_build
import pickle
from nltk.stem import PorterStemmer
from create_index import tokenize
from search import search
import os

DIRECTORY = 'c4/realnewslike'
PERCENTAGE_USE = 1e-4
SAVE_DIR = 'result/%s' % PERCENTAGE_USE

if not os.path.exists(SAVE_DIR):
    searchengine_build(DIRECTORY, PERCENTAGE_USE, SAVE_DIR)

index_path = os.path.join(SAVE_DIR, 'inverted_index.pkl')
docfreq_path = os.path.join(SAVE_DIR, 'doc_freq.pkl')
document_path = os.path.join(SAVE_DIR, 'document.pkl')

with open(index_path, 'rb') as f:
    invert_index = pickle.load(f)

with open(docfreq_path, 'rb') as f:
    doc_freq = pickle.load(f)

with open(document_path, 'rb') as f:
    document = pickle.load(f)

stemmer = PorterStemmer()
query = ['cook breakfast', 'Donald Trump'][1]
query_terms = tokenize(query, stemmer)
topk = search(query_terms, invert_index, doc_freq, document, maxnum=5)
contents = [f"The {i}-th news: {topk[i]['content']}\n" for i in range(len(topk))]
contents = [f"{topk[i]['content']}\n" for i in range(len(topk))]
print('Original:', contents)

from transformers import AutoTokenizer, LongT5ForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained("./long-t5-tglobal-base")
model = LongT5ForConditionalGeneration.from_pretrained("./long-t5-tglobal-base")
# prompt = 'Summarize the previous news into 100 words.'
prompt = 'How old is Donald Trump?'
inputs = tokenizer('\n'.join(contents) + prompt, return_tensors="pt").input_ids
outputs = model.generate(inputs, max_length=1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))