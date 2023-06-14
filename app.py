import os
from create_index import searchengine_build
import pickle
from nltk.stem import PorterStemmer
from create_index import tokenize
from search import search
from transformers import AutoTokenizer, LongT5ForConditionalGeneration


import os
import time
import gradio as gr
import numpy as np

import requests
from tqdm import tqdm
import sys
import glob
import math
import pprint
import random


DIRECTORY = 'c4/realnewslike'
PERCENTAGE_USE = 1e-4
SAVE_DIR = 'result/%s' % PERCENTAGE_USE

if not os.path.exists(SAVE_DIR):
    searchengine_build(DIRECTORY, PERCENTAGE_USE, SAVE_DIR)

index_path = os.path.join(SAVE_DIR, 'inverted_index.pkl')
docfreq_path = os.path.join(SAVE_DIR, 'doc_freq.pkl')
document_path = os.path.join(SAVE_DIR, 'document.pkl')
stemmer = PorterStemmer()
with open(index_path, 'rb') as f:
    invert_index = pickle.load(f)

with open(docfreq_path, 'rb') as f:
    doc_freq = pickle.load(f)

with open(document_path, 'rb') as f:
    document = pickle.load(f)



def boolean_search(query):
    return query


def dict2str(dic):
    return 'Score: {:.4f} \n \nUrl: {} \n \nContent: {}'.format(dic['score'], dic['url'], dic['content'])


def ranked_search(query):
    query_terms = tokenize(query, stemmer)
    topk = search(query_terms, invert_index, doc_freq, document, maxnum=5)
    return dict2str(topk[0]), dict2str(topk[1]), dict2str(topk[2]), dict2str(topk[3]), dict2str(topk[4]) 


def summerize(query):
    query_terms = tokenize(query, stemmer)
    topk = search(query_terms, invert_index, doc_freq, document, maxnum=3)
    contents = [f"{topk[i]['content']}" for i in range(len(topk))]

    tokenizer = AutoTokenizer.from_pretrained("./long-t5-tglobal-base")
    model = LongT5ForConditionalGeneration.from_pretrained("./long-t5-tglobal-base")
    prompt = 'Summarize the background news into 100 words.'

    inputs = tokenizer('Background: ' + ''.join(contents) + '\nTASK: ' + prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def qa(query_r, query,):
    query_terms = tokenize(query_r, stemmer)
    topk = search(query_terms, invert_index, doc_freq, document, maxnum=3)
    # contents = [f"The {i}-th news: {topk[i]['content']}\n" for i in range(len(topk))]
    contents = [f"{topk[i]['content']}" for i in range(len(topk))]

    tokenizer = AutoTokenizer.from_pretrained("./long-t5-tglobal-base")
    model = LongT5ForConditionalGeneration.from_pretrained("./long-t5-tglobal-base")

    inputs = tokenizer('Background: ' + ''.join(contents) + '\nQuestion:' + query, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    title = "WSM Project: Search Engine"
    css = "footer {visibility: hidden}"

    with gr.Blocks(title=title, css=css) as demo:
        gr.Markdown("Welcome to our search engine.")

        with gr.Tab("Boolean Search"):
            query_b = gr.Textbox(label="Input Query")
            output_b = gr.Textbox(label="Search Result")
            boolean_btn = gr.Button("Search")

        with gr.Tab("Ranked Search"):
            query_r = gr.Textbox(label="Input Query")
            ranked_btn = gr.Button("Search")
            with gr.Accordion("Search Result"):
                output_r1 = gr.Textbox(label="TOP-1")
                output_r2 = gr.Textbox(label="TOP-2")
                output_r3 = gr.Textbox(label="TOP-3")
                output_r4 = gr.Textbox(label="TOP-4")
                output_r5 = gr.Textbox(label="TOP-5")
            
            with gr.Accordion("Advance Functions"):
                with gr.Tab("Summarization"):
                    output_sum = gr.Textbox(label='Result:')
                    sum_btn = gr.Button("Summerize")
                with gr.Tab("Q&A"):
                    query_qa = gr.Textbox(label="Question:")
                    output_qa = gr.Textbox(label='Result:')
                    qa_btn = gr.Button("Ask")
            
        boolean_btn.click(boolean_search, [query_b], [output_b])
        ranked_btn.click(ranked_search, [query_r], [output_r1, output_r2, output_r3, output_r4, output_r5, ])
        sum_btn.click(summerize, [query_r], [output_sum])
        qa_btn.click(qa, [query_r, query_qa], [output_qa])

        with gr.Accordion("Open for More!"):
            gr.Markdown("Team Member: <br> Jiale Zhang: Boolean Search <br> Shuo Tang: Ranked Search <br> \
                Shengkai Lin: Q&A <br> Jin Gao: Summarization, GUI")

    demo.queue()
    demo.launch(show_api=False, server_name="0.0.0.0", share=True)
    # demo.launch(show_api=False, share=True)