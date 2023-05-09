'''
 * Copyright (c) 2023, Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import nltk
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
import numpy as np


def get_instruction(sample):
    if sample["prompt_all"] == "":
        return ' '.join([sample["prompt_task"], sample["prompt_context"]])
    return sample["prompt_all"]


def preprocess_knowledge(text, english_stopwords):
    tokens = nltk.word_tokenize(text.lower())
    return [token for token in tokens if token not in english_stopwords]


def retrieve_global_knowledge(output, global_knowledge, top_n=5):
    if len(global_knowledge) < top_n:
        return '\n'.join(global_knowledge)
    english_stopwords = set(stopwords.words('english'))
    knowledge = [preprocess_knowledge(paragraph, english_stopwords) for paragraph in global_knowledge]
    bm25 = BM25Okapi(knowledge)
    query = preprocess_knowledge(output, english_stopwords)
    retrieved_knowledge = bm25.get_top_n(query, global_knowledge, n=top_n)  # For now, retrieving top 5
    return '\n'.join(retrieved_knowledge)
