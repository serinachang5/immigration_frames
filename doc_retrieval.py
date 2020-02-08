import numpy as np
import re
import os
import pickle
import string
from collections import Counter
import word_lists as wl
import utils

PATH_TO_NYT_DATA = "../sample/nyt/"
PATH_TO_OUTPUT = '../saved/'

def matching_score(qterms, doc_tokens, term_weights, avg_doc_len, k = 1.2, b = 0.75):
    '''
    :param - qterms: query terms
    :param - doc_tokens: tokens of the current document being scored
    :param - term weights: weights for each term
    :param - avg_doc_len: average document length
    :param - k: term frequency normalization factor
    :param - b: tuning constant
    return type: float
    returns: how relevant the document is to the query terms
    '''
    score_sum = 0
    terms_found = 0
    doc_counter = Counter(doc_tokens)
    K = k * ((1-b) + (b * len(doc_tokens) / avg_doc_len))
    for term in qterms:
        if term in term_weights:
            terms_found += 1
            tf = doc_counter[term]
            numerator = tf * (k + 1) * term_weights[term]
            denom = K + tf
            score_sum += numerator / denom
    if terms_found > 0:
        return score_sum / terms_found
    return None

def relevance_weight(word, rel_doc_counts, R, doc_counts, N):
    '''
    :param - word: the term being scored
    :param - rel_doc_counts(Counter): terms mapped to how many relevant documents they appear in
    :param - R: number of relevant documents
    :param - doc_counts(Counter): terms mapped to how many documents they appear in
    :param - N: total number of documents in the corpus
    return type: float
    returns: the relevance of the word to the relevant documents
    '''
    r = rel_doc_counts[word]
    n = doc_counts[word]
    numerator = (r + 0.5) * (N - n - R + r + 0.5)
    denom = (R - r + 0.5) * (n - r + 0.5)
    return np.log(numerator / denom)

def sparck_jones(qterms, doc_keys, documents, word_doc_counts, avg_doc_len,
                 print_top_docs=True, print_top_terms=True, R=100):
    '''
    :param - qterms: the seed query terms
    :param - R: the number of relevant documents to keep
    return type: list of tuples
    returns: the documents scored wrt the query terms, using the Sparck Jones matching score
    '''
    term_weights = {}
    for term, doc_count in word_doc_counts.most_common():
        term_weights[term] = np.log(len(documents) / doc_count)

    scored_documents = []
    for i, (doc_key, doc) in enumerate(zip(doc_keys, documents)):
        score = matching_score(qterms, doc, term_weights, avg_doc_len)
        scored_documents.append((doc_key, doc, score))
    scored_documents = sorted(scored_documents, key=lambda x:x[2], reverse=True)
    if print_top_docs:
        print('Top 10 ranked documents:')
        for doc_key, doc, score in scored_documents[:10]:
            print(doc_key, '->', round(score, 5))

    word_rel_doc_counts = utils.compute_doc_counts([t[1] for t in scored_documents[:R]])
    rel_vocab = list(word_rel_doc_counts.keys())
    for word in rel_vocab:
        rw = relevance_weight(word, word_rel_doc_counts, R, word_doc_counts, len(documents))
        term_weights[word] = word_rel_doc_counts[word] * rw
    ranked = sorted(rel_vocab, key=lambda x:term_weights[x], reverse=True)
    if print_top_terms:
        top_terms = ranked[:50]
        print('Top 50 query terms:', top_terms)
    return scored_documents, ranked

def retrieve_docs(qterms, decade, mode, all_keys=None, all_docs=None):
    if all_keys is None or all_docs is None:
        all_keys, all_docs = utils.prep_documents(decade, mode)
        print('Prepped corpus of size', len(all_docs))
    all_counts = utils.compute_doc_counts(all_docs)
    avg_len = np.mean([len(doc) for doc in all_docs])
    return sparck_jones(qterms, all_keys, all_docs, all_counts, avg_len)

if __name__ == "__main__":
    all_keys, all_docs = utils.prep_documents(1880, 'general')
    for qterms in [wl.GERMAN_IDENTIFIERS, wl.IRISH_IDENTIFIERS, wl.JAPANESE_IDENTIFIERS]:
        retrieve_docs(qterms, 1880, 'general', all_keys, all_docs)
