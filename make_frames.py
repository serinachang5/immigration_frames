import spacy
import networkx as nx
import matplotlib.pyplot as plt
import utils
import word_lists as wl
import random
import pickle
from collections import Counter
import numpy as np

sent1 = 'The howl against the Chinese is waxing louder and louder at the East, as our neighbors watch the incoming tide of celestials pouring into their midst.'
sent2 = 'Socialism in Chicago has lifted up its voice against them, and many of the New York journals are decrying against the multiplicity of their approach.'

nlp = spacy.load("en_core_web_sm")

def extract_frames_in_sentence(seed_set, sent, id_mode, nhops=5):
    '''
        Given a set of identifiers and a sentence, this function
        finds instance(s) of the identifiers and extracts the frames
        surrounding those instances. For each frame, we record the
        anchor, the unparsed window of text encompassing the frame,
        and tokens in the frame.
    '''
    all_frames = []
    doc = nlp(sent)
    for i, tok in enumerate(doc):
        anchor, blacklist = None, None
        if id_mode == 'immigrant':
            anchor, blacklist = _check_if_immigrant_anchor(tok, seed_set)
        elif id_mode.startswith('lemma-pos'):
            pos = id_mode.split('::', 1)[1]  # e.g. lemma-pos::NOUN
            if tok.lemma_ in seed_set and tok.pos_ == pos:
                anchor = tok.lemma_ + '::' + tok.pos_
                blacklist = {tok.i}
        else:
            text = tok.text.lower()
            if text in seed_set:
                anchor = text
                blacklist = {tok.i}
        if anchor is not None:
            frame, window = extract_frame(doc, tok, blacklist, nhops)
            all_frames.append((anchor, window, frame))
    return all_frames

def _check_if_immigrant_anchor(tok, seed_set):
    '''
        Helper function to check if a token is an instance, i.e.
        an anchor for a frame, of the given set of immigrant identifiers.
    '''
    if tok.pos_ in {'NOUN', 'PROPN'}:
        if tok.text.lower() in seed_set:
            return tok.text.lower(), {tok.i}
        elif tok.text.lower() in wl.IMMIGRANT_IDENTIFIERS:
            for c in tok.children:
                if c.text.lower() in seed_set and c.dep_ == 'amod':
                    anchor = sorted([c, tok], key=lambda x:x.i)
                    anchor = '_'.join(t.text.lower() for t in anchor)
                    blacklist = {tok.i, c.i}
                    return anchor, blacklist
    return None, None

def extract_frame(doc, anchor, blacklist, nhops):
    '''
        Given a spacy-parsed document and an anchor token, this
        function extracts the frame surrounding the anchor. The
        frame includes all tokens j with l <= nhops, where l is
        the collapsed path length from the anchor to j.
    '''
    curr = [(anchor, [])]
    next = []
    frame = []
    seen_toks = {anchor.text}
    max_pos = anchor.i
    min_pos = anchor.i
    while len(curr) > 0:
        for t, deps in curr:
            if len(deps) == 0:  # processing anchor token
                next.append((t.head, [t.dep_]))
                next.extend([(c, [c.dep_]) for c in t.children])
            elif t.text not in seen_toks:
                cl = get_collapsed_len(deps)
                if cl <= nhops:
                    if t.i not in blacklist:
                        frame.append((t.lemma_, t.pos_, deps))
                    next.append((t.head, deps + [t.dep_]))
                    next.extend([(c, deps + [c.dep_]) for c in t.children])
                    seen_toks.add(t.text)
                    position = t.i
                    if position > max_pos:
                        max_pos = position
                    if position < min_pos:
                        min_pos = position
        curr = next
        next = []
    left_window = max(0, min_pos-2)
    right_window = min(len(doc)-1, max_pos+2)
    window = ' '.join([t.text for t in doc[left_window:right_window+1]])
    return frame, window

def get_compound_phrase(tok):
    pieces = []
    curr = [tok]
    next = []
    while len(curr) > 0:
        for t in curr:
            pieces.append(t)
            if t.dep_ == 'compound':
                next.append(t.head)
            for c in t.children:
                if c.dep_ == 'compound':
                    next.append(c)
            curr = next
            next = []
    pieces = sorted(pieces, key=lambda x:x.i)
    text = [p.lemma_ for p in pieces]
    return '_'.join(text)

def get_collapsed_len(deps):
    num = 0
    for dep in deps:
        if dep != 'prep' and dep != 'conj':
            num += 1
    return num

def extract_frames_in_corpus(seed_set, id_mode, data_mode, sample_size=1000):
    '''
        Given a set of identifiers and a corpus, this function extracts all
        frames from the corpus.
    '''
    all_keys, all_docs = utils.prep_documents(1880, data_mode, sample_size=sample_size, prep=None, join_lines=False)
    print('Loaded corpus:', len(all_docs))
    if sample_size is None:
        sample = all_docs
    else:
        sample = random.sample(all_docs, sample_size)

    results = []
    for i, doc in enumerate(sample):
        if i % 500 == 0:
            excerpt = doc if len(doc) <= 10 else doc[:10]
            print('Parsing sample {}: {}...'.format(i, excerpt))
        lowercased = ' '.join(doc).lower()
        if any(ci in lowercased for ci in seed_set):
            frames = extract_frames_in_sentence(seed_set, ' '.join(doc), id_mode)
            results.extend(frames)
    print('Num results found:', len(results))
    return results

def visualize_parse_tree(doc):
    G = nx.DiGraph()
    for token in doc:
        if not G.has_node(token.text):
            G.add_node(token.text)
        head = token.head
        if not G.has_node(head.text):
            G.add_node(head.text)
        G.add_edge(head.text, token.text)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

if __name__ == "__main__":
    results = extract_frames_in_corpus(wl.GENERIC_IDENTIFIERS + wl.PRONOUNS, 'generic', 'general', sample_size=None)
    pickle.dump(results, open('../saved/california/general_frames.pkl', 'wb'))

    # top_chinese_nsubj_verbs = ['flock', 'celebrate', 'citizenship', 'retreat', 'pour',
    #                            'evacuate', 'land', 'flee', 'arrive', 'organize',
    #                            'convict', 'discharge', 'emigrate', 'depart', 'arrest',
    #                            'engage', 'attempt', 'contribute', 'employ', 'apply',
    #                            'escape', 'prevent', 'permit', 'invade', 'increase']
    # top_pronoun_nsubj_verbs = ['love', 'possess', 'read', 'accept', 'win', 'jump',
    #                            'choose', 'please', 'arc', 'fly']
    # results = extract_frames_in_corpus(top_chinese_nsubj_verbs + top_pronoun_nsubj_verbs, 'lemma-pos::VERB', 'general', sample_size=None)
    # print('Found {} results for verbs'.format(len(results)))
    # pickle.dump(results, open('temp.pkl', 'wb'))

    # chinese_results, _ = pickle.load(open('temp.pkl', 'rb'))
    # examples = []
    # for seed, window, frame in chinese_results:
    #     if seed == 'pour::VERB':
    #         examples.append(window)
    # examples = random.sample(examples, 10)
    # for e in examples:
    #     print(e)
