import os
from sklearn.feature_extraction.text import TfidfVectorizer

def load_regulatory_corpus():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'regulatory_raw')
    docs = []
    if not os.path.isdir(base):
        return docs
    for fname in os.listdir(base):
        if fname.endswith('.txt'):
            with open(os.path.join(base, fname), 'r', encoding='utf-8') as f:
                docs.append({'title': fname, 'text': f.read()})
    return docs

def build_tfidf_index(reg_docs):
    texts = [d['text'] for d in reg_docs]
    vectorizer = TfidfVectorizer(stop_words='english')
    if len(texts) == 0:
        return vectorizer, None
    tfidf = vectorizer.fit_transform(texts)
    return vectorizer, tfidf

def retrieve_top_k(query, reg_docs, vectorizer, tfidf_matrix, top_k=3):
    if tfidf_matrix is None:
        return reg_docs[:top_k]
    qv = vectorizer.transform([query])
    import numpy as np
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    idx = list((-scores).argsort()[:top_k])
    out = []
    for i in idx:
        out.append({'title': reg_docs[i]['title'], 'text': reg_docs[i]['text'], 'score': float(scores[i])})
    return out
