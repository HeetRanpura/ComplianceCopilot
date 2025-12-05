# regulation_index/build_reg_index.py
"""
Hybrid retriever:
- TF-IDF lexical retriever (scikit-learn)
- FAISS semantic retriever (sentence-transformers embeddings)
- Reciprocal Rank Fusion (RRF) to combine ranks

Expected data folder:
  data/regulatory_raw/*.txt

Public functions:
- load_regulatory_corpus()
- build_tfidf_index(reg_docs)
- build_faiss_index(reg_docs, embed_model_name='all-MiniLM-L6-v2')
- retrieve_top_k(query, reg_docs, vectorizer, tfidf_matrix, faiss_index, embeddings, embed_model, top_k=3)
"""

import os
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Lazy imports for heavy deps (only when needed)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise ImportError("Install sentence-transformers (pip install sentence-transformers)") from exc
    return SentenceTransformer(model_name)

def _load_faiss():
    try:
        import faiss
    except Exception as exc:
        raise ImportError("Install faiss-cpu (pip install faiss-cpu) or faiss-gpu if you prefer") from exc
    return faiss


# ----------------------
# Load regulatory docs
# ----------------------
def load_regulatory_corpus(data_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load regulatory text files from data/regulatory_raw/*.txt
    Returns list of dicts: [{ 'title': filename, 'text': file_contents }, ...]
    """
    base = data_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'regulatory_raw')
    base = os.path.normpath(base)
    docs = []
    if not os.path.isdir(base):
        return docs
    for fname in sorted(os.listdir(base)):
        if fname.lower().endswith('.txt'):
            fp = os.path.join(base, fname)
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    text = f.read()
                docs.append({'title': fname, 'text': text})
            except Exception:
                # skip unreadable files
                continue
    return docs


# ----------------------
# TF-IDF index (lexical)
# ----------------------
def build_tfidf_index(reg_docs: List[Dict[str, str]]) -> Tuple[TfidfVectorizer, Optional[object]]:
    """
    Build TF-IDF vectorizer and matrix from reg_docs.
    Returns (vectorizer, tfidf_matrix). tfidf_matrix can be None if no docs.
    """
    texts = [d['text'] for d in reg_docs]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=50000)
    if len(texts) == 0:
        return vectorizer, None
    tfidf = vectorizer.fit_transform(texts)
    return vectorizer, tfidf


# ----------------------
# FAISS index (semantic)
# ----------------------
def build_faiss_index(reg_docs: List[Dict[str, str]], embed_model_name: str = 'all-MiniLM-L6-v2'):
    """
    Build sentence-transformers embeddings for reg_docs and create a FAISS index.
    Returns (faiss_index, embeddings_matrix, embed_model)
     - faiss_index: FAISS index object
     - embeddings_matrix: numpy.ndarray (n_docs, dim)
     - embed_model: SentenceTransformer model instance (useful for encoding new queries)
    """
    if len(reg_docs) == 0:
        return None, None, None

    # lazy load embedding model and faiss
    embed_model = _load_sentence_transformer(embed_model_name)
    texts = [d['text'] for d in reg_docs]
    embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    faiss = _load_faiss()
    dim = embeddings.shape[1]

    # normalize embeddings for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    return index, embeddings, embed_model


# ----------------------
# Fusion retrieval helpers
# ----------------------
def _tfidf_rank_scores(query, vectorizer, tfidf_matrix, top_k=10):
    """
    Return list of (doc_index, score) from TF-IDF (cosine similarity)
    """
    if tfidf_matrix is None:
        return []
    qv = vectorizer.transform([query])
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0.0]


def _faiss_rank_scores(query, faiss_index, embed_model, top_k=10):
    """
    Return list of (doc_index, score) from FAISS semantic search.
    """
    if faiss_index is None or embed_model is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    D, I = faiss_index.search(q_emb.astype('float32'), top_k)
    res = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        res.append((int(idx), float(score)))
    return res


def reciprocal_rank_fusion(tfidf_scores, faiss_scores, top_k=5, rrf_k=60):
    """
    Simple Reciprocal Rank Fusion (RRF)
    tfidf_scores: list of (idx, score) in descending TF-IDF order
    faiss_scores: list of (idx, score) in descending FAISS order
    Returns: list of (idx, fused_score, tfidf_rank, faiss_rank) sorted by fused score
    """
    rank_map = {}
    for rank, (idx, _) in enumerate(tfidf_scores, start=1):
        rank_map.setdefault(idx, {})['tfidf_rank'] = rank
    for rank, (idx, _) in enumerate(faiss_scores, start=1):
        rank_map.setdefault(idx, {})['faiss_rank'] = rank

    fused = []
    for idx, parts in rank_map.items():
        tfidf_rank = parts.get('tfidf_rank', None)
        faiss_rank = parts.get('faiss_rank', None)
        score = 0.0
        if tfidf_rank:
            score += 1.0 / (rrf_k + tfidf_rank)
        if faiss_rank:
            score += 1.0 / (rrf_k + faiss_rank)
        fused.append((idx, score, tfidf_rank, faiss_rank))
    fused_sorted = sorted(fused, key=lambda x: -x[1])
    return fused_sorted[:top_k]


def retrieve_top_k(query: str,
                   reg_docs: List[Dict[str, str]],
                   vectorizer,
                   tfidf_matrix,
                   faiss_index,
                   embeddings,
                   embed_model,
                   top_k: int = 3):
    """
    Retrieve top_k regulatory documents for a query using hybrid strategy.
    Returns list of dicts: [{ 'title','text','score','source_scores':{'tfidf':..,'faiss':..} }, ...]
    """
    tfidf_scores = _tfidf_rank_scores(query, vectorizer, tfidf_matrix, top_k=top_k*5 if tfidf_matrix is not None else top_k)
    faiss_scores = _faiss_rank_scores(query, faiss_index, embed_model, top_k=top_k*5 if faiss_index is not None else top_k)

    fused = reciprocal_rank_fusion(tfidf_scores, faiss_scores, top_k=top_k, rrf_k=60)

    results = []
    for idx, fused_score, tfidf_rank, faiss_rank in fused:
        doc = reg_docs[idx]
        tfidf_score_val = next((s for (i, s) in tfidf_scores if i == idx), 0.0)
        faiss_score_val = next((s for (i, s) in faiss_scores if i == idx), 0.0)
        results.append({
            'title': doc.get('title', f'doc_{idx}'),
            'text': doc.get('text', ''),
            'score': float(fused_score),
            'source_scores': {
                'tfidf': float(tfidf_score_val),
                'faiss': float(faiss_score_val),
                'tfidf_rank': int(tfidf_rank) if tfidf_rank is not None else None,
                'faiss_rank': int(faiss_rank) if faiss_rank is not None else None
            }
        })
    # If fused is empty (no ranks), fallback to top tfidf or top reg_docs
    if not results:
        fallback = []
        for i, d in enumerate(reg_docs[:top_k]):
            fallback.append({'title': d.get('title',''), 'text': d.get('text',''), 'score': 0.0, 'source_scores': {}})
        return fallback
    return results
