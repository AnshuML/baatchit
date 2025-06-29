from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import os
import pickle

from config import *
from utils.llm_handler import embed_fn as embed_text

# Load ChromaDB
vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embed_text)

# Build BM25 index
bm25_path = os.path.join(INDEX_DIR, "bm25.pkl")

if os.path.exists(bm25_path):
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
else:
    result = vectordb._collection.get(include=["documents"])
    documents = result["documents"]
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

def hybrid_search(query, top_k=7):
    """Search with policy-specific boosting and metadata filtering"""
    query = query.lower()
    
    # Policy type detection
    policy_types = {
        "code": ["code", "conduct", "ethics"],
        "leave": ["leave", "holiday", "maternity", "casual"],
        "induction": ["induction", "joining", "onboarding"]
    }

    matched_policy = None
    for p_type, keywords in policy_types.items():
        if any(k in query for k in keywords):
            matched_policy = p_type
            break

    # Semantic Search
    dense_results = vectordb.similarity_search(query, k=top_k * 2)
    filtered_dense_results = dense_results
    if matched_policy:
        filtered_dense_results = [r for r in dense_results if r.metadata.get("policy_type") == matched_policy]
        # Fallback: if nothing found, use all dense_results
        if not filtered_dense_results:
            filtered_dense_results = dense_results
    filtered_dense_results = filtered_dense_results[:top_k]

    # Keyword Search
    tokenized_query = query.split(" ")
    bm_scores = bm25.get_scores(tokenized_query)
    bm_top_idx = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:top_k]
    
    try:
        sparse_texts = [doc.page_content for doc in vectordb._collection.get(
            ids=[str(i) for i in bm_top_idx], 
            include=["documents"]
        )["documents"]]
    except:
        sparse_texts = []

    # Deduplicate and prioritize relevant chunks
    combined_texts = list(set([r.page_content for r in filtered_dense_results] + sparse_texts))
    combined_texts.sort(key=lambda x: any(word in x.lower() for word in query.split()), reverse=True)

    return combined_texts[:top_k]
        
   
