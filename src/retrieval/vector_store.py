"""
Vector Store Module
Handles FAISS vector store creation and operations
"""
import numpy as np
from langchain_community.vectorstores import FAISS


def create_vector_store(all_docs, all_embeddings):
    """
    Create a FAISS vector store from documents and embeddings
    
    Args:
        all_docs (list): List of Document objects
        all_embeddings (list): List of embeddings (numpy arrays)
    
    Returns:
        FAISS: Initialized FAISS vector store
    """
    embeddings_array = np.array(all_embeddings)
    
    vector_store = FAISS.from_embeddings(
        text_embeddings=[
            (doc.page_content, emb) 
            for doc, emb in zip(all_docs, embeddings_array)
        ],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs]
    )
    
    return vector_store

