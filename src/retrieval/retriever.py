"""
Retrieval Module
Handles document retrieval using vector similarity search
"""
import config
from src.embeddings import embed_text


def retrieve_multimodal(query, vector_store, clip_model, clip_processor, k=None):
    """
    Retrieve relevant documents using multimodal (text + image) similarity search
    
    Args:
        query (str): Query text
        vector_store: FAISS vector store instance
        clip_model: CLIP model instance
        clip_processor: CLIP processor instance
        k (int, optional): Number of documents to retrieve. Defaults to config.TOP_K_RESULTS
    
    Returns:
        list: List of retrieved Document objects
    """
    if k is None:
        k = config.TOP_K_RESULTS
    
    # Generate embedding for query
    query_embedding = embed_text(query, clip_model, clip_processor)
    
    # Search in vector store
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    
    return results

