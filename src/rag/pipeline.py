"""
RAG Pipeline Module
Main orchestration of the RAG (Retrieval-Augmented Generation) pipeline
"""
from langchain_core.messages import HumanMessage
from src.retrieval import retrieve_multimodal


def create_multimodal_message(query, retrieved_docs, image_data_store):
    """
    Create a multimodal message with both text and images for GPT-4V
    
    Args:
        query (str): User's question
        retrieved_docs (list): List of retrieved Document objects
        image_data_store (dict): Dictionary mapping image IDs to base64 encoded images
    
    Returns:
        HumanMessage: Formatted message for LLM with text and images
    """
    content = []
    
    # Add the query
    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n"
    })
    
    # Separate text and image documents
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    # Add text context
    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })
    
    # Add images
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}"
                }
            })
    
    # Add instruction
    content.append({
        "type": "text",
        "text": "\n\nPlease answer the question based on the provided text and images."
    })
    
    return HumanMessage(content=content)


def ask_question(query, vector_store, image_data_store, llm, clip_model, clip_processor):
    """
    Main RAG pipeline - retrieve context and generate answer
    
    Args:
        query (str): User's question
        vector_store: FAISS vector store instance
        image_data_store (dict): Dictionary of image IDs to base64 images
        llm: Language model instance
        clip_model: CLIP model instance
        clip_processor: CLIP processor instance
    
    Returns:
        str: Generated answer from the LLM
    """
    # Retrieve relevant documents
    context_docs = retrieve_multimodal(query, vector_store, clip_model, clip_processor)
    
    # Create multimodal message
    message = create_multimodal_message(query, context_docs, image_data_store)
    
    # Get response from LLM
    response = llm.invoke([message])
    
    return response.content

