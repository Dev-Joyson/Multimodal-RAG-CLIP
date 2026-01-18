"""
PDF Processing Module
Handles extraction and processing of text and images from PDF files
"""
import fitz
from PIL import Image
import base64
import io
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from src.embeddings import embed_text, embed_image


def process_pdf(pdf_file, clip_model, clip_processor):
    """
    Process uploaded PDF and extract text and images
    
    Args:
        pdf_file: Uploaded PDF file object
        clip_model: CLIP model instance
        clip_processor: CLIP processor instance
    
    Returns:
        tuple: (image_data_store, all_docs, all_embeddings)
            - image_data_store (dict): Base64 encoded images with their IDs
            - all_docs (list): List of Document objects
            - all_embeddings (list): List of embeddings
    """
    # Save uploaded file temporarily
    with open(config.TEMP_PDF_PATH, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Open PDF
    doc = fitz.open(config.TEMP_PDF_PATH)
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    # Process each page
    for page_num, page in enumerate(doc):
        # Extract and process text
        text = page.get_text()
        if text.strip():
            temp_doc = Document(
                page_content=text,
                metadata={"page": page_num, "type": "text"}
            )
            text_chunks = splitter.split_documents([temp_doc])
            
            for chunk in text_chunks:
                embedding = embed_text(chunk.page_content, clip_model, clip_processor)
                all_embeddings.append(embedding)
                all_docs.append(chunk)
        
        # Extract and process images
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"page_{page_num}_img_{img_index}"
                
                # Store image as base64 for GPT-4V
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = image_base64
                
                # Generate embedding for image
                embedding = embed_image(pil_image, clip_model, clip_processor)
                all_embeddings.append(embedding)
                
                # Create document for image
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": page_num, "type": "image", "image_id": image_id}
                )
                all_docs.append(image_doc)
            except Exception as e:
                # Skip problematic images
                continue
    
    # Clean up
    doc.close()
    os.remove(config.TEMP_PDF_PATH)
    
    return image_data_store, all_docs, all_embeddings

