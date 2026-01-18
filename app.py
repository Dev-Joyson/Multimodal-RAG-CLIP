import streamlit as st
import fitz
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os
import base64
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG with CLIP",
    page_icon="📚",
    layout="centered"
)

# Custom CSS for clean UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'image_data_store' not in st.session_state:
    st.session_state.image_data_store = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'clip_model' not in st.session_state:
    st.session_state.clip_model = None
if 'clip_processor' not in st.session_state:
    st.session_state.clip_processor = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = ""
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""

# Functions
@st.cache_resource(show_spinner=False)
def load_models():
    """Load CLIP model and LLM"""
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return clip_model, clip_processor, llm

def embed_image(image_data, clip_model, clip_processor):
    """Embed an image using CLIP"""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text, clip_model, clip_processor):
    """Embed text using CLIP"""
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def process_pdf(pdf_file, clip_model, clip_processor):
    """Process uploaded PDF and extract text and images"""
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    doc = fitz.open("temp.pdf")
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    # progress_bar = st.progress(0)
    total_pages = len(doc)
    
    for i, page in enumerate(doc):
        # progress_bar.progress((i + 1) / total_pages)
        
        # Process text
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            
            for chunk in text_chunks:
                embedding = embed_text(chunk.page_content, clip_model, clip_processor)
                all_embeddings.append(embedding)
                all_docs.append(chunk)
        
        # Process images
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"page_{i}_img_{img_index}"
                
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = image_base64
                
                embedding = embed_image(pil_image, clip_model, clip_processor)
                all_embeddings.append(embedding)
                
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id}
                )
                all_docs.append(image_doc)
            except:
                continue
    
    doc.close()
    # progress_bar.empty()
    
    # Create FAISS vector store
    embeddings_array = np.array(all_embeddings)
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs]
    )
    
    os.remove("temp.pdf")
    
    return image_data_store, vector_store

def retrieve_multimodal(query, vector_store, clip_model, clip_processor, k=5):
    """Unified retrieval using CLIP"""
    query_embedding = embed_text(query, clip_model, clip_processor)
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    return results

def create_multimodal_message(query, retrieved_docs, image_data_store):
    """Create a message with both text and images for GPT-4V"""
    content = []
    
    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n"
    })
    
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })
    
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}"
                }
            })
    
    content.append({
        "type": "text",
        "text": "\n\nPlease answer the question based on the provided text and images."
    })
    
    return HumanMessage(content=content)

def ask_question(query, vector_store, image_data_store, llm, clip_model, clip_processor):
    """Main pipeline for multimodal RAG"""
    context_docs = retrieve_multimodal(query, vector_store, clip_model, clip_processor, k=5)
    message = create_multimodal_message(query, context_docs, image_data_store)
    response = llm.invoke([message])
    return response.content

# Main UI
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("Ask your PDF 📚")
st.markdown("Upload your document and ask questions about it")
st.markdown('</div>', unsafe_allow_html=True)

# File upload section - always visible
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

if uploaded_file is not None:
    # Check if this is a new file
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        
        with st.spinner("Reading your document..."):
            # Load models if not already loaded
            if st.session_state.clip_model is None:
                clip_model, clip_processor, llm = load_models()
                st.session_state.clip_model = clip_model
                st.session_state.clip_processor = clip_processor
                st.session_state.llm = llm
            
            # Process the PDF
            image_data_store, vector_store = process_pdf(
                uploaded_file,
                st.session_state.clip_model,
                st.session_state.clip_processor
            )
            
            st.session_state.image_data_store = image_data_store
            st.session_state.vector_store = vector_store
            st.session_state.processed = True
            st.session_state.pdf_name = uploaded_file.name
            
            st.rerun()

# Question section - show only if document is processed
if st.session_state.processed:
    st.markdown("---")
    st.write("Ask a question about the PDF")
    
    # Question input - no form, question stays visible
    question = st.text_input(
        "Your question:",
        placeholder="",
        label_visibility="collapsed",
        key="question_input"
    )
    
    # Check if question changed and is not empty
    if question and question != st.session_state.last_question:
        st.session_state.last_question = question
        with st.spinner("Thinking..."):
            answer = ask_question(
                question,
                st.session_state.vector_store,
                st.session_state.image_data_store,
                st.session_state.llm,
                st.session_state.clip_model,
                st.session_state.clip_processor
            )
            st.session_state.current_answer = answer
    
    # Display answer if exists
    if st.session_state.current_answer:
        st.write(st.session_state.current_answer)