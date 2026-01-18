"""
Multimodal PDF RAG Application with CLIP and GPT-4V
Streamlit UI for asking questions about PDF documents
"""
import streamlit as st
from dotenv import load_dotenv

# Import configuration
import config

# Import our modules
from src.models import load_clip_model, load_llm
from src.processing import process_pdf
from src.retrieval import create_vector_store
from src.rag import ask_question

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
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


@st.cache_resource(show_spinner=False)
def load_models():
    """Load CLIP model and LLM (cached for performance)"""
    clip_model, clip_processor = load_clip_model()
    llm = load_llm()
    return clip_model, clip_processor, llm


# Main UI
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title(config.APP_TITLE)
st.markdown(config.APP_SUBTITLE)
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
            image_data_store, all_docs, all_embeddings = process_pdf(
                uploaded_file,
                st.session_state.clip_model,
                st.session_state.clip_processor
            )
            
            # Create vector store
            vector_store = create_vector_store(all_docs, all_embeddings)
            
            # Store in session state
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
        placeholder=config.QUESTION_PLACEHOLDER,
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
