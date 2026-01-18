"""
Configuration file for Multimodal RAG with CLIP
All settings and constants in one place
"""

# Model Configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "gpt-4.1"
LLM_TEMPERATURE = 0

# Text Processing Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CLIP_MAX_TOKEN_LENGTH = 77

# Retrieval Configuration
TOP_K_RESULTS = 5

# File Configuration
TEMP_PDF_PATH = "temp.pdf"
MAX_FILE_SIZE_MB = 200

# UI Configuration
PAGE_TITLE = "Multimodal RAG with CLIP"
APP_TITLE = "Ask your PDF 📚"
APP_SUBTITLE = "Upload your document and ask questions about it"
QUESTION_PLACEHOLDER = ""

