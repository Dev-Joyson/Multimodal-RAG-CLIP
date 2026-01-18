"""
Model Loading Module
Handles initialization of CLIP and LLM models
"""
from transformers import CLIPProcessor, CLIPModel
from langchain_openai import ChatOpenAI
import config


def load_clip_model():
    """
    Load and initialize CLIP model and processor
    
    Returns:
        tuple: (clip_model, clip_processor)
    """
    clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    clip_model.eval()
    return clip_model, clip_processor


def load_llm():
    """
    Load and initialize the LLM (GPT-4o)
    
    Returns:
        ChatOpenAI: Initialized LLM instance
    """
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE
    )
    return llm

