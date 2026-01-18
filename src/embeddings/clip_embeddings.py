"""
CLIP Embedding Functions
Handles text and image embedding using CLIP model
"""
from PIL import Image
import torch
import config


def embed_text(text, clip_model, clip_processor):
    """
    Embed text using CLIP model
    
    Args:
        text (str): Text to embed
        clip_model: CLIP model instance
        clip_processor: CLIP processor instance
    
    Returns:
        numpy.ndarray: Normalized text embedding
    """
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.CLIP_MAX_TOKEN_LENGTH
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        # Normalize embeddings to unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


def embed_image(image_data, clip_model, clip_processor):
    """
    Embed image using CLIP model
    
    Args:
        image_data: PIL Image or path to image file
        clip_model: CLIP model instance
        clip_processor: CLIP processor instance
    
    Returns:
        numpy.ndarray: Normalized image embedding
    """
    # Handle both file path and PIL Image
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        # Normalize embeddings to unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

