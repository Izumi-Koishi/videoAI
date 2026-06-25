"""CLIP1 feature extraction module for the YOLO-CLIP video AI project."""

from .config import CLIPConfig
from .descriptions import build_text_desc
from .encoder import CLIPEncoder, cosine_similarity, l2_normalize

__all__ = [
    "CLIPConfig",
    "CLIPEncoder",
    "build_text_desc",
    "cosine_similarity",
    "l2_normalize",
]
