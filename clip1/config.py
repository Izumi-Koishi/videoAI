"""Configuration for the CLIP1 feature extraction module."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class CLIPConfig:
    """Runtime settings for CLIPEncoder.

    Defaults use OpenAI CLIP because it is widely supported by Hugging Face
    Transformers. For Chinese text prompts, use the Chinese CLIP example in
    README.md.
    """

    model_name: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    model_type: str = os.getenv("CLIP_MODEL_TYPE", "auto")
    device: str = os.getenv("CLIP_DEVICE", "auto")
    batch_size: int = int(os.getenv("CLIP_BATCH_SIZE", "32"))
    embedding_dim: int = int(os.getenv("CLIP_EMBEDDING_DIM", "512"))
    local_files_only: bool = os.getenv("CLIP_LOCAL_FILES_ONLY", "0") == "1"
    trust_remote_code: bool = os.getenv("CLIP_TRUST_REMOTE_CODE", "0") == "1"

    @classmethod
    def from_env(cls) -> "CLIPConfig":
        """Build a config from environment variables."""
        return cls()
