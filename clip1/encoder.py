"""CLIP image/text feature extraction.

This file intentionally imports torch/transformers lazily so basic unit tests
can run without downloading a model. Instantiate CLIPEncoder only in runtime
scripts that actually need model inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .config import CLIPConfig

ArrayLike = Sequence[float] | np.ndarray


def l2_normalize(values: ArrayLike | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector or a batch of vectors.

    Args:
        values: 1D vector or 2D matrix.
        eps: Small value to avoid division by zero.

    Returns:
        NumPy array with the same rank as input and unit L2 norm on the last axis.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("values must be a vector or a matrix, got scalar")
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return arr / norm


def cosine_similarity(vector_a: ArrayLike, vector_b: ArrayLike) -> float:
    """Compute cosine similarity after L2 normalization."""
    a = l2_normalize(np.asarray(vector_a, dtype=np.float32).reshape(1, -1))[0]
    b = l2_normalize(np.asarray(vector_b, dtype=np.float32).reshape(1, -1))[0]
    return float(np.dot(a, b))


def _batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_rgb_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


class CLIPEncoder:
    """Unified encoder for OpenAI CLIP and Chinese CLIP models.

    Public methods always return Python list[float] / list[list[float]] so the
    output can be serialized or passed to the vector database module directly.
    """

    def __init__(
        self,
        config: CLIPConfig | None = None,
        *,
        model_name: str | None = None,
        model_type: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        base = config or CLIPConfig.from_env()
        self.config = CLIPConfig(
            model_name=model_name or base.model_name,
            model_type=model_type or base.model_type,
            device=device or base.device,
            batch_size=batch_size or base.batch_size,
            embedding_dim=base.embedding_dim,
            local_files_only=base.local_files_only,
            trust_remote_code=base.trust_remote_code,
        )
        self.device = self._resolve_device(self.config.device)
        self.model, self.processor = self._load_model_and_processor()
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _infer_model_type(model_name: str, model_type: str) -> str:
        if model_type and model_type != "auto":
            return model_type.lower()
        lowered = model_name.lower()
        if "chinese" in lowered or "ofa-sys" in lowered or "taiyi" in lowered:
            return "chinese"
        return "openai"

    def _load_model_and_processor(self):
        import torch  # noqa: F401 - imported to ensure device support is available

        model_kind = self._infer_model_type(self.config.model_name, self.config.model_type)
        kwargs = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
        }

        if model_kind == "chinese":
            try:
                from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            except ImportError as exc:
                raise ImportError(
                    "Chinese CLIP requires a recent transformers version that provides "
                    "ChineseCLIPModel and ChineseCLIPProcessor. Try: pip install -U transformers"
                ) from exc

            model = ChineseCLIPModel.from_pretrained(self.config.model_name, **kwargs)
            processor = ChineseCLIPProcessor.from_pretrained(self.config.model_name, **kwargs)
        elif model_kind in {"openai", "clip"}:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(self.config.model_name, **kwargs)
            processor = CLIPProcessor.from_pretrained(self.config.model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {self.config.model_type}")

        return model.to(self.device), processor

    def _to_device(self, inputs: dict):
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _postprocess_features(self, features) -> list[list[float]]:
        arr = features.detach().cpu().float().numpy()
        arr = l2_normalize(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {arr.shape}")
        return arr.astype(float).tolist()

    def encode_images(self, image_paths: Sequence[str | Path], batch_size: int | None = None) -> list[list[float]]:
        """Encode a batch of image paths into L2-normalized CLIP vectors."""
        import torch

        paths = list(image_paths)
        if not paths:
            return []
        size = batch_size or self.config.batch_size
        output: list[list[float]] = []
        for batch_paths in _batched(paths, size):
            images = [_load_rgb_image(path) for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = self._to_device(inputs)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            output.extend(self._postprocess_features(features))
        return output

    def encode_image(self, image_path: str | Path) -> list[float]:
        """Encode a single image into one L2-normalized vector."""
        return self.encode_images([image_path])[0]

    def encode_texts(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        """Encode a batch of texts into L2-normalized CLIP vectors."""
        import torch

        values = [str(text) for text in texts]
        if not values:
            return []
        size = batch_size or self.config.batch_size
        output: list[list[float]] = []
        for batch_texts in _batched(values, size):
            inputs = self.processor(
                text=list(batch_texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = self._to_device(inputs)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            output.extend(self._postprocess_features(features))
        return output

    def encode_text(self, text: str) -> list[float]:
        """Encode a single text into one L2-normalized vector."""
        return self.encode_texts([text])[0]

    def validate_embedding(self, embedding: Sequence[float], *, expected_dim: int | None = None) -> bool:
        """Validate vector dimensionality and L2 norm.

        Raises ValueError with an actionable message if invalid.
        """
        expected = expected_dim or self.config.embedding_dim
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"embedding must be 1D, got shape {arr.shape}")
        if arr.shape[0] != expected:
            raise ValueError(f"embedding dim mismatch: expected {expected}, got {arr.shape[0]}")
        norm = float(np.linalg.norm(arr))
        if not 0.99 <= norm <= 1.01:
            raise ValueError(f"embedding is not L2-normalized, norm={norm:.6f}")
        return True
