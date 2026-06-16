"""Batch CLIP feature extraction for YOLO crop metadata.

This is CLIP1 output only. It does not write to Chroma/vectorDB. CLIP2 can read
this JSONL or call build_feature_records directly to create vectorDB items.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .config import CLIPConfig
from .descriptions import build_text_desc
from .encoder import CLIPEncoder


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yolo_metadata(metadata_path: str | Path) -> list[dict[str, Any]]:
    """Load YOLO metadata from a JSON file.

    Supports either a plain list of detections or a dict containing a common
    key such as detections/results/items/data.
    """
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("detections", "results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(
        "metadata JSON must be a list or a dict containing one of: "
        "detections, results, items, data"
    )


def _safe_id(det: dict[str, Any], index: int) -> str:
    value = det.get("image_id") or det.get("id") or det.get("target_id")
    return str(value if value is not None else f"target_{index:06d}")


def _safe_image_path(det: dict[str, Any]) -> str:
    value = det.get("image_path") or det.get("crop_path") or det.get("path")
    if not value:
        raise ValueError(f"Cannot find image path in detection item: {det}")
    return str(value)


def _safe_class_name(det: dict[str, Any]) -> str:
    value = det.get("class_name") or det.get("category") or det.get("label")
    if value is None:
        return "unknown"
    return str(value)


def _safe_confidence(det: dict[str, Any]) -> float:
    value = det.get("confidence", det.get("detection_conf", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_feature_records(
    metadata_path: str | Path,
    encoder: CLIPEncoder,
    *,
    include_text_embedding: bool = True,
) -> list[dict[str, Any]]:
    """Create CLIP feature records from YOLO detections.

    Each output record keeps YOLO metadata and adds:
    - image_embedding: normalized image vector
    - text_desc: domain description generated from category
    - text_embedding: normalized text vector, optional
    """
    detections = load_yolo_metadata(metadata_path)
    image_paths = [_safe_image_path(det) for det in detections]
    image_embeddings = encoder.encode_images(image_paths)

    class_names = [_safe_class_name(det) for det in detections]
    text_descs = [build_text_desc(name) for name in class_names]
    text_embeddings = encoder.encode_texts(text_descs) if include_text_embedding else [None] * len(detections)

    records: list[dict[str, Any]] = []
    for index, det in enumerate(detections):
        records.append(
            {
                "id": _safe_id(det, index),
                "image_path": image_paths[index],
                "class_name": class_names[index],
                "confidence": _safe_confidence(det),
                "bbox": det.get("bbox"),
                "text_desc": text_descs[index],
                "image_embedding": image_embeddings[index],
                "text_embedding": text_embeddings[index],
            }
        )
    return records


def save_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save feature records to JSON Lines."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON Lines feature records."""
    return list(_iter_jsonl(Path(path)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch extract CLIP features from YOLO crop metadata.")
    parser.add_argument("--metadata", required=True, help="YOLO detection metadata JSON path")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--model-name", default=None, help="Hugging Face model name or local path")
    parser.add_argument("--model-type", default=None, choices=["auto", "openai", "clip", "chinese"], help="Model family")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for inference")
    parser.add_argument(
        "--no-text-embedding",
        action="store_true",
        help="Only extract image embeddings; keep text_desc but skip text_embedding",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = CLIPConfig.from_env()
    encoder = CLIPEncoder(
        config,
        model_name=args.model_name,
        model_type=args.model_type,
        device=args.device,
        batch_size=args.batch_size,
    )
    records = build_feature_records(
        args.metadata,
        encoder,
        include_text_embedding=not args.no_text_embedding,
    )
    save_jsonl(records, args.out)
    print(f"Saved {len(records)} CLIP feature records to {args.out}")


if __name__ == "__main__":
    main()
