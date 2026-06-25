"""Image-text semantic alignment evaluation for CLIP1."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .batch_extract import load_yolo_metadata
from .config import CLIPConfig
from .descriptions import build_text_desc
from .encoder import CLIPEncoder, cosine_similarity


def _class_name(det: dict[str, Any]) -> str:
    return str(det.get("class_name") or det.get("category") or det.get("label") or "unknown")


def _image_path(det: dict[str, Any]) -> str:
    value = det.get("image_path") or det.get("crop_path") or det.get("path")
    if not value:
        raise ValueError(f"Cannot find image path in item: {det}")
    return str(value)


def _choose_negative_class(class_names: list[str], current: str) -> str | None:
    for name in class_names:
        if name != current:
            return name
    return None


def evaluate_yolo_metadata(
    metadata_path: str | Path,
    encoder: CLIPEncoder,
    *,
    sample_per_class: int = 3,
    same_threshold: float = 0.7,
    diff_threshold: float = 0.3,
) -> dict[str, Any]:
    """Evaluate same-class and different-class image-text similarities."""
    detections = load_yolo_metadata(metadata_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for det in detections:
        grouped[_class_name(det)].append(det)

    class_names = sorted(grouped)
    same_pairs: list[dict[str, Any]] = []
    diff_pairs: list[dict[str, Any]] = []

    for class_name in class_names:
        samples = grouped[class_name][:sample_per_class]
        negative_class = _choose_negative_class(class_names, class_name)
        for det in samples:
            img_path = _image_path(det)
            image_vec = encoder.encode_image(img_path)

            positive_text = build_text_desc(class_name)
            positive_vec = encoder.encode_text(positive_text)
            same_score = cosine_similarity(image_vec, positive_vec)
            same_pairs.append(
                {
                    "image_path": img_path,
                    "class_name": class_name,
                    "text_desc": positive_text,
                    "similarity": round(same_score, 6),
                    "passed": same_score >= same_threshold,
                }
            )

            if negative_class is not None:
                negative_text = build_text_desc(negative_class)
                negative_vec = encoder.encode_text(negative_text)
                diff_score = cosine_similarity(image_vec, negative_vec)
                diff_pairs.append(
                    {
                        "image_path": img_path,
                        "image_class": class_name,
                        "negative_class": negative_class,
                        "text_desc": negative_text,
                        "similarity": round(diff_score, 6),
                        "passed": diff_score <= diff_threshold,
                    }
                )

    same_scores = [item["similarity"] for item in same_pairs]
    diff_scores = [item["similarity"] for item in diff_pairs]
    same_avg = float(np.mean(same_scores)) if same_scores else 0.0
    diff_avg = float(np.mean(diff_scores)) if diff_scores else 0.0

    return {
        "metadata_path": str(metadata_path),
        "classes": class_names,
        "sample_per_class": sample_per_class,
        "thresholds": {
            "same_class_min": same_threshold,
            "different_class_max": diff_threshold,
        },
        "summary": {
            "same_pair_count": len(same_pairs),
            "different_pair_count": len(diff_pairs),
            "same_avg": round(same_avg, 6),
            "different_avg": round(diff_avg, 6),
            "same_pass_rate": round(sum(x["passed"] for x in same_pairs) / len(same_pairs), 6) if same_pairs else 0.0,
            "different_pass_rate": round(sum(x["passed"] for x in diff_pairs) / len(diff_pairs), 6) if diff_pairs else 0.0,
        },
        "same_pairs": same_pairs,
        "different_pairs": diff_pairs,
    }


def save_report(report: dict[str, Any], output_path: str | Path) -> None:
    """Save report as JSON or Markdown based on file suffix."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".md":
        lines = [
            "# CLIP1 图文语义对齐验证报告",
            "",
            f"- 元数据文件：`{report['metadata_path']}`",
            f"- 类别数：{len(report['classes'])}",
            f"- 每类采样数：{report['sample_per_class']}",
            f"- 同类阈值：>= {report['thresholds']['same_class_min']}",
            f"- 异类阈值：<= {report['thresholds']['different_class_max']}",
            "",
            "## 汇总",
            "",
            f"- 同类样本数：{report['summary']['same_pair_count']}",
            f"- 异类样本数：{report['summary']['different_pair_count']}",
            f"- 同类平均相似度：{report['summary']['same_avg']}",
            f"- 异类平均相似度：{report['summary']['different_avg']}",
            f"- 同类通过率：{report['summary']['same_pass_rate']}",
            f"- 异类通过率：{report['summary']['different_pass_rate']}",
            "",
            "## 同类图文样本",
            "",
            "| image_path | class_name | text_desc | similarity | passed |",
            "|---|---|---|---:|---|",
        ]
        for item in report["same_pairs"]:
            lines.append(
                f"| `{item['image_path']}` | {item['class_name']} | {item['text_desc']} | "
                f"{item['similarity']} | {item['passed']} |"
            )
        lines.extend(
            [
                "",
                "## 异类图文样本",
                "",
                "| image_path | image_class | negative_class | text_desc | similarity | passed |",
                "|---|---|---|---|---:|---|",
            ]
        )
        for item in report["different_pairs"]:
            lines.append(
                f"| `{item['image_path']}` | {item['image_class']} | {item['negative_class']} | "
                f"{item['text_desc']} | {item['similarity']} | {item['passed']} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CLIP image-text alignment on YOLO crop metadata.")
    parser.add_argument("--metadata", required=True, help="YOLO detection metadata JSON path")
    parser.add_argument("--out", required=True, help="Output report path, .md or .json")
    parser.add_argument("--model-name", default=None, help="Hugging Face model name or local path")
    parser.add_argument("--model-type", default=None, choices=["auto", "openai", "clip", "chinese"], help="Model family")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for inference")
    parser.add_argument("--sample-per-class", type=int, default=3)
    parser.add_argument("--same-threshold", type=float, default=0.7)
    parser.add_argument("--diff-threshold", type=float, default=0.3)
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
    report = evaluate_yolo_metadata(
        args.metadata,
        encoder,
        sample_per_class=args.sample_per_class,
        same_threshold=args.same_threshold,
        diff_threshold=args.diff_threshold,
    )
    save_report(report, args.out)
    print(f"Saved CLIP1 similarity report to {args.out}")


if __name__ == "__main__":
    main()
