"""Domain text descriptions for YOLO categories.

CLIP alignment is very sensitive to prompt wording. Keep these descriptions
short, visual and domain-specific. You can extend this file when your dataset
category list becomes fixed.
"""

from __future__ import annotations

from typing import Mapping

DEFAULT_TEXT_TEMPLATES: dict[str, str] = {
    "person": "校园场景中的行人",
    "car": "校园道路上的汽车",
    "bus": "校园道路或校门附近的公交车",
    "truck": "校园道路上的货车",
    "bicycle": "校园场景中的自行车",
    "motorcycle": "校园道路上的摩托车",
    "backpack": "学生携带的双肩背包",
    "handbag": "校园场景中的手提包",
    "book": "学习场景中的书本",
    "cell phone": "行人手中的手机",
    "laptop": "学习或办公场景中的笔记本电脑",
    "chair": "教室或校园场景中的椅子",
    "bench": "校园道路旁或操场附近的长椅",
    "bottle": "桌面或地面上的水瓶",
    "cup": "桌面上的杯子",
    "traffic light": "道路交叉口的交通信号灯",
    "stop sign": "道路旁的停车标志",
    "fire hydrant": "校园或道路旁的消防设施",
    # Common Chinese labels used by the mock Gradio module.
    "行人": "校园道路上的行人",
    "汽车": "校园道路上的汽车",
    "自行车": "校园场景中的自行车",
    "摩托车": "校园道路上的摩托车",
    "公交车": "校门口或道路上的公交车",
    "卡车": "校园道路上的货车",
    "交通灯": "道路交叉口的交通信号灯",
    "停车标志": "道路旁的停车标志",
    "长椅": "校园路边供人休息的长椅",
    "背包": "学生携带的双肩背包",
    "手提包": "行人携带的手提包",
    "手机": "行人手中的智能手机",
    "笔记本电脑": "学习或办公场景中的笔记本电脑",
    "书本": "学习场景中的书本",
    "瓶子": "桌面或地面上的水瓶",
    "杯子": "桌面上的杯子",
    "椅子": "教室或校园场景中的椅子",
    "桌子": "学习或办公场景中的桌子",
    "显示器": "桌面上的电脑显示器",
    "键盘": "电脑旁边的键盘",
}


def normalize_label(label: str) -> str:
    """Normalize labels before template lookup."""
    return str(label).strip()


def build_text_desc(class_name: str, templates: Mapping[str, str] | None = None) -> str:
    """Return a CLIP-friendly Chinese description for one detected category."""
    label = normalize_label(class_name)
    mapping = templates or DEFAULT_TEXT_TEMPLATES
    if label in mapping:
        return mapping[label]
    lower_label = label.lower()
    if lower_label in mapping:
        return mapping[lower_label]
    return f"校园场景中的{label}目标"
