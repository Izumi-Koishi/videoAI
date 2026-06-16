"""
共享数据类
定义系统中跨模块使用的数据结构。
"""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """向量检索结果"""
    target_id: int
    class_name: str
    confidence: float
    similarity: float
    description: str
    image_path: str
    bbox: tuple