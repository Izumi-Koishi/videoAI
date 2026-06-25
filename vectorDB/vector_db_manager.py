import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
import numpy as np

from config import VECTOR_DB_CONFIG, SEARCH_CONFIG

logger = logging.getLogger(__name__)

class VectorDBManager:
    """
    向量数据库核心管理类
    
    提供数据库初始化、批量入库、向量检索、索引管理等功能
    基于 Chroma PersistentClient 实现本地持久化存储
    """
    
    def __init__(self, persist_dir: str = None, collection_name: str = None):
        """
        初始化向量数据库管理器
        
        Args:
            persist_dir: 数据持久化目录，默认为配置中的路径
            collection_name: 集合名称，默认为配置中的名称
        """
        self.persist_dir = Path(persist_dir or VECTOR_DB_CONFIG["persist_directory"])
        self.collection_name = collection_name or VECTOR_DB_CONFIG["collection_name"]
        self.feature_dim = VECTOR_DB_CONFIG["feature_dim"]
        
        # 创建数据目录
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Chroma 客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                persist_directory=str(self.persist_dir),
                anonymized_telemetry=False
            )
        )
        
        # 获取或创建集合
        self.collection = self._get_or_create_collection()
        
        logger.info("VectorDBManager initialized successfully")
    
    def _get_or_create_collection(self):
        """
        获取或创建集合
        
        首次创建时启用 HNSW 索引，距离函数为余弦相似度
        """
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                collection = self.client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists")
                return collection
            else:
                raise ValueError("Collection not found")
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection created with cosine similarity and HNSW index")
            return collection
    
    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        判断集合是否存在
        
        Args:
            collection_name: 集合名称，默认为配置中的集合名
            
        Returns:
            集合是否存在
        """
        name = collection_name or self.collection_name
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            return name in collection_names
        except Exception:
            return False
    
    def delete_collection(self, collection_name: Optional[str] = None):
        """
        删除指定集合
        
        Args:
            collection_name: 集合名称，默认为配置中的集合名
        """
        name = collection_name or self.collection_name
        if self.collection_exists(name):
            self.client.delete_collection(name)
            logger.info(f"Collection '{name}' deleted")
        else:
            logger.warning(f"Collection '{name}' does not exist")
    
    def reset_collection(self):
        """重置集合（删除后重新创建）"""
        self.delete_collection()
        self.collection = self._get_or_create_collection()
        logger.info("Collection reset completed")
    
    def _validate_single_item(self, item: Dict[str, Any]) -> bool:
        """
        校验单条入库数据
        
        Args:
            item: 待校验的数据项
            
        Returns:
            校验是否通过
            
        Raises:
            ValueError: 校验失败时抛出
        """
        required_fields = ["id", "embedding", "metadata"]
        for field in required_fields:
            if field not in item:
                raise ValueError(f"缺少必填字段: {field}")
        
        if len(item["embedding"]) != self.feature_dim:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.feature_dim} 维，实际 {len(item['embedding'])} 维"
            )
        
        required_metadata = ["category", "text_desc", "image_path", "detection_conf", "frame_index"]
        for field in required_metadata:
            if field not in item["metadata"]:
                raise ValueError(f"metadata 缺少必填字段: {field}")
        
        if not (0.0 <= item["metadata"]["detection_conf"] <= 1.0):
            raise ValueError("detection_conf 必须在 0.0-1.0 范围内")
        
        return True
    
    def batch_upsert(self, items: List[Dict[str, Any]]) -> int:
        """
        批量插入/更新向量数据
        
        Args:
            items: 待入库的数据列表，每个元素包含 id, embedding, metadata
            
        Returns:
            成功入库的数据条数
            
        Raises:
            ValueError: 数据校验失败
        """
        if not items or len(items) == 0:
            raise ValueError("入库数据列表不能为空")
        
        logger.info(f"Processing {len(items)} items for batch upsert")
        
        valid_items = []
        for item in items:
            try:
                self._validate_single_item(item)
                valid_items.append(item)
            except ValueError as e:
                logger.warning(f"Skipping invalid item: {str(e)}")
        
        if len(valid_items) == 0:
            logger.warning("No valid items to upsert")
            return 0
        
        ids = [item["id"] for item in valid_items]
        embeddings = [item["embedding"] for item in valid_items]
        metadatas = [item["metadata"] for item in valid_items]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        count = len(valid_items)
        logger.info(f"Successfully upserted {count} items")
        
        return count
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            集合统计信息，包含向量数量等
        """
        return {
            "collection_name": self.collection_name,
            "vector_count": self.collection.count(),
        }
    
    def search(self, query_embedding: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        """
        向量检索接口
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量，默认为配置中的默认值
            
        Returns:
            按相似度降序排列的检索结果列表
        """
        top_k = top_k or SEARCH_CONFIG["default_top_k"]
        top_k = min(top_k, SEARCH_CONFIG["max_top_k"])
        
        logger.info(f"Vector search: top_k={top_k}")
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return self._format_results(results)
    
    def _format_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        格式化检索结果
        
        Args:
            raw_results: Chroma 查询原始结果
            
        Returns:
            标准化的结果列表
        """
        formatted_results = []
        
        ids = raw_results.get("ids", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        
        for i, _id in enumerate(ids):
            score = 1.0 - distances[i] if distances else 0.0
            metadata = metadatas[i] if metadatas else {}
            
            result = {
                "id": _id,
                "score": round(score, 4),
                "category": metadata.get("category", ""),
                "text_desc": metadata.get("text_desc", ""),
                "image_path": metadata.get("image_path", ""),
                "detection_conf": metadata.get("detection_conf", 0.0)
            }
            formatted_results.append(result)
        
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        
        return formatted_results
    
    @staticmethod
    def generate_random_embedding(dim: int = 512) -> List[float]:
        """
        生成随机归一化特征向量（模拟 CLIP 输出）
        
        Args:
            dim: 特征维度，默认为 512
            
        Returns:
            归一化后的特征向量
        """
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()