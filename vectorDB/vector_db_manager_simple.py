import logging
import os
from typing import List, Dict, Optional, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorDBManager:
    """简化版向量数据库管理器（不依赖 CLIP）"""
    
    def __init__(self, persist_dir: str = "./data/vector_db", collection_name: str = "campus_targets"):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                persist_directory=str(self.persist_dir),
                anonymized_telemetry=False
            )
        )
        
        self.collection = self._get_or_create_collection()
        logger.info("VectorDBManager initialized successfully")
    
    def _get_or_create_collection(self):
        try:
            # 尝试获取现有集合
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                collection = self.client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists")
                return collection
            else:
                # 集合不存在，创建新集合
                raise ValueError("Collection not found")
        except Exception as e:
            logger.info(f"Creating new collection: {self.collection_name}")
            # 创建集合时指定余弦距离
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection created with cosine similarity and HNSW index")
            return collection
    
    def batch_upsert(self, items: List[Dict[str, Any]]) -> int:
        if not items or len(items) == 0:
            raise ValueError("入库数据列表不能为空")
        
        logger.info(f"Processing {len(items)} items for batch upsert")
        
        valid_items = []
        for item in items:
            try:
                if "id" not in item or "embedding" not in item or "metadata" not in item:
                    raise ValueError("缺少必填字段")
                if len(item["embedding"]) != 512:
                    raise ValueError(f"特征维度不匹配")
                valid_items.append(item)
            except ValueError as e:
                logger.warning(f"Skipping invalid item: {str(e)}")
        
        if len(valid_items) == 0:
            logger.warning("No valid items to upsert")
            return 0
        
        ids = [item["id"] for item in valid_items]
        embeddings = [item["embedding"] for item in valid_items]
        metadatas = [item["metadata"] for item in valid_items]
        
        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        
        count = len(valid_items)
        logger.info(f"Successfully upserted {count} items")
        return count
    
    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "vector_count": self.collection.count(),
        }
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """向量检索接口"""
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        formatted_results = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
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