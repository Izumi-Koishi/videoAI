"""
向量数据库模块 - 负责文本特征提取、向量存储与检索
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional


class VectorStore:
    """基于 ChromaDB 的向量存储与检索"""

    def __init__(self, persist_dir: str = "./vector_db"):
        """
        初始化向量数据库
        Args:
            persist_dir: 数据持久化目录
        """
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._init_store()

    def _init_store(self):
        """初始化 ChromaDB 和嵌入模型"""
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            print("[VectorStore] ChromaDB 初始化成功")
        except Exception as e:
            print(f"[VectorStore] ChromaDB 初始化失败: {e}")
            self.client = None

        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[VectorStore] 嵌入模型加载成功 (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"[VectorStore] 嵌入模型加载失败: {e}")
            self.embedding_model = None

    def is_ready(self) -> bool:
        """检查是否就绪"""
        return self.client is not None

    def _get_or_create_collection(self, video_id: str):
        """获取或创建集合"""
        collection_name = f"video_{video_id.replace('-', '_').replace('.', '_')}"
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"[VectorStore] 创建集合失败: {e}")
            self.collection = None

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """将文本编码为向量"""
        if self.embedding_model is not None:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        else:
            # 回退方案：使用 ChromaDB 默认嵌入
            return None

    def add_detection_results(
        self,
        video_id: str,
        detection_results: list,
        detection_summary: str
    ):
        """
        将检测结果存入向量数据库
        Args:
            video_id: 视频标识
            detection_results: YOLO 检测结果列表
            detection_summary: 检测摘要文本
        """
        if not self.is_ready():
            print("[VectorStore] 数据库未就绪，跳过存储")
            return

        self._get_or_create_collection(video_id)
        if self.collection is None:
            return

        # 将每帧检测结果作为独立文档存储
        documents = []
        metadatas = []
        ids = []

        for i, result in enumerate(detection_results):
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            frame_idx = result.get("frame_index", 0)
            timestamp = result.get("timestamp", 0)
            saved_image = result.get("saved_image", "")
            cropped_objects = result.get("cropped_objects", [])

            if not labels:
                continue

            # 构建文档文本
            label_count = {}
            for label, score in zip(labels, scores):
                if label not in label_count:
                    label_count[label] = {"count": 0, "max_score": 0}
                label_count[label]["count"] += 1
                label_count[label]["max_score"] = max(
                    label_count[label]["max_score"], score
                )

            doc_text = f"时间戳{timestamp:.1f}秒(第{frame_idx}帧): 检测到 " + ", ".join(
                [f"{label}({info['count']}个, 置信度{info['max_score']:.2f})"
                 for label, info in label_count.items()]
            )

            # 裁剪目标信息
            crop_paths = [obj["image_path"] for obj in cropped_objects]

            documents.append(doc_text)
            metadatas.append({
                "frame_index": frame_idx,
                "timestamp": timestamp,
                "saved_image": saved_image,
                "crop_paths": json.dumps(crop_paths, ensure_ascii=False),
                "labels": json.dumps(labels, ensure_ascii=False)
            })
            ids.append(f"frame_{i}")

        # 也添加整体摘要
        if detection_summary:
            documents.append(f"视频整体检测摘要:\n{detection_summary}")
            metadatas.append({
                "frame_index": -1,
                "timestamp": -1,
                "saved_image": "",
                "crop_paths": "[]",
                "labels": "[]"
            })
            ids.append("summary")

        if not documents:
            print("[VectorStore] 没有可存储的文档")
            return

        # 编码并存储
        embeddings = self._encode(documents)

        try:
            if embeddings is not None:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                # 使用 ChromaDB 默认嵌入函数
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            print(f"[VectorStore] 成功存储 {len(documents)} 条文档")
        except Exception as e:
            print(f"[VectorStore] 存储失败: {e}")

    def search(
        self,
        video_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        检索与查询最相关的文档
        Args:
            video_id: 视频标识
            query: 查询文本
            top_k: 返回最相关的 k 条结果
        Returns:
            检索结果列表
        """
        if not self.is_ready():
            return []

        self._get_or_create_collection(video_id)
        if self.collection is None:
            return []

        # 编码查询
        query_embedding = self._encode([query])
        if query_embedding is not None:
            query_embedding = query_embedding[0]

        try:
            count = self.collection.count()
            if count == 0:
                return []
            n_results = min(top_k, count)
            if query_embedding is not None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
        except Exception as e:
            print(f"[VectorStore] 检索失败: {e}")
            return []

        # 格式化结果
        formatted = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                item = {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                }
                formatted.append(item)

        return formatted

    def get_context_and_images(
        self,
        video_id: str,
        query: str,
        top_k: int = 5
    ) -> Tuple[str, List[str]]:
        """
        获取检索到的上下文文本和匹配图片路径
        Returns:
            (context_text, image_paths)
        """
        results = self.search(video_id, query, top_k)

        context_parts = []
        image_paths = []

        for item in results:
            doc = item["document"]
            metadata = item["metadata"]

            context_parts.append(doc)

            # 收集检测标注图
            saved_img = metadata.get("saved_image", "")
            if saved_img and os.path.exists(saved_img) and saved_img not in image_paths:
                image_paths.append(saved_img)

            # 收集裁剪目标图
            try:
                crop_paths = json.loads(metadata.get("crop_paths", "[]"))
                for cp in crop_paths:
                    if os.path.exists(cp) and cp not in image_paths:
                        image_paths.append(cp)
            except (json.JSONDecodeError, TypeError):
                pass

        context_text = "\n".join(context_parts)
        return context_text, image_paths
