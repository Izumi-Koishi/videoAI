import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 向量数据库配置
VECTOR_DB_CONFIG = {
    "persist_directory": str(PROJECT_ROOT / "data" / "vector_db"),
    "collection_name": "campus_targets",
    "feature_dim": 512,  # 特征向量维度
}

# HNSW 索引参数配置
HNSW_CONFIG = {
    "M": 16,              
    "ef_construction": 200,  
    "ef_search": 50,      
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# 入库配置
INGEST_CONFIG = {
    "batch_size": 100,
    "duplicate_strategy": "overwrite",  
}

# 检索配置
SEARCH_CONFIG = {
    "default_top_k": 5,
    "max_top_k": 100,
}