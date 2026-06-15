"""
全局配置文件
用于管理 LLM 后端选择、模型参数、路径等配置项。
队友模块（YOLO检测、CLIP向量库）的接口配置也在此统一管理。
"""

import os

# ============================================================
# LLM 配置
# ============================================================

# 后端选择: "ollama" | "openai" | "mock"
LLM_BACKEND = os.getenv("LLM_BACKEND", "mock")

# Ollama 配置
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # 支持 Llama3 / Qwen / ChatGLM3

# OpenAI 兼容 API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-api-key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 通用生成参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # 低温度减少幻觉
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

# ============================================================
# 向量检索配置
# ============================================================

VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")  # chroma | faiss | milvus
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))  # 相关性最低阈值

# ============================================================
# 队友模块接口配置
# ============================================================

# YOLO 检测器（队友提供）
# 接口: detect(video_path: str) -> List[DetectionResult]
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./models/yolo")

# CLIP 特征提取器（队友提供）
# 接口: extract_image_features(image_path: str) -> ndarray
# 接口: extract_text_features(text: str) -> ndarray
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "./models/clip")

# 向量数据库检索器（队友提供）
# 接口: search(query_vector: ndarray, top_k: int) -> List[SearchResult]
# 接口: search_by_text(text: str, top_k: int) -> List[SearchResult]

# ============================================================
# 路径配置
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Gradio 配置
# ============================================================

GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
GRADIO_TITLE = "YOLO-CLIP 多模态问答系统"
