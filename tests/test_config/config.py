"""
videoAI 集成系统 — 统一配置中心
=================================
集中管理所有模块的配置项，支持环境变量覆盖。
"""

import os
from pathlib import Path

# ============================================================
# 项目根目录
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
# 路径配置
# ============================================================
DATA_DIR = PROJECT_ROOT / "data" / "data"          # 数据准备组提供的测试数据
TESTS_DIR = PROJECT_ROOT / "tests"                  # 集成测试产出目录
OUTPUT_DIR = TESTS_DIR / "output"                   # 测试运行时输出
LOG_DIR = TESTS_DIR / "logs"                        # 日志目录
VIDEO_DIR = PROJECT_ROOT / "data"                   # 视频文件目录（兼容）

# 确保目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- 数据文件 ---
KNOWLEDGE_BASE_PATH = DATA_DIR / "campus_knowledge_base.json"
TEST_QA_PATH = DATA_DIR / "campus_test_qa.json"
TEST_VIDEO_PATH = DATA_DIR / "test_video.mp4"

# --- 向量数据库 ---
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(OUTPUT_DIR / "vector_db"))

# --- 检测结果 ---
DETECTION_OUTPUT_DIR = str(OUTPUT_DIR / "detection")

# ============================================================
# YOLO 检测配置
# ============================================================
YOLO_CONFIG = {
    "model_name": os.getenv("YOLO_MODEL_NAME", "yolov8n.pt"),
    "frame_interval": int(os.getenv("YOLO_FRAME_INTERVAL", "30")),
    "confidence_threshold": float(os.getenv("YOLO_CONFIDENCE", "0.25")),
    "iou_threshold": float(os.getenv("YOLO_IOU", "0.45")),
}

# ============================================================
# CLIP 特征提取配置
# ============================================================
CLIP_CONFIG = {
    "model_name": os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
    "model_type": os.getenv("CLIP_MODEL_TYPE", "auto"),
    "device": os.getenv("CLIP_DEVICE", "auto"),
    "batch_size": int(os.getenv("CLIP_BATCH_SIZE", "32")),
    "embedding_dim": int(os.getenv("CLIP_EMBEDDING_DIM", "512")),
    "local_files_only": os.getenv("CLIP_LOCAL_FILES_ONLY", "0") == "1",
}

# ============================================================
# 向量数据库配置
# ============================================================
VECTOR_DB_CONFIG = {
    "persist_directory": VECTOR_DB_PATH,
    "collection_name": os.getenv("VECTOR_COLLECTION", "campus_targets"),
    "feature_dim": int(os.getenv("VECTOR_FEATURE_DIM", "512")),
    "default_top_k": int(os.getenv("VECTOR_TOP_K", "5")),
}

# ============================================================
# LLM 配置
# ============================================================
LLM_CONFIG = {
    "backend": os.getenv("LLM_BACKEND", "openai"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "api_base": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "model_name": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    "ollama_url": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
}

# ============================================================
# Gradio Web 界面配置
# ============================================================
GRADIO_CONFIG = {
    "host": os.getenv("GRADIO_HOST", "127.0.0.1"),
    "port": int(os.getenv("GRADIO_PORT", "7860")),
    "share": os.getenv("GRADIO_SHARE", "false").lower() == "true",
    "title": "videoAI - 校园监控智能问答系统",
}

# ============================================================
# 集成测试配置
# ============================================================
TEST_CONFIG = {
    "test_qa_sample_size": int(os.getenv("TEST_QA_SAMPLE", "0")),  # 0=全部
    "similarity_threshold": float(os.getenv("TEST_SIMILARITY_THRESHOLD", "0.3")),
    "skip_llm_tests": os.getenv("SKIP_LLM_TESTS", "false").lower() == "true",
    "report_format": os.getenv("TEST_REPORT_FORMAT", "markdown"),  # markdown | json
}
