import logging
from typing import List, Optional
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from config import CLIP_CONFIG

logger = logging.getLogger(__name__)

class CLIPEmbedder:
    """
    CLIP特征提取封装类，统一图像/文本特征提取逻辑
    
    支持中文CLIP和OpenAI CLIP，默认使用中文CLIP
    特征向量自动归一化处理
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化CLIP特征提取器
        
        Args:
            model_name: CLIP模型名称，默认为中文CLIP
        """
        self.model_name = model_name or CLIP_CONFIG["model_name"]
        self.feature_dim = CLIP_CONFIG["feature_dim"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading CLIP model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
        
        logger.info("CLIP model loaded successfully")
    
    def _load_model(self):
        """加载CLIP模型，支持备选模型"""
        try_models = [self.model_name]
        
        # 添加备选模型
        if self.model_name == "OFA-Sys/chinese-clip-vit-base-patch16":
            try_models.append("openai/clip-vit-base-patch16")
        
        last_error = None
        for model_name in try_models:
            try:
                logger.info(f"Trying to load model: {model_name}")
                self.processor = CLIPProcessor.from_pretrained(
                    model_name,
                    cache_dir=CLIP_CONFIG["model_cache_dir"]
                )
                self.model = CLIPModel.from_pretrained(
                    model_name,
                    cache_dir=CLIP_CONFIG["model_cache_dir"]
                ).to(self.device)
                self.model.eval()
                self.model_name = model_name
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load model {model_name}: {str(e)}")
        
        raise RuntimeError(f"无法加载任何CLIP模型: {str(last_error)}")
    
    def extract_text_features(self, texts: List[str]) -> List[List[float]]:
        """
        提取文本特征向量
        
        Args:
            texts: 文本列表
            
        Returns:
            归一化后的特征向量列表，每个向量维度为feature_dim
            
        Raises:
            ValueError: 输入文本为空
        """
        if not texts or len(texts) == 0:
            raise ValueError("输入文本列表不能为空")
        
        logger.debug(f"Extracting features for {len(texts)} texts")
        
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            outputs = self.model.get_text_features(**inputs)
            
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            
        return outputs.cpu().tolist()
    
    def extract_image_features(self, image_paths: List[str]) -> List[List[float]]:
        """
        提取图像特征向量
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            归一化后的特征向量列表，每个向量维度为feature_dim
            
        Raises:
            ValueError: 输入路径为空或文件不存在
        """
        if not image_paths or len(image_paths) == 0:
            raise ValueError("输入图像路径列表不能为空")
        
        logger.debug(f"Extracting features for {len(image_paths)} images")
        
        images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to open image: {path}, error: {str(e)}")
                raise ValueError(f"无法打开图像文件: {path}")
        
        with torch.no_grad():
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model.get_image_features(**inputs)
            
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            
        return outputs.cpu().tolist()
    
    def get_feature_dim(self) -> int:
        """
        获取特征向量维度
        
        Returns:
            特征向量维度
        """
        return self.feature_dim