"""
LLM 抽象基类
定义统一的 LLM 调用接口，所有后端必须实现此接口。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLM(ABC):
    """LLM 统一抽象基类"""

    def __init__(self, model_name: str, temperature: float = 0.3,
                 max_tokens: int = 1024, top_p: float = 0.9):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 **kwargs) -> str:
        """
        生成回答

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（角色设定）
            **kwargs: 其他参数

        Returns:
            模型生成的文本
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "backend": self.__class__.__name__,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
