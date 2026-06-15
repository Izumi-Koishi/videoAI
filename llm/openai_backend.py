"""
OpenAI 兼容 API 后端
支持 OpenAI 官方 API 及兼容服务（如本地 vLLM、Ollama OpenAI 模式等）。
"""

from typing import Optional
from .base import BaseLLM


class OpenAICompatibleBackend(BaseLLM):
    """OpenAI 兼容 API 后端"""

    def __init__(self, model_name: str = "gpt-4o-mini",
                 api_key: str = "",
                 base_url: str = "https://api.openai.com/v1",
                 temperature: float = 0.3, max_tokens: int = 1024,
                 top_p: float = 0.9):
        super().__init__(model_name, temperature, max_tokens, top_p)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "未提供 API Key，请在界面中填写或设置环境变量 OPENAI_API_KEY"
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError(
                    "请安装 openai 库: pip install openai"
                )
        return self._client

    def is_available(self) -> bool:
        """检查 API 是否可用"""
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 **kwargs) -> str:
        """
        调用 OpenAI 兼容 API 生成回答

        Args:
            prompt: 用户提示词
            system_prompt: 系统角色设定
        """
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", self.top_p),
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API 调用失败: {e}")
