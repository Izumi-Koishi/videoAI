"""
Ollama 本地模型后端
支持 Llama 3、Qwen、ChatGLM3 等开源模型。
通过 Ollama 服务进行本地推理。
"""

from typing import Optional
from .base import BaseLLM


class OllamaBackend(BaseLLM):
    """Ollama 本地推理后端"""

    def __init__(self, model_name: str = "qwen2.5:7b",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.3, max_tokens: int = 1024,
                 top_p: float = 0.9):
        super().__init__(model_name, temperature, max_tokens, top_p)
        self.host = host
        self._client = None

    def _get_client(self):
        """延迟初始化 Ollama 客户端"""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise ImportError(
                    "请安装 ollama 库: pip install ollama"
                )
        return self._client

    def is_available(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            client = self._get_client()
            client.list()
            return True
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 **kwargs) -> str:
        """
        调用 Ollama 生成回答

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
            response = client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                    "top_p": kwargs.get("top_p", self.top_p),
                }
            )
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama 生成失败: {e}")
