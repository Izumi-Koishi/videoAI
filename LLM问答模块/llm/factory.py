"""
LLM 工厂函数
根据配置选择合适的 LLM 后端，并支持 Mock 模式（无 LLM 时演示用）。
"""

import sys
from .base import BaseLLM


# ============================================================
# Mock LLM（无真实模型时演示用）
# ============================================================

class MockLLM(BaseLLM):
    """Mock LLM 后端，用于演示和测试"""

    def __init__(self, model_name: str = "mock", **kwargs):
        super().__init__(model_name, **kwargs)

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """返回模拟回答"""
        question = ""
        for line in prompt.split("\n"):
            if line.strip().startswith("【用户问题】"):
                question = line.replace("【用户问题】", "").strip()
                break

        n_targets = prompt.count("【检测目标")

        return f"""【回答】
基于视频分析结果，我为您提供以下回答：

问题：{question}

## 分析结果
在视频中检测到 {n_targets} 个相关目标物体。根据检索到的视觉特征和语义信息，
这些目标在当前场景中具有以下特征：

## 详细说明
1. **目标识别**：系统通过 YOLO 检测器在视频帧中识别出目标物体，并经过 CLIP
   模型进行语义对齐验证。
2. **语义匹配**：相关目标与您的查询具有较高的语义相关性（余弦相似度 > 0.7），
   表明检索结果可信。
3. **场景分析**：当前场景中的目标物体分布在视频的不同帧中，各目标与查询的
   上下文匹配度良好。

## 注意事项
⚠️ 当前运行在 **Mock 模式**下，以上回答基于模板生成，非真实模型推理结果。
如需真实回答，请配置 LLM 后端（Ollama 或 OpenAI API），详见 config.py。

---
📊 检索置信度: 0.85 (Mock) | 引用来源: 向量数据库 Top-{n_targets} 检索结果
"""


# ============================================================
# 工厂函数
# ============================================================

def get_available_backends() -> list:
    """检测当前环境可用的后端"""
    available = ["mock"]  # Mock 始终可用

    # 检测 Ollama
    try:
        from .ollama_backend import OllamaBackend
        backend = OllamaBackend()
        if backend.is_available():
            available.append("ollama")
    except Exception:
        pass

    # 检测 OpenAI
    try:
        from .openai_backend import OpenAICompatibleBackend
        backend = OpenAICompatibleBackend()
        if backend.is_available():
            available.append("openai")
    except Exception:
        pass

    return available


def create_llm(backend: str = None, **kwargs) -> BaseLLM:
    """
    创建 LLM 实例

    Args:
        backend: "ollama" | "openai" | "mock"
        **kwargs: 覆盖默认配置参数

    Returns:
        BaseLLM 实例
    """
    if backend is None:
        import config
        backend = config.LLM_BACKEND

    if backend == "ollama":
        from .ollama_backend import OllamaBackend
        import config
        return OllamaBackend(
            model_name=kwargs.get("model_name", config.OLLAMA_MODEL),
            host=kwargs.get("host", config.OLLAMA_HOST),
            temperature=kwargs.get("temperature", config.LLM_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", config.LLM_MAX_TOKENS),
            top_p=kwargs.get("top_p", config.LLM_TOP_P),
        )

    elif backend == "openai":
        from .openai_backend import OpenAICompatibleBackend
        import config
        return OpenAICompatibleBackend(
            model_name=kwargs.get("model_name", config.OPENAI_MODEL),
            api_key=kwargs.get("api_key", config.OPENAI_API_KEY),
            base_url=kwargs.get("base_url", config.OPENAI_BASE_URL),
            temperature=kwargs.get("temperature", config.LLM_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", config.LLM_MAX_TOKENS),
            top_p=kwargs.get("top_p", config.LLM_TOP_P),
        )

    elif backend == "mock":
        return MockLLM(
            model_name="Mock LLM (演示模式)",
            temperature=0.3,
            max_tokens=1024,
        )

    else:
        print(f"⚠️  未知后端 '{backend}'，回退到 Mock 模式")
        return MockLLM()
