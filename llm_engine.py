"""
LLM 引擎模块 - 负责与大语言模型交互，生成回答
"""
import os
from typing import List, Dict, Optional


# 领域专属 Prompt 模板库
PROMPT_TEMPLATES = {
    "default": """你是一个专业的视频内容分析助手。根据以下视频检测结果和用户问题，提供准确、专业的回答。

## 视频检测结果
{context}

## 用户问题
{question}

## 回答要求
1. 基于检测结果中的信息进行回答，不要编造不存在的内容
2. 如果检测结果不足以回答问题，请诚实说明
3. 回答要简洁、准确、有条理
4. 如涉及检测到的目标，请指明其出现的时间位置

请给出你的回答:""",

    "object_count": """你是一个专业的视频内容分析助手，擅长目标统计。根据以下视频检测结果回答用户的目标统计问题。

## 视频检测结果
{context}

## 用户问题
{question}

## 回答要求
1. 仔细统计检测结果中各类目标的出现次数
2. 给出明确的数量和位置信息
3. 如果某个类别未检测到，请明确说明

请给出你的回答:""",

    "scene_desc": """你是一个专业的视频内容分析助手，擅长场景描述。根据以下视频检测结果，描述视频中的场景内容。

## 视频检测结果
{context}

## 用户问题
{question}

## 回答要求
1. 描述视频中出现的场景和目标
2. 按时间顺序组织描述
3. 注意目标之间的空间关系和交互

请给出你的回答:""",

    "technical": """你是一个专业的计算机视觉分析助手。根据以下 YOLO 目标检测结果，提供技术性的分析回答。

## 检测结果详情
{context}

## 用户问题
{question}

## 回答要求
1. 使用专业技术术语
2. 分析检测的置信度和准确度
3. 讨论可能存在的漏检或误检
4. 如有必要，建议改进方案

请给出你的回答:"""
}


class LLMEngine:
    """大语言模型引擎，支持多种后端"""

    def __init__(
        self,
        backend: str = "openai",
        api_key: str = "",
        api_base: str = "https://api.openai.com/v1",
        model_name: str = "gpt-3.5-turbo",
        ollama_model: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434"
    ):
        """
        初始化 LLM 引擎
        Args:
            backend: 后端类型 "openai"(兼容API) 或 "ollama"(本地模型)
            api_key: OpenAI 兼容 API 的密钥
            api_base: API 基础地址
            model_name: 模型名称
            ollama_model: Ollama 模型名称
            ollama_url: Ollama 服务地址
        """
        self.backend = backend
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.client = None

        self._init_client()

    def _init_client(self):
        """初始化 LLM 客户端"""
        if self.backend == "openai":
            effective_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
            if not effective_key:
                print("[LLM] 未提供 API Key，请在界面中填写或设置环境变量 OPENAI_API_KEY")
                self.client = None
                return
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=effective_key,
                    base_url=self.api_base
                )
                print(f"[LLM] OpenAI 兼容客户端初始化成功 (模型: {self.model_name})")
            except Exception as e:
                print(f"[LLM] OpenAI 客户端初始化失败: {e}")
                self.client = None

    def is_ready(self) -> bool:
        """检查 LLM 是否就绪"""
        if self.backend == "openai":
            return self.client is not None
        elif self.backend == "ollama":
            return True  # Ollama 按请求调用
        return False

    def _build_prompt(self, question: str, context: str, template_name: str = "default") -> str:
        """
        构建 LLM 提示词
        Args:
            question: 用户问题
            context: 检索到的上下文
            template_name: 使用的模板名称
        Returns:
            构建好的提示词
        """
        template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["default"])
        return template.format(context=context, question=question)

    def _select_template(self, question: str) -> str:
        """根据问题内容自动选择 Prompt 模板"""
        question_lower = question.lower()

        # 统计类问题
        count_keywords = ["多少", "几个", "数量", "计数", "统计", "how many", "count"]
        if any(kw in question_lower for kw in count_keywords):
            return "object_count"

        # 描述类问题
        desc_keywords = ["描述", "什么场景", "发生了什么", "什么情况", "describe", "what happened"]
        if any(kw in question_lower for kw in desc_keywords):
            return "scene_desc"

        # 技术类问题
        tech_keywords = ["置信度", "精度", "误检", "漏检", "confidence", "accuracy", "precision"]
        if any(kw in question_lower for kw in tech_keywords):
            return "technical"

        return "default"

    def generate(
        self,
        question: str,
        context: str,
        template_name: str = "auto",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        生成回答
        Args:
            question: 用户问题
            context: 检索到的上下文
            template_name: Prompt 模板名称，"auto" 为自动选择
            max_tokens: 最大生成 token 数
            temperature: 生成温度
        Returns:
            LLM 生成的回答
        """
        if template_name == "auto":
            template_name = self._select_template(question)

        prompt = self._build_prompt(question, context, template_name)

        if self.backend == "openai":
            return self._generate_openai(prompt, max_tokens, temperature)
        elif self.backend == "ollama":
            return self._generate_ollama(prompt, max_tokens, temperature)
        else:
            return f"[错误] 不支持的后端类型: {self.backend}"

    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """通过 OpenAI 兼容 API 生成回答"""
        if not self.client:
            return "[错误] OpenAI 客户端未初始化，请检查 API Key 和 Base URL 配置"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的视频内容分析助手，基于YOLO目标检测结果回答用户问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[错误] LLM 调用失败: {str(e)}\n请检查 API Key、Base URL 和模型名称是否正确。"

    def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """通过 Ollama 生成回答"""
        try:
            import requests
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "[错误] Ollama 返回空响应")
            else:
                return f"[错误] Ollama 返回状态码 {response.status_code}: {response.text}"
        except ImportError:
            return "[错误] 需要安装 requests 库: pip install requests"
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "refused" in error_msg:
                return f"[错误] 无法连接到 Ollama 服务 ({self.ollama_url})，请确保 Ollama 已启动并运行。"
            return f"[错误] Ollama 调用失败: {error_msg}"

    def get_available_templates(self) -> List[str]:
        """获取可用的 Prompt 模板列表"""
        return list(PROMPT_TEMPLATES.keys())

    def get_template_description(self, template_name: str) -> str:
        """获取模板描述"""
        descriptions = {
            "default": "通用问答模板 - 适用于一般性问题",
            "object_count": "目标统计模板 - 适用于计数、统计类问题",
            "scene_desc": "场景描述模板 - 适用于场景描述类问题",
            "technical": "技术分析模板 - 适用于技术性分析问题"
        }
        return descriptions.get(template_name, "未知模板")
