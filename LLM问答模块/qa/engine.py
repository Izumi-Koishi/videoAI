"""
问答主逻辑引擎
实现完整的问答流程：问题理解 -> 检索 -> Prompt拼接 -> LLM生成 -> 后处理
"""

import time
import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field

from prompts.templates import PromptBuilder, RetrievedContext
from prompts.examples import FEW_SHOT_EXAMPLES
from llm.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """问答结果"""
    question: str
    answer: str
    contexts: List[RetrievedContext]
    elapsed_time: float
    model_info: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    confidence: Optional[str] = None


class QAEngine:
    """
    问答引擎
    封装完整的问答流程，支持自定义检索器和LLM后端
    """

    def __init__(self,
                 llm: BaseLLM,
                 retriever: Callable = None,
                 domain: str = "通用场景"):
        """
        Args:
            llm: LLM 后端实例
            retriever: 检索函数，签名为 (query_text: str, top_k: int) -> List[SearchResult]
            domain: 领域名称
        """
        self.llm = llm
        self.retriever = retriever
        self.domain = domain
        self.prompt_builder = PromptBuilder()

        # 可选的外部文本特征提取器（队友提供）
        self.text_encoder = None

    def set_retriever(self, retriever: Callable):
        """设置检索器（由队友模块注入）"""
        self.retriever = retriever

    def set_text_encoder(self, encoder: Callable):
        """设置文本特征提取器（由队友模块注入）"""
        self.text_encoder = encoder

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievedContext]:
        """
        执行检索

        流程：问题 -> (可选)CLIP文本特征 -> 向量数据库检索
        """
        if self.retriever is None:
            logger.warning("检索器未设置，返回空结果")
            return []

        try:
            search_results = self.retriever(question, top_k=top_k)

            # 过滤低相似度结果
            from config import SIMILARITY_THRESHOLD
            search_results = [s for s in search_results
                            if getattr(s, 'similarity', 1.0) >= SIMILARITY_THRESHOLD]

            # 转换为 Prompt 可用的上下文格式
            contexts = []
            for sr in search_results:
                contexts.append(RetrievedContext(
                    target_id=sr.target_id,
                    class_name=sr.class_name,
                    confidence=sr.confidence,
                    similarity=sr.similarity,
                    description=sr.description,
                    image_path=getattr(sr, 'image_path', ''),
                ))

            return contexts

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def generate_answer(self,
                        question: str,
                        contexts: List[RetrievedContext],
                        use_few_shot: bool = True) -> str:
        """
        生成回答

        Args:
            question: 用户问题
            contexts: 检索到的上下文
            use_few_shot: 是否使用Few-shot示例
        """
        # 1. 构建完整 Prompt
        few_shot = FEW_SHOT_EXAMPLES if use_few_shot else ""
        prompt = self.prompt_builder.build_full_prompt(
            question=question,
            contexts=contexts,
            domain=self.domain,
            few_shot_examples=few_shot,
            include_anti_hallucination=True,
            include_format_guide=True,
        )

        # 2. 构建系统提示词
        system_prompt = self.prompt_builder.get_system_prompt(self.domain)

        # 3. 调用 LLM 生成
        logger.info(f"正在生成回答: {question[:50]}...")
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        return answer

    def ask(self, question: str, top_k: int = 5,
            use_few_shot: bool = True) -> QAResult:
        """
        完整问答流程

        Args:
            question: 用户问题
            top_k: 检索数量
            use_few_shot: 是否使用Few-shot示例

        Returns:
            QAResult 包含回答、上下文、耗时等信息
        """
        start_time = time.time()

        # Step 1: 检索
        contexts = self.retrieve(question, top_k=top_k)

        # Step 2: LLM 生成
        answer = self.generate_answer(
            question=question,
            contexts=contexts,
            use_few_shot=use_few_shot,
        )

        elapsed = time.time() - start_time

        # Step 3: 后处理
        from .postprocess import AnswerPostProcessor
        retrieved_classes = [c.class_name for c in contexts]
        processed_answer, warnings = AnswerPostProcessor.postprocess(
            answer, retrieved_classes
        )
        confidence = AnswerPostProcessor.extract_confidence(processed_answer)

        return QAResult(
            question=question,
            answer=processed_answer,
            contexts=contexts,
            elapsed_time=elapsed,
            model_info=self.llm.get_model_info(),
            warnings=warnings,
            confidence=confidence,
        )
