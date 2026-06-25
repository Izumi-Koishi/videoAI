"""
回答后处理模块
对 LLM 生成的回答进行质量检查、幻觉检测和格式修正。
"""

import re
from typing import List, Optional, Tuple


class AnswerPostProcessor:
    """回答后处理器：检测幻觉、修正格式、评估质量"""

    # 幻觉关键词模式
    HALLUCINATION_PATTERNS = [
        r"根据我的了解",
        r"一般来说",
        r"通常情况下",
        r"可能是",
        r"也许是",
        r"据我所知",
        r"众所周知",
    ]

    # 不确定性表述（正面）
    UNCERTAINTY_PATTERNS = [
        r"根据现有检测结果无法确定",
        r"检测结果不足以",
        r"建议.*重新检测",
        r"信息不足",
    ]

    @classmethod
    def detect_potential_hallucination(cls, answer: str,
                                       retrieved_classes: List[str]) -> List[str]:
        """
        检测回答中可能存在的幻觉

        Args:
            answer: LLM 生成的回答
            retrieved_classes: 检索结果中出现的类别名称

        Returns:
            潜在幻觉警告列表
        """
        warnings = []

        # 检查1：是否引用了不在检索结果中的类别
        for match in re.finditer(r"检测到[的]*[一\d]+[个台辆只]+\s*(\w+)", answer):
            mentioned_class = match.group(1)
            if mentioned_class not in retrieved_classes and \
               any(c not in mentioned_class for c in retrieved_classes):
                warnings.append(
                    f"回答中提到了未在检索结果中出现的类别: '{mentioned_class}'"
                )

        # 检查2：是否存在模糊推断
        for pattern in cls.HALLUCINATION_PATTERNS:
            if re.search(pattern, answer):
                warnings.append(
                    f"检测到模糊推断表述: '{pattern}'，可能增加幻觉风险"
                )

        # 检查3：置信度标识完整性
        if "置信度" not in answer:
            warnings.append("回答中缺少置信度评估")

        return warnings

    @classmethod
    def extract_confidence(cls, answer: str) -> Optional[str]:
        """从回答中提取置信度标识"""
        match = re.search(r"置信度\**[:：]\s*(高|中|低)", answer)
        if match:
            return match.group(1)
        return None

    @classmethod
    def extract_cited_targets(cls, answer: str) -> List[str]:
        """提取回答中引用的目标ID"""
        targets = re.findall(r"#(\d+)", answer)
        return targets

    @classmethod
    def append_quality_note(cls, answer: str,
                            warnings: List[str]) -> str:
        """在回答末尾附加质量备注"""
        if not warnings:
            return answer

        note = "\n\n---\n### 质量提示\n"
        for w in warnings:
            note += f"- {w}\n"
        return answer + note

    @classmethod
    def postprocess(cls, answer: str,
                    retrieved_classes: List[str]) -> Tuple[str, List[str]]:
        """
        完整的后处理流程

        Args:
            answer: LLM 原始回答
            retrieved_classes: 检索到的类别名列表

        Returns:
            (处理后回答, 警告列表)
        """
        # 1. 幻觉检测
        warnings = cls.detect_potential_hallucination(answer, retrieved_classes)

        # 2. 附加质量备注
        processed = cls.append_quality_note(answer, warnings)

        return processed, warnings
