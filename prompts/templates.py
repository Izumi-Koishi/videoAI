"""
Prompt 模板库
提供领域特定的 Prompt 构建功能，包括系统角色、上下文注入、反幻觉指令等。
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RetrievedContext:
    """检索到的上下文信息"""
    target_id: int
    class_name: str
    confidence: float
    similarity: float
    description: str
    image_path: str = ""


class PromptBuilder:
    """
    Prompt 构建器
    负责将检索结果和用户问题组装成结构化 Prompt
    """

    # ================================================================
    # 系统角色定义
    # ================================================================

    SYSTEM_ROLE = """你是一个专业的领域特定多模态问答系统助手。
你的任务是基于视频分析系统提供的检测和检索结果，回答用户关于视频内容的问题。

## 你的能力
- 基于YOLO目标检测结果分析视频中的物体
- 利用CLIP多模态语义对齐理解图像与文本的关系
- 结合向量数据库检索的上下文信息提供准确回答

## 行为准则
1. **基于证据回答**：只根据提供的检索上下文回答问题，不要编造信息
2. **明确不确定性**：如果检索结果不足以回答问题，请明确说明
3. **引用来源**：回答时引用具体的检测目标ID和相似度
4. **结构化输出**：使用清晰的结构组织回答（如分点、标题等）
5. **领域专业性**：使用准确的专业术语描述检测到的物体"""

    # ================================================================
    # 上下文注入模板
    # ================================================================

    CONTEXT_HEADER = """
## 视频检测结果摘要
以下是YOLO检测器从视频中识别到的目标物体，经过CLIP语义对齐和向量数据库检索：

"""

    CONTEXT_ITEM_TEMPLATE = """### 【检测目标 {target_id}】
- **类别**: {class_name}
- **YOLO置信度**: {confidence:.2%}
- **语义相似度**: {similarity:.4f}
- **描述**: {description}
"""

    # ================================================================
    # 用户问题模板
    # ================================================================

    QUESTION_TEMPLATE = """
## 用户问题
【用户问题】{question}

请基于以上检测和检索结果，回答用户的问题。
"""

    # ================================================================
    # 反幻觉指令
    # ================================================================

    ANTI_HALLUCINATION = """
## 重要提醒
- 你的回答**必须严格基于**上述检测结果和检索上下文
- 如果某个信息在上下文**没有出现**，请说"根据现有检测结果无法确定"
- 不要推测检测结果之外的信息
- 如果相似度较低（< 0.5），请提醒用户结果可能不可靠
- 回答末尾请附上**置信度评估**：高/中/低，以及判断依据
"""

    # ================================================================
    # 输出格式指令
    # ================================================================

    OUTPUT_FORMAT = """
## 回答格式要求
请按以下结构组织你的回答：

### 1. 直接回答
用1-2句话直接回答用户的核心问题

### 2. 详细分析
结合检测到的目标，展开详细分析

### 3. 置信度评估
- **置信度**: [高/中/低]
- **依据**: 简要说明判断依据
- **引用目标**: 列出引用的检测目标ID及相似度
"""

    # ================================================================
    # 构建方法
    # ================================================================

    def build_system_prompt(self, domain: str = "通用场景") -> str:
        """构建系统提示词"""
        return self.SYSTEM_ROLE.replace("领域特定", f"{domain}领域")

    def build_context_prompt(self, contexts: List[RetrievedContext]) -> str:
        """构建上下文注入部分"""
        if not contexts:
            return "\n## 视频检测结果\n本次未检索到相关目标物体。\n"

        parts = [self.CONTEXT_HEADER]
        for ctx in contexts:
            parts.append(self.CONTEXT_ITEM_TEMPLATE.format(
                target_id=ctx.target_id,
                class_name=ctx.class_name,
                confidence=ctx.confidence,
                similarity=ctx.similarity,
                description=ctx.description,
            ))
        return "\n".join(parts)

    def build_question_prompt(self, question: str) -> str:
        """构建用户问题部分"""
        return self.QUESTION_TEMPLATE.format(question=question)

    def build_full_prompt(self,
                          question: str,
                          contexts: List[RetrievedContext],
                          domain: str = "通用场景",
                          few_shot_examples: str = "",
                          include_anti_hallucination: bool = True,
                          include_format_guide: bool = True) -> str:
        """
        构建完整的 Prompt

        Args:
            question: 用户问题
            contexts: 检索到的上下文列表
            domain: 领域名称
            few_shot_examples: Few-shot 示例文本
            include_anti_hallucination: 是否包含反幻觉指令
            include_format_guide: 是否包含输出格式指导

        Returns:
            完整的 Prompt 字符串
        """
        parts = []

        # 1. 上下文
        parts.append(self.build_context_prompt(contexts))

        # 2. Few-shot 示例
        if few_shot_examples:
            parts.append(few_shot_examples)

        # 3. 用户问题
        parts.append(self.build_question_prompt(question))

        # 4. 反幻觉指令
        if include_anti_hallucination:
            parts.append(self.ANTI_HALLUCINATION)

        # 5. 输出格式
        if include_format_guide:
            parts.append(self.OUTPUT_FORMAT)

        return "\n".join(parts)

    def get_system_prompt(self, domain: str = "通用场景") -> str:
        """获取系统提示词（用于 messages API）"""
        return self.build_system_prompt(domain)
