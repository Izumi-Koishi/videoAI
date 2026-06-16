"""
Gradio 完整交互界面
实现视频上传、检测结果可视化、问答交互、匹配图片展示等全部功能区。
"""

import os
import sys
import time
import traceback
from typing import List, Optional, Tuple

import gradio as gr

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GRADIO_TITLE, RETRIEVAL_TOP_K
from llm.factory import create_llm, get_available_backends
from qa.engine import QAEngine
from .mock_modules import MockYOLODetector, MockCLIPRetriever, DetectionResult


# ================================================================
# 全局状态
# ================================================================

class AppState:
    """应用全局状态"""
    def __init__(self):
        self.detector = MockYOLODetector()
        self.retriever = MockCLIPRetriever()
        self.llm = create_llm()
        self.qa_engine = QAEngine(
            llm=self.llm,
            retriever=self.retriever.search,
            domain="校园场景",
        )
        self.detections: List[DetectionResult] = []
        self.video_processed = False

    def reset_detections(self):
        self.detections = []
        self.video_processed = False


_state = AppState()


# ================================================================
# 回调函数
# ================================================================

def process_video(video_path: Optional[str],
                  progress=gr.Progress()) -> Tuple:
    """
    处理上传的视频：运行YOLO检测 + 建立向量索引

    Returns:
        (状态信息, 检测结果Gallery)
    """
    if video_path is None:
        return "### ⚠️ 请先上传视频文件", []

    try:
        progress(0.1, desc="初始化检测器...")
        _state.reset_detections()

        progress(0.3, desc="YOLO检测中...")
        detections = _state.detector.detect_video(video_path,
                                                  sample_interval=30)
        _state.detections = detections

        progress(0.7, desc="建立CLIP向量索引...")
        _state.retriever.set_detections(detections)

        progress(0.9, desc="生成可视化结果...")

        # 构建 Gallery 数据
        gallery_items = []
        for det in detections[:16]:
            if os.path.exists(det.crop_path):
                gallery_items.append((det.crop_path,
                                      f"帧{det.frame_id}: {det.class_name} "
                                      f"({det.confidence:.0%})"))
            else:
                gallery_items.append((None,
                                      f"帧{det.frame_id}: {det.class_name} "
                                      f"({det.confidence:.0%})"))

        # 统计信息
        class_counts = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        summary_lines = [
            "## ✅ 视频处理完成",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 检测目标总数 | {len(detections)} |",
            f"| 检测类别数 | {len(class_counts)} |",
            f"| 采样帧数 | {len(set(d.frame_id for d in detections))} |",
            "",
            "### 各类别统计",
        ]
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
            summary_lines.append(f"- {cls}: {cnt} 个")

        _state.video_processed = True
        progress(1.0, desc="处理完成!")

        return "\n".join(summary_lines), gallery_items if gallery_items else []

    except Exception as e:
        error_msg = f"### ❌ 视频处理失败\n```\n{traceback.format_exc()}\n```"
        return error_msg, []


def ask_question(question: str, top_k: int = 5) -> Tuple[str, List, str, str]:
    """
    处理用户问答请求

    Returns:
        (回答, 匹配图片Gallery, 检索信息, 耗时)
    """
    if not question.strip():
        return "### ⚠️ 请输入问题", [], "", ""

    try:
        result = _state.qa_engine.ask(question, top_k=top_k)

        # 匹配图片
        matched_images = []
        for ctx in result.contexts:
            img_path = ctx.image_path
            if os.path.exists(img_path):
                matched_images.append((img_path,
                                       f"#{ctx.target_id} {ctx.class_name} "
                                       f"(相似度:{ctx.similarity:.2f})"))

        # 检索信息
        retrieval_lines = ["### 📊 检索结果"]
        for ctx in result.contexts:
            retrieval_lines.append(
                f"- **#{ctx.target_id}** {ctx.class_name} "
                f"(相似度: {ctx.similarity:.4f}, "
                f"置信度: {ctx.confidence:.0%})"
            )
            retrieval_lines.append(f"  - {ctx.description}")

        # 耗时和模型信息
        time_info = (
            f"⏱ 耗时: {result.elapsed_time:.2f}s"
        )
        if result.confidence:
            time_info += f" | 置信度: {result.confidence}"

        model_info = (
            f"🤖 {result.model_info.get('model', 'N/A')} "
            f"({result.model_info.get('backend', 'N/A')})"
        )

        # 警告
        warning_text = ""
        if result.warnings:
            warning_lines = ["", "### ⚠️ 质量提示"]
            for w in result.warnings:
                warning_lines.append(f"- {w}")
            warning_text = "\n".join(warning_lines)

        return (result.answer, matched_images,
                "\n".join(retrieval_lines),
                f"{time_info}\n{model_info}{warning_text}")

    except Exception as e:
        error_msg = f"### ❌ 问答处理失败\n```\n{traceback.format_exc()}\n```"
        return error_msg, [], "", ""


def update_llm_backend(backend_choice: str) -> str:
    """切换 LLM 后端"""
    try:
        _state.llm = create_llm(backend=backend_choice)
        _state.qa_engine.llm = _state.llm
        info = _state.llm.get_model_info()
        return (f"✅ 已切换到 **{backend_choice}** 后端 "
                f"(模型: {info['model']})")
    except Exception as e:
        return f"❌ 切换失败: {str(e)}"


def clear_all():
    """重置所有状态"""
    _state.reset_detections()
    return ("", [], [], "*等待提问...*", "*检索详情将在此显示*",
            "*系统信息*")


# ================================================================
# 界面构建
# ================================================================

CSS = """
.gradio-container { max-width: 1400px !important; }
.main-header { text-align: center; margin-bottom: 20px; }
.answer-box { min-height: 300px; border: 1px solid #e5e7eb; padding: 15px;
              border-radius: 8px; background: #fafafa; }
.hint-text { color: #6b7280; font-size: 0.9em; }
"""


def create_app() -> gr.Blocks:
    """创建 Gradio 应用"""

    with gr.Blocks(title=GRADIO_TITLE) as app:

        # ============================================
        # 标题区
        # ============================================
        gr.HTML(f"""
        <div class="main-header">
            <h1>🎯 {GRADIO_TITLE}</h1>
            <p>YOLO目标检测 + CLIP语义对齐 + 向量检索 + LLM智能问答</p>
            <p class="hint-text">
            上传视频 → 系统自动检测 → 输入问题 → 获取智能分析回答</p>
            <hr>
        </div>
        """)

        # ============================================
        # 配置区
        # ============================================
        with gr.Accordion("⚙️ 系统配置", open=False):
            with gr.Row():
                backend_choice = gr.Dropdown(
                    choices=get_available_backends(),
                    value="mock",
                    label="LLM 后端",
                    info="选择问答使用的语言模型后端",
                    scale=2,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10,
                    value=RETRIEVAL_TOP_K, step=1,
                    label="检索数量 (Top-K)",
                    info="每次检索返回的最相关目标数",
                    scale=1,
                )
                backend_status = gr.Textbox(
                    label="后端状态",
                    value="Mock 演示模式",
                    interactive=False,
                    scale=2,
                )

            backend_choice.change(
                update_llm_backend,
                inputs=[backend_choice],
                outputs=[backend_status],
            )

        # ============================================
        # 主区域：左右分栏
        # ============================================
        with gr.Row(equal_height=False):
            # ---- 左侧：视频 + 检测结果 ----
            with gr.Column(scale=1):
                gr.Markdown("### 📹 视频输入")

                video_input = gr.Video(
                    label="上传视频文件",
                    sources=["upload"],
                    height=240,
                )

                with gr.Row():
                    process_btn = gr.Button(
                        "🔍 开始检测",
                        variant="primary",
                        size="lg",
                    )
                    clear_btn = gr.Button(
                        "🗑 清空",
                        variant="secondary",
                        size="lg",
                    )

                processing_status = gr.Markdown("")

                gr.Markdown("### 🖼 检测结果预览")
                detection_gallery = gr.Gallery(
                    label="YOLO检测到的目标物体",
                    columns=3,
                    rows=4,
                    height=400,
                    object_fit="contain",
                )

            # ---- 右侧：问答区域 ----
            with gr.Column(scale=1):
                gr.Markdown("### 💬 智能问答")

                question_input = gr.Textbox(
                    label="输入您的问题",
                    placeholder="例如：视频中检测到了哪些车辆？"
                                "这些目标之间有什么关系？"
                                "描述一下场景中的人物活动...",
                    lines=3,
                )

                ask_btn = gr.Button(
                    "🤖 提交问题",
                    variant="primary",
                    size="lg",
                )

                answer_output = gr.Markdown(
                    value="*等待提问...*",
                    elem_classes="answer-box",
                )

                gr.Markdown("### 📸 匹配目标图像")
                matched_gallery = gr.Gallery(
                    label="向量检索匹配的目标",
                    columns=3,
                    rows=2,
                    height=250,
                    object_fit="contain",
                )

        # ============================================
        # 底部：检索详情 + 系统信息
        # ============================================
        with gr.Row():
            with gr.Column(scale=1):
                retrieval_detail = gr.Markdown(
                    "*检索详情将在此显示*",
                )
            with gr.Column(scale=1):
                sys_info = gr.Markdown(
                    "*系统信息*",
                )

        # ============================================
        # 使用说明
        # ============================================
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 操作步骤
            1. **上传视频** — 点击左侧上传区域选择视频文件
            2. **开始检测** — 点击"开始检测"，系统运行 YOLO 检测和 CLIP 索引
            3. **输入问题** — 在右侧问题框输入您想了解的内容
            4. **获取回答** — 点击"提交问题"，LLM 将基于检测结果回答

            ### 当前运行模式
            - **Mock 演示模式**：使用模拟检测和检索结果，无需安装额外依赖
            - 如需真实模型推理，请安装 [Ollama](https://ollama.com) 并设置
              环境变量 `LLM_BACKEND=ollama`
            - 队友模块（YOLO检测 / CLIP向量库）就绪后，
              将 `mock_modules` 替换为真实实现即可

            ### 支持的领域场景
            校园场景 · 交通场景 · 超市商品 · 实验室设备
            """)

        # ============================================
        # 事件绑定
        # ============================================

        process_btn.click(
            fn=process_video,
            inputs=[video_input],
            outputs=[processing_status, detection_gallery],
        )

        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, matched_gallery,
                     retrieval_detail, sys_info],
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[question_input, detection_gallery,
                     matched_gallery, answer_output,
                     retrieval_detail, sys_info],
        )

        # 回车提交
        question_input.submit(
            fn=ask_question,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, matched_gallery,
                     retrieval_detail, sys_info],
        )

    return app
