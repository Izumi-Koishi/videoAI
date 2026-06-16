"""
视频AI模型 - Gradio 交互界面
完整的端到端交互 Demo：视频上传 → YOLO检测 → 向量检索 → LLM问答
"""
import os
import uuid
from pathlib import Path

import gradio as gr

from yolo_detector import YOLODetector
from vector_store import VectorStore
from llm_engine import LLMEngine, PROMPT_TEMPLATES

# ============================================================
# 全局状态
# ============================================================
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_BASE, exist_ok=True)

# 全局组件（延迟初始化）
detector = None
vector_store = None


def get_detector() -> YOLODetector:
    """获取/初始化 YOLO 检测器"""
    global detector
    if detector is None:
        detector = YOLODetector(model_name="yolov8n.pt")
    return detector


def get_vector_store() -> VectorStore:
    """获取/初始化向量数据库"""
    global vector_store
    if vector_store is None:
        db_dir = os.path.join(OUTPUT_BASE, "vector_db")
        vector_store = VectorStore(persist_dir=db_dir)
    return vector_store


def create_llm_engine(backend, api_key, api_base, model_name, ollama_model, ollama_url) -> LLMEngine:
    """创建 LLM 引擎"""
    return LLMEngine(
        backend=backend,
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        ollama_model=ollama_model,
        ollama_url=ollama_url
    )


# ============================================================
# 核心业务逻辑
# ============================================================

def process_video(video_path, frame_interval, conf_threshold, progress=gr.Progress()):
    """
    处理上传的视频：YOLO 检测 + 向量存储
    """
    if video_path is None:
        gr.Warning("请先上传视频文件！")
        return None, "❌ 未上传视频", "", None

    progress(0, desc="🔄 初始化检测模型...")
    det = get_detector()
    if not det.is_ready():
        return None, "❌ YOLO 模型加载失败，请检查 ultralytics 是否安装", "", None

    vs = get_vector_store()

    # 生成视频 ID 和输出目录
    video_id = str(uuid.uuid4())[:8]
    video_name = Path(video_path).stem
    output_dir = os.path.join(OUTPUT_BASE, video_id, "detection")
    os.makedirs(output_dir, exist_ok=True)

    progress(0.1, desc="🔄 正在处理视频帧...")

    try:
        detection_results, saved_images, detection_video_path = det.process_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=int(frame_interval),
            conf=float(conf_threshold)
        )
    except Exception as e:
        return None, f"❌ 视频处理失败: {str(e)}", "", None

    progress(0.7, desc="🔄 生成检测摘要...")

    detection_summary = det.generate_detection_summary(detection_results)

    progress(0.8, desc="🔄 存入向量数据库...")

    # 存入向量数据库
    if vs.is_ready() and detection_results:
        vs.add_detection_results(video_id, detection_results, detection_summary)

    progress(1.0, desc="✅ 处理完成！")

    # 生成状态信息
    total_objects = sum(len(r.get("labels", [])) for r in detection_results)
    unique_labels = set()
    for r in detection_results:
        unique_labels.update(r.get("labels", []))

    status_msg = (
        f"✅ 视频处理完成！\n"
        f"📹 视频ID: {video_id}\n"
        f"🎞️ 检测帧数: {len(detection_results)}\n"
        f"🎯 检测目标总数: {total_objects}\n"
        f"📋 检测类别: {', '.join(unique_labels) if unique_labels else '无'}\n"
        f"💾 向量数据库: {'已存储' if vs.is_ready() else '未就绪'}"
    )

    return detection_video_path, status_msg, detection_summary, video_id


def ask_question(
    question,
    video_id,
    backend,
    api_key,
    api_base,
    model_name,
    ollama_model,
    ollama_url,
    template_name,
    temperature,
    top_k,
    progress=gr.Progress()
):
    """
    问答主逻辑：问题 → 向量检索 → Prompt拼接 → LLM生成回答
    """
    if not question or not question.strip():
        gr.Warning("请输入问题！")
        return "❌ 请输入问题", "", []

    
    if not video_id or not video_id.strip():
        gr.Warning("请先处理视频！")
        return "❌ 请先上传并处理视频", "", []

    progress(0.1, desc="🔄 正在检索相关内容...")

    vs = get_vector_store()
    if not vs.is_ready():
        return "❌ 向量数据库未就绪", "", []

    # 向量检索
    context_text, image_paths = vs.get_context_and_images(
        video_id=video_id.strip(),
        query=question,
        top_k=int(top_k)
    )

    if not context_text:
        return "❌ 未检索到相关内容，请确认视频已正确处理", "", []

    progress(0.4, desc="🔄 正在调用 LLM 生成回答...")

    # 创建 LLM 引擎
    engine = create_llm_engine(
        backend=backend,
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        ollama_model=ollama_model,
        ollama_url=ollama_url
    )

    if not engine.is_ready():
        if backend == "openai":
            return "❌ LLM 未就绪：未提供 API Key，请在界面顶部“模型配置”中填写 API Key，或设置环境变量 OPENAI_API_KEY", context_text, image_paths[:6]
        else:
            return "❌ LLM 引擎未就绪，请检查配置", context_text, image_paths[:6]

    # LLM 生成回答
    answer = engine.generate(
        question=question,
        context=context_text,
        template_name=template_name,
        temperature=float(temperature)
    )

    progress(1.0, desc="✅ 回答生成完成！")

    return answer, context_text, image_paths[:6]


# ============================================================
# Gradio 界面构建
# ============================================================

def build_ui():
    """构建完整的 Gradio 交互界面"""

    with gr.Blocks(title="视频AI模型 - 智能问答系统") as demo:

        gr.Markdown(
            """
            # 🎬 视频 AI 模型 - 智能问答系统
            ### 基于 YOLO 目标检测 + 向量检索 + 大语言模型的端到端视频理解系统
            ---
            """
        )

        # ==================== 顶部：模型配置 ====================
        with gr.Accordion("⚙️ 模型配置（展开修改）", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**LLM 后端设置**")
                    backend_radio = gr.Radio(
                        choices=["openai", "ollama"],
                        value="openai",
                        label="LLM 后端",
                        info="选择 OpenAI 兼容 API 或 Ollama 本地模型"
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**OpenAI 兼容 API 配置**")
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="sk-xxx 或留空使用环境变量",
                        type="password"
                    )
                    api_base_input = gr.Textbox(
                        label="API Base URL",
                        value="https://api.openai.com/v1",
                        placeholder="https://api.openai.com/v1"
                    )
                    model_name_input = gr.Textbox(
                        label="模型名称",
                        value="gpt-3.5-turbo",
                        placeholder="gpt-3.5-turbo / qwen-plus / etc."
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Ollama 本地模型配置**")
                    ollama_model_input = gr.Textbox(
                        label="Ollama 模型名称",
                        value="qwen2.5:7b",
                        placeholder="qwen2.5:7b / llama3 / etc."
                    )
                    ollama_url_input = gr.Textbox(
                        label="Ollama 服务地址",
                        value="http://localhost:11434",
                        placeholder="http://localhost:11434"
                    )

        # ==================== 主体区域 ====================
        with gr.Row():
            # ---- 左侧：视频输入与检测 ----
            with gr.Column(scale=1):
                gr.Markdown("## 📹 视频输入与检测")

                # 视频上传 / 摄像头采集
                video_input = gr.Video(
                    label="上传视频 / 摄像头采集",
                    sources=["upload", "webcam"],
                    format="mp4"
                )

                # 检测参数
                with gr.Row():
                    frame_interval = gr.Slider(
                        minimum=5, maximum=120, value=30, step=5,
                        label="检测帧间隔",
                        info="每隔多少帧进行一次检测"
                    )
                    conf_threshold = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="置信度阈值",
                        info="YOLO 检测置信度下限"
                    )

                # 处理按钮
                process_btn = gr.Button("🚀 开始检测处理", variant="primary", size="lg")

                # 处理状态
                process_status = gr.Textbox(
                    label="处理状态",
                    lines=6,
                    interactive=False
                )

                # 检测摘要
                detection_summary = gr.Textbox(
                    label="检测摘要",
                    lines=5,
                    interactive=False
                )

            # ---- 右侧：检测结果可视化 ----
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 YOLO 检测结果可视化")

                # 检测后视频
                detection_video = gr.Video(
                    label="检测结果视频（含标注框）"
                )

                # 视频 ID（隐藏状态）
                video_id_state = gr.Textbox(
                    label="视频 ID",
                    interactive=False,
                    visible=True
                )

        gr.Markdown("---")

        # ==================== 问答区域 ====================
        with gr.Row():
            # ---- 左侧：问题输入与配置 ----
            with gr.Column(scale=1):
                gr.Markdown("## 💬 智能问答")

                question_input = gr.Textbox(
                    label="输入你的问题",
                    placeholder="例如：视频中出现了哪些物体？有多少辆车？",
                    lines=3
                )

                # 问答参数
                with gr.Row():
                    template_dropdown = gr.Dropdown(
                        choices=["auto"] + list(PROMPT_TEMPLATES.keys()),
                        value="auto",
                        label="Prompt 模板",
                        info="auto 为自动选择"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0, maximum=1.5, value=0.7, step=0.1,
                        label="生成温度",
                        info="越高越随机"
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="检索 Top-K",
                        info="检索最相关的K条结果"
                    )

                # 快捷问题按钮
                with gr.Row():
                    q1_btn = gr.Button("出现了什么物体？", size="sm")
                    q2_btn = gr.Button("有多少个目标？", size="sm")
                    q3_btn = gr.Button("描述视频场景", size="sm")
                    q4_btn = gr.Button("检测的置信度如何？", size="sm")

                ask_btn = gr.Button("🤔 提问", variant="primary", size="lg")

            # ---- 右侧：回答输出与图片展示 ----
            with gr.Column(scale=1):
                gr.Markdown("## 📝 LLM 回答输出")

                answer_output = gr.Textbox(
                    label="LLM 回答",
                    lines=8,
                    interactive=False
                )

                # 检索上下文（可折叠）
                with gr.Accordion("🔍 检索上下文（点击展开）", open=False):
                    context_output = gr.Textbox(
                        label="检索到的上下文",
                        lines=5,
                        interactive=False
                    )

                # 匹配目标图像展示区域
                gr.Markdown("## 🖼️ 匹配目标图像")
                matched_images = gr.Gallery(
                    label="匹配的目标图像",
                    columns=3,
                    rows=2,
                    height="auto"
                )

        # ==================== 底部说明 ====================
        gr.Markdown(
            """
            ---
            ### 📖 使用说明
            1. **上传视频**：支持上传视频文件或使用摄像头采集
            2. **检测处理**：点击"开始检测处理"，系统将自动进行 YOLO 目标检测并将结果存入向量数据库
            3. **智能问答**：在问题输入框中输入问题，系统将检索相关内容并调用 LLM 生成回答
            4. **查看结果**：回答、检索上下文和匹配的目标图像将同步展示
            5. **模型配置**：在顶部"模型配置"中设置 LLM 后端（支持 OpenAI 兼容 API 和 Ollama 本地模型）
            """
        )

        # ==================== 事件绑定 ====================

        # 视频处理
        process_btn.click(
            fn=process_video,
            inputs=[video_input, frame_interval, conf_threshold],
            outputs=[detection_video, process_status, detection_summary, video_id_state]
        )

        # 问答
        ask_btn.click(
            fn=ask_question,
            inputs=[
                question_input, video_id_state,
                backend_radio, api_key_input, api_base_input,
                model_name_input, ollama_model_input, ollama_url_input,
                template_dropdown, temperature_slider, top_k_slider
            ],
            outputs=[answer_output, context_output, matched_images]
        )

        # 快捷问题按钮
        q1_btn.click(fn=lambda: "视频中出现了什么物体？", outputs=question_input)
        q2_btn.click(fn=lambda: "视频中一共检测到多少个目标？", outputs=question_input)
        q3_btn.click(fn=lambda: "请描述视频中的场景内容", outputs=question_input)
        q4_btn.click(fn=lambda: "检测结果的置信度如何？有没有可能的误检？", outputs=question_input)

    return demo


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
