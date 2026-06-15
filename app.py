"""
YOLO-CLIP 多模态问答系统 — 整合版
融合 LLM 问答模块（QAEngine + PromptBuilder + PostProcessor）
与 Gradio 交互界面（真实 YOLODetector + VectorStore）
"""

import os
import sys
import uuid
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GRADIO_TITLE, GRADIO_HOST, GRADIO_PORT, GRADIO_SHARE,
    YOLO_MODEL_PATH, YOLO_FRAME_INTERVAL, YOLO_CONF_THRESHOLD,
    VECTOR_DB_PATH, RETRIEVAL_TOP_K,
)
from yolo_detector import YOLODetector
from vector_store import VectorStore
from llm.factory import create_llm, get_available_backends
from qa.engine import QAEngine
from prompts.templates import PromptBuilder, RetrievedContext


# ============================================================
# 检索适配器：将 VectorStore 的接口适配为 QAEngine 需要的格式
# ============================================================

class VectorStoreRetriever:
    """将真实 VectorStore 适配为 QAEngine 的 retriever 接口"""

    def __init__(self, vector_store: VectorStore, video_id: str):
        self.vector_store = vector_store
        self.video_id = video_id

    def search(self, query_text: str, top_k: int = 5) -> list:
        """检索接口，返回 QAEngine 可用的 SearchResult 列表"""
        from data_types import SearchResult

        results = self.vector_store.search(self.video_id, query_text, top_k)

        search_results = []
        for i, item in enumerate(results):
            doc = item.get("document", "")
            metadata = item.get("metadata", {})
            distance = item.get("distance", 0)

            # 从文档文本中提取类别信息
            labels = []
            try:
                import json
                labels = json.loads(metadata.get("labels", "[]"))
            except (json.JSONDecodeError, TypeError):
                pass

            # 计算相似度（ChromaDB cosine distance → similarity）
            similarity = max(0, 1 - distance)
            class_name = labels[0] if labels else "未知目标"
            # 置信度：优先从向量库结果提取，否则用相似度估算
            confidence = similarity

            # 收集图片路径
            image_path = metadata.get("saved_image", "")
            if not image_path or not os.path.exists(image_path):
                try:
                    import json
                    crop_paths = json.loads(metadata.get("crop_paths", "[]"))
                    image_path = crop_paths[0] if crop_paths else ""
                except (json.JSONDecodeError, TypeError, IndexError):
                    pass

            bbox = (0, 0, 0, 0)

            search_results.append(SearchResult(
                target_id=i + 1,
                class_name=class_name,
                confidence=confidence,
                similarity=similarity,
                description=doc,
                image_path=image_path,
                bbox=bbox,
            ))

        return search_results


# ============================================================
# 全局状态
# ============================================================

OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_BASE, exist_ok=True)

# 延迟初始化的全局组件
_detector = None
_vector_store = None
_video_id = None
# LLM 实例缓存：避免每次提问都重新创建客户端
_llm_cache = {}  # key: (backend, model_name, api_base) -> BaseLLM instance


def get_detector() -> YOLODetector:
    global _detector
    if _detector is None:
        _detector = YOLODetector(model_name=YOLO_MODEL_PATH)
    return _detector


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        db_dir = os.path.join(OUTPUT_BASE, "vector_db")
        _vector_store = VectorStore(persist_dir=db_dir)
    return _vector_store


# ============================================================
# 核心业务逻辑
# ============================================================

def process_video(video_path, frame_interval, conf_threshold, progress=gr.Progress()):
    """处理上传的视频：YOLO 检测 + 向量存储"""
    global _video_id

    if video_path is None:
        gr.Warning("请先上传视频文件！")
        return None, "❌ 未上传视频", "", None

    progress(0, desc="🔄 初始化检测模型...")
    det = get_detector()
    if not det.is_ready():
        return None, "❌ YOLO 模型加载失败，请检查 ultralytics 是否安装", "", None

    vs = get_vector_store()

    video_id = str(uuid.uuid4())[:8]
    _video_id = video_id
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
    if vs.is_ready() and detection_results:
        vs.add_detection_results(video_id, detection_results, detection_summary)

    progress(1.0, desc="✅ 处理完成！")

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
    temperature,
    top_k,
    progress=gr.Progress()
):
    """
    问答主逻辑（整合 QAEngine）：
    问题 → VectorStore检索 → PromptBuilder拼接 → LLM生成 → PostProcessor后处理
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

    # 1. 创建检索适配器
    retriever = VectorStoreRetriever(vs, video_id.strip())

    # 2. 获取或创建 LLM 后端（带缓存）
    try:
        if backend == "ollama":
            cache_key = ("ollama", ollama_model, ollama_url)
            if cache_key not in _llm_cache:
                _llm_cache[cache_key] = create_llm(
                    backend="ollama",
                    model_name=ollama_model,
                    host=ollama_url,
                    temperature=float(temperature),
                )
            llm = _llm_cache[cache_key]
        elif backend == "openai":
            effective_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not effective_key:
                return ("❌ 未提供 API Key，请在界面中填写或设置环境变量 OPENAI_API_KEY",
                        "", [])
            cache_key = ("openai", model_name, api_base, effective_key)
            if cache_key not in _llm_cache:
                _llm_cache[cache_key] = create_llm(
                    backend="openai",
                    model_name=model_name,
                    api_key=effective_key,
                    base_url=api_base,
                    temperature=float(temperature),
                )
                # 缓存最多保留 10 个实例，防止内存泄漏
                if len(_llm_cache) > 10:
                    _llm_cache.pop(next(iter(_llm_cache)))
            llm = _llm_cache[cache_key]
        else:
            llm = create_llm(backend="mock")
    except Exception as e:
        return f"❌ LLM 初始化失败: {str(e)}", "", []

    # 3. 创建 QAEngine 并执行完整问答流程
    progress(0.3, desc="🔄 QAEngine 正在检索 + 生成回答...")
    qa_engine = QAEngine(
        llm=llm,
        retriever=retriever.search,
        domain="视频场景",
        similarity_threshold=0.3,
    )

    try:
        result = qa_engine.ask(question, top_k=int(top_k))
    except Exception as e:
        return f"❌ 问答处理失败:\n{traceback.format_exc()}", "", []

    progress(0.9, desc="🔄 收集匹配图片...")

    # 4. 收集匹配图片
    matched_images = []
    for ctx in result.contexts:
        img_path = ctx.image_path
        if img_path and os.path.exists(img_path):
            matched_images.append((img_path,
                                   f"#{ctx.target_id} {ctx.class_name} "
                                   f"(相似度:{ctx.similarity:.2f})"))

    # 5. 构建检索信息文本
    retrieval_lines = ["### 📊 检索结果"]
    for ctx in result.contexts:
        retrieval_lines.append(
            f"- **#{ctx.target_id}** {ctx.class_name} "
            f"(相似度: {ctx.similarity:.4f}, "
            f"置信度: {ctx.confidence:.0%})"
        )
        retrieval_lines.append(f"  - {ctx.description}")
    retrieval_text = "\n".join(retrieval_lines)

    # 6. 追加系统信息
    time_info = f"⏱ 耗时: {result.elapsed_time:.2f}s"
    if result.confidence:
        time_info += f" | 置信度: {result.confidence}"

    model_info_str = (
        f"🤖 {result.model_info.get('model', 'N/A')} "
        f"({result.model_info.get('backend', 'N/A')})"
    )

    if result.warnings:
        warning_lines = ["", "### ⚠️ 质量提示"]
        for w in result.warnings:
            warning_lines.append(f"- {w}")
        time_info += "\n".join(warning_lines)

    answer_with_info = f"{result.answer}\n\n---\n{time_info}\n{model_info_str}"

    progress(1.0, desc="✅ 回答生成完成！")

    return answer_with_info, retrieval_text, matched_images if matched_images else []


# ============================================================
# Gradio 界面构建
# ============================================================

def build_ui():
    """构建完整的 Gradio 交互界面"""

    with gr.Blocks(title=GRADIO_TITLE) as demo:

        gr.Markdown(
            f"""
            # 🎬 {GRADIO_TITLE}
            ### 基于 YOLO 目标检测 + CLIP 语义对齐 + 向量检索 + LLM 智能问答的端到端视频理解系统
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
                    )
                    model_name_input = gr.Textbox(
                        label="模型名称",
                        value="gpt-3.5-turbo",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Ollama 本地模型配置**")
                    ollama_model_input = gr.Textbox(
                        label="Ollama 模型名称",
                        value="qwen2.5:7b",
                    )
                    ollama_url_input = gr.Textbox(
                        label="Ollama 服务地址",
                        value="http://localhost:11434",
                    )

        # ==================== 主体区域：视频输入与检测 ====================
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📹 视频输入与检测")

                video_input = gr.Video(
                    label="上传视频 / 摄像头采集",
                    sources=["upload", "webcam"],
                    format="mp4"
                )

                with gr.Row():
                    frame_interval = gr.Slider(
                        minimum=5, maximum=120, value=YOLO_FRAME_INTERVAL, step=5,
                        label="检测帧间隔",
                        info="每隔多少帧进行一次检测"
                    )
                    conf_threshold = gr.Slider(
                        minimum=0.1, maximum=0.9, value=YOLO_CONF_THRESHOLD, step=0.05,
                        label="置信度阈值",
                        info="YOLO 检测置信度下限"
                    )

                process_btn = gr.Button("🚀 开始检测处理", variant="primary", size="lg")

                process_status = gr.Textbox(
                    label="处理状态",
                    lines=6,
                    interactive=False
                )

                detection_summary = gr.Textbox(
                    label="检测摘要",
                    lines=5,
                    interactive=False
                )

            with gr.Column(scale=1):
                gr.Markdown("## 🎯 YOLO 检测结果可视化")

                detection_video = gr.Video(
                    label="检测结果视频（含标注框）"
                )

                video_id_state = gr.Textbox(
                    label="视频 ID",
                    interactive=False,
                    visible=True
                )

        gr.Markdown("---")

        # ==================== 问答区域 ====================
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 💬 智能问答")

                question_input = gr.Textbox(
                    label="输入你的问题",
                    placeholder="例如：视频中出现了哪些物体？有多少辆车？",
                    lines=3
                )

                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.0, maximum=1.5, value=0.3, step=0.1,
                        label="生成温度",
                        info="越低越精确"
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=RETRIEVAL_TOP_K, step=1,
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

            with gr.Column(scale=1):
                gr.Markdown("## 📝 LLM 回答输出")

                answer_output = gr.Markdown(
                    value="*等待提问...*",
                )

                with gr.Accordion("🔍 检索详情（点击展开）", open=False):
                    retrieval_detail = gr.Markdown(
                        value="*检索详情将在此显示*",
                    )

                gr.Markdown("## 🖼️ 匹配目标图像")
                matched_gallery = gr.Gallery(
                    label="向量检索匹配的目标",
                    columns=3,
                    rows=2,
                    height=250,
                    object_fit="contain",
                )

        # ==================== 底部说明 ====================
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 操作步骤
            1. **上传视频** — 点击左侧上传区域选择视频文件，或使用摄像头采集
            2. **开始检测** — 点击"开始检测处理"，系统运行 YOLO 检测并将结果存入向量数据库
            3. **输入问题** — 在右侧问题框输入您想了解的内容
            4. **获取回答** — 点击"提问"，LLM 将基于检索上下文生成回答

            ### 问答流程
            用户问题 → 文本特征提取 → 向量库检索匹配 → Prompt 拼接 → LLM 生成回答 → 后处理（幻觉检测）

            ### LLM 后端
            - **OpenAI 兼容 API**：支持 OpenAI、DeepSeek、Qwen 等兼容接口
            - **Ollama 本地模型**：支持 Llama3、Qwen、ChatGLM3 等开源模型
            """)

        # ==================== 事件绑定 ====================

        process_btn.click(
            fn=process_video,
            inputs=[video_input, frame_interval, conf_threshold],
            outputs=[detection_video, process_status, detection_summary, video_id_state]
        )

        ask_btn.click(
            fn=ask_question,
            inputs=[
                question_input, video_id_state,
                backend_radio, api_key_input, api_base_input,
                model_name_input, ollama_model_input, ollama_url_input,
                temperature_slider, top_k_slider
            ],
            outputs=[answer_output, retrieval_detail, matched_gallery]
        )

        # 回车提交
        question_input.submit(
            fn=ask_question,
            inputs=[
                question_input, video_id_state,
                backend_radio, api_key_input, api_base_input,
                model_name_input, ollama_model_input, ollama_url_input,
                temperature_slider, top_k_slider
            ],
            outputs=[answer_output, retrieval_detail, matched_gallery]
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
    print("=" * 60)
    print(f"  {GRADIO_TITLE}")
    print("=" * 60)
    print(f"\n可用的 LLM 后端: {get_available_backends()}")
    print(f"\n启动 Gradio 界面...")
    print(f"本地地址: http://{GRADIO_HOST}:{GRADIO_PORT}")

    demo = build_ui()
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        show_error=True,
        inbrowser=True,
    )