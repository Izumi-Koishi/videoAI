"""
YOLO-CLIP 多模态问答系统 — 启动入口
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GRADIO_HOST, GRADIO_PORT, GRADIO_SHARE
import gradio as gr
from interface.gradio_app import create_app, CSS
from llm.factory import get_available_backends


def main():
    print("=" * 60)
    print("  YOLO-CLIP 多模态问答系统")
    print("=" * 60)
    print(f"\n可用的 LLM 后端: {get_available_backends()}")
    print(f"当前后端: Mock 演示模式")
    print(f"\n启动 Gradio 界面...")
    print(f"本地地址: http://{GRADIO_HOST}:{GRADIO_PORT}")

    app = create_app()
    app.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        show_error=True,
        css=CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
