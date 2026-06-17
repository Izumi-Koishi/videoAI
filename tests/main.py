#!/usr/bin/env python3
"""
videoAI 集成系统 — 主入口 (main.py)
====================================
校园监控视频 AI 智能问答系统 — 统一启动与测试入口

用法:
    python main.py --mode web          # 启动 Web 界面 (Gradio)
    python main.py --mode test         # 运行集成测试
    python main.py --mode process      # 处理视频 (YOLO检测 + 向量入库)
    python main.py --mode pipeline     # 完整管线 (处理 + 测试)
    python main.py --mode interactive  # 交互式命令行问答
    python main.py --mode verify       # 系统环境验证

环境要求: Python 3.10+
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# 路径初始化 — 确保可导入项目各模块
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TESTS_ROOT = Path(__file__).resolve().parent

# 将各模块路径加入 sys.path
for _mod_dir in [
    _PROJECT_ROOT / "llm_gradio",
    _PROJECT_ROOT / "vectorDB",
    _PROJECT_ROOT / "yolo1" / "src",
    _PROJECT_ROOT / "yolo2" / "src",
]:
    if _mod_dir.exists():
        sys.path.insert(0, str(_mod_dir))

# 确保 tests 目录在 path 中，以导入 test_config
sys.path.insert(0, str(_TESTS_ROOT))

# 导入统一配置
from test_config.config import (
    PROJECT_ROOT, DATA_DIR, TESTS_DIR, OUTPUT_DIR, LOG_DIR,
    KNOWLEDGE_BASE_PATH, TEST_QA_PATH, TEST_VIDEO_PATH,
    YOLO_CONFIG, CLIP_CONFIG, VECTOR_DB_CONFIG, LLM_CONFIG, GRADIO_CONFIG,
    DETECTION_OUTPUT_DIR, VECTOR_DB_PATH,
)

# ============================================================
# 日志工具
# ============================================================
def _log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)
    # 同时写入日志文件
    log_file = LOG_DIR / f"videoai_{datetime.now().strftime('%Y%m%d')}.log"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except IOError:
        pass


# ============================================================
# 系统环境验证
# ============================================================
def verify_environment() -> Dict[str, bool]:
    """验证系统运行环境，返回各项检查结果"""
    _log("=" * 60)
    _log("  videoAI 系统环境验证")
    _log("=" * 60)

    results = {}

    # Python 版本
    py_ver = sys.version_info
    py_ok = py_ver >= (3, 10)
    results["Python 3.10+"] = py_ok
    _log(f"  Python {py_ver.major}.{py_ver.minor}.{py_ver.micro} {'[OK]' if py_ok else '[FAIL] (need 3.10+)'} ")

    # 数据文件
    for name, path in [
        ("知识库", KNOWLEDGE_BASE_PATH),
        ("测试问答", TEST_QA_PATH),
    ]:
        ok = path.exists()
        results[f"数据文件: {name}"] = ok
        _log(f"  {name}: {path} {'[OK]' if ok else '[FAIL] 未找到'} ")

    # 视频文件
    video_ok = TEST_VIDEO_PATH.exists()
    results["测试视频"] = video_ok
    _log(f"  测试视频: {'[OK]' if video_ok else '[FAIL] 未找到（部分测试需要）'} ")

    # 核心依赖
    for mod_name, pkg_name in [
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("chromadb", "chromadb"),
        ("gradio", "gradio"),
        ("openai", "openai"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
    ]:
        try:
            __import__(mod_name)
            results[f"依赖: {pkg_name}"] = True
            _log(f"  {pkg_name} [OK]")
        except ImportError:
            results[f"依赖: {pkg_name}"] = False
            _log(f"  {pkg_name} [FAIL] 未安装")

    # 可选依赖
    for mod_name, pkg_name in [
        ("sentence_transformers", "sentence-transformers"),
    ]:
        try:
            __import__(mod_name)
            results[f"可选依赖: {pkg_name}"] = True
            _log(f"  {pkg_name} [OK]")
        except ImportError:
            results[f"可选依赖: {pkg_name}"] = False
            _log(f"  {pkg_name} [FAIL] 未安装（推荐）")

    # 总结
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    _log(f"\n  总计: {passed}/{total} 项通过")
    if passed == total:
        _log("  环境验证全部通过！")
    else:
        failed_items = [k for k, v in results.items() if not v]
        _log(f"  未通过项: {', '.join(failed_items)}")
        _log("  请安装缺失的依赖: pip install -r tests/requirements.txt")

    _log("=" * 60)
    return results


# ============================================================
# 视频处理管线
# ============================================================
def process_video_pipeline(
    video_path: str,
    frame_interval: int = None,
    conf_threshold: float = None,
) -> Tuple[Optional[str], Optional[List[Dict]], Optional[str]]:
    """
    处理视频的完整管线: YOLO 检测 → 向量数据库存储

    Returns:
        (video_id, detection_results, detection_summary)
    """
    _log(f"开始处理视频: {video_path}")

    # --- 初始化 YOLO 检测器 ---
    _log("初始化 YOLO 检测器...")
    try:
        from yolo_detector import YOLODetector
        det = YOLODetector(
            model_name=YOLO_CONFIG["model_name"]
        )
        if not det.is_ready():
            _log("YOLO 模型加载失败！请检查 ultralytics 安装", "ERROR")
            return None, None, None
        _log("YOLO 检测器就绪")
    except ImportError as e:
        _log(f"导入 yolo_detector 失败: {e}", "ERROR")
        return None, None, None

    # --- 初始化向量数据库 ---
    _log("初始化向量数据库...")
    try:
        from vector_store import VectorStore
        vs = VectorStore(persist_dir=str(VECTOR_DB_PATH))
        if not vs.is_ready():
            _log("向量数据库未就绪！请检查 chromadb 安装", "ERROR")
            return None, None, None
        _log("向量数据库就绪")
    except ImportError as e:
        _log(f"导入 vector_store 失败: {e}", "ERROR")
        return None, None, None

    # --- 生成视频ID和输出目录 ---
    import uuid
    video_id = str(uuid.uuid4())[:8]
    video_name = Path(video_path).stem
    output_dir = os.path.join(
        DETECTION_OUTPUT_DIR, video_id
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- YOLO 检测 ---
    fi = frame_interval if frame_interval is not None else YOLO_CONFIG["frame_interval"]
    conf = conf_threshold if conf_threshold is not None else YOLO_CONFIG["confidence_threshold"]

    _log(f"YOLO 检测参数: 帧间隔={fi}, 置信度阈值={conf}")
    try:
        detection_results, saved_images, detection_video_path = det.process_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=int(fi),
            conf=float(conf),
        )
    except Exception as e:
        _log(f"视频处理失败: {e}", "ERROR")
        return None, None, None

    total_objects = sum(len(r.get("labels", [])) for r in detection_results)
    unique_labels = set()
    for r in detection_results:
        unique_labels.update(r.get("labels", []))
    _log(f"检测完成: {len(detection_results)} 帧, {total_objects} 个目标, 类别: {unique_labels}")

    # --- 生成摘要 ---
    detection_summary = det.generate_detection_summary(detection_results)

    # --- 向量入库 ---
    if detection_results:
        _log("存储检测结果到向量数据库...")
        vs.add_detection_results(video_id, detection_results, detection_summary)
        _log(f"向量入库完成，视频ID: {video_id}")

    # --- 保存检测结果元数据 ---
    metadata_path = os.path.join(output_dir, "detection_metadata.json")
    serializable_results = []
    for r in detection_results:
        item = {
            "frame_index": r.get("frame_index", 0),
            "timestamp": r.get("timestamp", 0),
            "labels": r.get("labels", []),
            "scores": r.get("scores", []),
            "saved_image": r.get("saved_image", ""),
            "cropped_objects": [
                {"label": c.get("label", ""), "image_path": c.get("image_path", "")}
                for c in r.get("cropped_objects", [])
            ]
        }
        serializable_results.append(item)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    _log(f"检测元数据已保存: {metadata_path}")

    return video_id, detection_results, detection_summary


# ============================================================
# 单次问答
# ============================================================
def ask_question(
    question: str,
    video_id: str,
    top_k: int = 5,
    template_name: str = "auto",
    temperature: float = None,
) -> Tuple[str, str, List[str]]:
    """
    基于已处理的视频进行问答
    Returns:
        (answer, context_text, image_paths)
    """
    from vector_store import VectorStore
    from llm_engine import LLMEngine

    vs = VectorStore(persist_dir=str(VECTOR_DB_PATH))
    if not vs.is_ready():
        return "❌ 向量数据库未就绪", "", []

    # 向量检索
    context_text, image_paths = vs.get_context_and_images(
        video_id=video_id,
        query=question,
        top_k=top_k,
    )

    if not context_text:
        return "❌ 未检索到相关内容", "", []

    # LLM 生成
    engine = LLMEngine(
        backend=LLM_CONFIG["backend"],
        api_key=LLM_CONFIG["api_key"],
        api_base=LLM_CONFIG["api_base"],
        model_name=LLM_CONFIG["model_name"],
        ollama_model=LLM_CONFIG["ollama_model"],
        ollama_url=LLM_CONFIG["ollama_url"],
    )

    if not engine.is_ready():
        if LLM_CONFIG["backend"] == "openai":
            return "❌ LLM 未就绪：请设置 OPENAI_API_KEY 环境变量或修改 config.py", context_text, image_paths[:6]
        else:
            return "❌ LLM 引擎未就绪", context_text, image_paths[:6]

    temp = temperature if temperature is not None else LLM_CONFIG["temperature"]
    answer = engine.generate(
        question=question,
        context=context_text,
        template_name=template_name,
        temperature=float(temp),
    )

    return answer, context_text, image_paths[:6]


# ============================================================
# 交互式问答模式
# ============================================================
def interactive_mode():
    """命令行交互式问答"""
    _log("进入交互式问答模式")

    # 检查是否已有处理过的视频
    import uuid

    print("\n" + "=" * 60)
    print("  videoAI 交互式问答")
    print("=" * 60)

    # 视频选择
    video_files = list(DATA_DIR.glob("*.mp4"))
    if not video_files:
        video_files = list(Path(PROJECT_ROOT / "data").glob("*.mp4"))

    video_id = None
    video_path = None
    if video_files:
        print(f"\n找到 {len(video_files)} 个视频文件:")
        for i, vf in enumerate(video_files):
            print(f"  [{i}] {vf.name}")
        print(f"  [N] 跳过，使用已有 video_id")

        choice = input("\n请选择视频 (输入序号): ").strip()
        if choice.isdigit() and int(choice) < len(video_files):
            video_path = str(video_files[int(choice)])
            print(f"正在处理视频: {video_path}")
            video_id, _, _ = process_video_pipeline(video_path)
            if video_id is None:
                print("视频处理失败！")
                return

    if video_id is None:
        video_id = input("请输入已有的 video_id: ").strip()
        if not video_id:
            print("未提供 video_id，退出。")
            return

    print(f"\n当前 video_id: {video_id}")
    print("输入问题开始问答，输入 'quit' 退出。")
    print("-" * 60)

    while True:
        question = input("\n🤔 你的问题: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        print("🔄 检索并生成回答...")
        answer, context, images = ask_question(
            question=question,
            video_id=video_id,
        )

        print(f"\n📝 回答:\n{answer}")
        if images:
            print(f"🖼️ 匹配图片 ({len(images)} 张):")
            for img in images[:3]:
                print(f"   - {img}")

# ============================================================
# 启动 Web 界面
# ============================================================
def launch_web_ui():
    """启动 Gradio Web 界面"""
    _log("启动 Gradio Web 界面...")

    # 导入 app.py 中的 Gradio 应用
    app_module_path = str(_PROJECT_ROOT / "llm_gradio" / "app.py")

    if not os.path.exists(app_module_path):
        _log(f"Web 界面文件未找到: {app_module_path}", "ERROR")
        return

    # 切换到 llm_gradio 目录以确保相对路径正确
    original_cwd = os.getcwd()
    os.chdir(str(_PROJECT_ROOT / "llm_gradio"))

    try:
        import gradio as gr
        import runpy

        # 使用 runpy 运行 app.py 的 __main__ 部分
        # 直接 import app 并调用
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", app_module_path)
        app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_mod)

        demo = app_mod.build_ui()
        _log(f"Web 界面启动: http://{GRADIO_CONFIG['host']}:{GRADIO_CONFIG['port']}")
        demo.launch(
            server_name=GRADIO_CONFIG["host"],
            server_port=GRADIO_CONFIG["port"],
            share=GRADIO_CONFIG["share"],
            inbrowser=True,
        )
    except Exception as e:
        _log(f"Web 界面启动失败: {e}", "ERROR")
        raise
    finally:
        os.chdir(original_cwd)


# ============================================================
# 完整管线模式
# ============================================================
def full_pipeline():
    """完整集成管线：视频处理 → 向量入库 → 自动化测试"""
    _log("=" * 60)
    _log("  videoAI 完整集成管线")
    _log("=" * 60)

    # 1. 环境验证
    results = verify_environment()
    core_ok = all(results.get(k, False) for k in ["Python 3.10+", "依赖: ultralytics", "依赖: chromadb", "依赖: openai"])
    if not core_ok:
        _log("核心依赖不满足，请先安装依赖", "ERROR")
        return

    # 2. 找视频文件
    video_path = None
    if TEST_VIDEO_PATH.exists():
        video_path = str(TEST_VIDEO_PATH)
    else:
        video_files = list(DATA_DIR.glob("*.mp4"))
        if video_files:
            video_path = str(video_files[0])

    if video_path is None:
        _log("未找到测试视频文件，请将视频放入 data/ 目录", "ERROR")
        return

    # 3. 视频处理
    _log(f"使用视频: {video_path}")
    video_id, _, detection_summary = process_video_pipeline(video_path)

    if video_id is None:
        _log("视频处理管线失败", "ERROR")
        return

    # 4. 加载测试问答
    if not TEST_QA_PATH.exists():
        _log(f"测试问答文件未找到: {TEST_QA_PATH}", "ERROR")
        return

    with open(TEST_QA_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    qa_pairs = qa_data.get("qa_pairs", [])
    _log(f"加载了 {len(qa_pairs)} 条测试问答对")

    # 5. 运行测试
    _log("开始自动化问答测试...")
    test_results = []
    passed = 0
    failed = 0

    for i, qa in enumerate(qa_pairs):
        q_id = qa["id"]
        question = qa["question"]
        expected = qa["expected_answer"]
        category = qa["category"]
        difficulty = qa["difficulty"]

        _log(f"[{i+1}/{len(qa_pairs)}] {q_id} [{category}][{difficulty}]: {question}")

        answer, context, images = ask_question(
            question=question,
            video_id=video_id,
        )

        # 简单判断：回答是否包含预期关键词
        expected_lower = expected.replace("（", "(").replace("）", ")").lower()
        # 提取预期答案中的核心词
        core_answer = expected.split("（")[0].split("(")[0].strip().lower()
        answer_lower = answer.lower()

        is_pass = core_answer in answer_lower and len(answer) > 10

        test_result = {
            "id": q_id,
            "category": category,
            "difficulty": difficulty,
            "question": question,
            "expected_answer": expected,
            "actual_answer": answer[:500],
            "pass": is_pass,
        }
        test_results.append(test_result)

        if is_pass:
            passed += 1
            _log(f"  [OK] 通过 (预期核心: {core_answer})")
        else:
            failed += 1
            _log(f"  [FAIL] 未通过 (预期核心: {core_answer}, 实际开头: {answer[:100]}...)")

    # 6. 生成测试报告
    report_path = OUTPUT_DIR / "pipeline_test_report.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "video_id": video_id,
        "video_path": video_path,
        "total": len(qa_pairs),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed / len(qa_pairs) * 100:.1f}%" if qa_pairs else "N/A",
        "results": test_results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _log("=" * 60)
    _log(f"  管线测试完成: {passed}/{len(qa_pairs)} 通过, 通过率 {report['pass_rate']}")
    _log(f"  报告已保存: {report_path}")
    _log("=" * 60)


# ============================================================
# 命令行参数解析与主函数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="videoAI — 校园监控视频 AI 智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode verify        # 验证系统环境
  python main.py --mode process       # 处理测试视频
  python main.py --mode interactive   # 交互式问答
  python main.py --mode test          # 运行集成测试
  python main.py --mode pipeline      # 完整管线（处理+测试）
  python main.py --mode web           # 启动 Web 界面
        """,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["verify", "process", "interactive", "test", "pipeline", "web"],
        default="verify",
        help="运行模式 (默认: verify)",
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="指定视频文件路径 (process/pipeline 模式)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="指定配置文件路径",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="跳过 LLM 相关测试 (test 模式)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        help="YOLO 检测帧间隔 (process/pipeline 模式)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="YOLO 置信度阈值 (process/pipeline 模式)",
    )
    parser.add_argument(
        "--qa-sample",
        type=int,
        default=0,
        help="测试问答采样数量，0=全部 (test/pipeline 模式)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    _log(f"videoAI 集成系统启动，模式: {mode}")

    if mode == "verify":
        results = verify_environment()
        sys.exit(0 if all(results.values()) else 1)

    elif mode == "process":
        video = args.video
        if video is None:
            # 自动寻找视频
            if TEST_VIDEO_PATH.exists():
                video = str(TEST_VIDEO_PATH)
            else:
                candidates = list(DATA_DIR.glob("*.mp4"))
                if candidates:
                    video = str(candidates[0])
                else:
                    _log("未找到视频文件，请用 --video 指定", "ERROR")
                    sys.exit(1)

        video_id, _, summary = process_video_pipeline(
            video_path=video,
            frame_interval=args.frame_interval,
            conf_threshold=args.conf_threshold,
        )
        if video_id:
            _log(f"处理完成! video_id: {video_id}")
            _log(f"检测摘要:\n{summary}")
        else:
            _log("处理失败", "ERROR")
            sys.exit(1)

    elif mode == "interactive":
        interactive_mode()

    elif mode == "test":
        _log("运行集成测试...")
        # 动态加载测试模块
        test_script = _TESTS_ROOT / "integration_test.py"
        if test_script.exists():
            import runpy
            sys.argv = [
                str(test_script),
                "--video", args.video or "",
                "--output", str(OUTPUT_DIR),
                "--report", str(OUTPUT_DIR / "integration_test_report"),
            ]
            if args.skip_llm:
                sys.argv.append("--skip-llm")
            if args.qa_sample > 0:
                sys.argv.extend(["--sample", str(args.qa_sample)])
            runpy.run_path(str(test_script), run_name="__main__")
        else:
            _log(f"测试脚本未找到: {test_script}", "ERROR")
            sys.exit(1)

    elif mode == "pipeline":
        full_pipeline()

    elif mode == "web":
        launch_web_ui()


if __name__ == "__main__":
    main()
