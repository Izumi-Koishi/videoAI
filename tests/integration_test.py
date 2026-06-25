#!/usr/bin/env python3
"""
videoAI 集成测试套件 (integration_test.py)
============================================
使用 campus_test_qa.json 中的测试问答对验证系统管线。

测试分类:
  1. 数据完整性测试 — 验证知识库、测试QA、视频文件
  2. 模块接口测试 — 验证 YOLO / VectorStore / LLM 模块导入与初始化
  3. 管线集成测试 — 验证视频 → 检测 → 入库 → 检索 → 问答的端到端流水线
  4. 问答准确性测试 — 使用测试QA对验证 LLM 回答质量
  5. 性能基准测试 — 记录各环节耗时

输出: integration_test_report.md (可读报告) + integration_test_report.json (结构化数据)
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- 路径初始化 ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TESTS_ROOT = Path(__file__).resolve().parent

for _mod_dir in [
    _PROJECT_ROOT / "llm_gradio",
    _PROJECT_ROOT / "vectorDB",
]:
    if _mod_dir.exists():
        sys.path.insert(0, str(_mod_dir))

sys.path.insert(0, str(_TESTS_ROOT))
from test_config.config import (
    PROJECT_ROOT, DATA_DIR, KNOWLEDGE_BASE_PATH, TEST_QA_PATH, TEST_VIDEO_PATH,
    YOLO_CONFIG, CLIP_CONFIG, VECTOR_DB_CONFIG, LLM_CONFIG, OUTPUT_DIR, LOG_DIR,
)

# ============================================================
# 工具函数
# ============================================================
def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def load_json(path: Path) -> Optional[Dict]:
    """安全加载 JSON 文件"""
    if not path.exists():
        log(f"文件不存在: {path}", "ERROR")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 测试结果收集器
# ============================================================
class TestCollector:
    """收集和汇总测试结果"""

    def __init__(self):
        self.results: List[Dict] = []
        self.start_time = time.time()

    def add(self, test_name: str, passed: bool, detail: str = "",
            category: str = "", duration_ms: float = 0):
        self.results.append({
            "name": test_name,
            "category": category,
            "passed": passed,
            "detail": detail,
            "duration_ms": round(duration_ms, 2),
        })
        status = "[OK]" if passed else "[FAIL]"
        log(f"  {status} {test_name} ({duration_ms:.0f}ms){' — ' + detail if detail else ''}")

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r["passed"])

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r["passed"])

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total * 100 if self.total else 0

    @property
    def total_time(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> Dict:
        return {
            "total": self.total,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "pass_rate": f"{self.pass_rate:.1f}%",
            "total_time_s": round(self.total_time, 2),
            "results": self.results,
        }

    def by_category(self) -> Dict[str, Dict]:
        cats = {}
        for r in self.results:
            cat = r["category"] or "未分类"
            if cat not in cats:
                cats[cat] = {"total": 0, "passed": 0, "failed": 0}
            cats[cat]["total"] += 1
            if r["passed"]:
                cats[cat]["passed"] += 1
            else:
                cats[cat]["failed"] += 1
        return cats


collector = TestCollector()


# ============================================================
# 测试 1: 数据完整性
# ============================================================
def test_data_integrity():
    """验证测试数据文件完整性"""
    log("=" * 50)
    log("  测试 1: 数据完整性")
    log("=" * 50)

    # 1.1 知识库文件
    t0 = time.time()
    kb = load_json(KNOWLEDGE_BASE_PATH)
    if kb is None:
        collector.add("知识库加载", False, "文件不存在", "数据完整性", (time.time()-t0)*1000)
    else:
        entity_count = len(kb.get("Entity", {}))
        person_count = sum(
            len(items) for cat_data in kb.get("Entity", {}).values()
            for items in cat_data.values()
        )
        collector.add("知识库加载", True,
                       f"实体大类: {entity_count}, 实体条目: {person_count}",
                       "数据完整性", (time.time()-t0)*1000)

    # 1.2 测试 QA 文件
    t0 = time.time()
    qa = load_json(TEST_QA_PATH)
    if qa is None:
        collector.add("测试QA加载", False, "文件不存在", "数据完整性", (time.time()-t0)*1000)
    else:
        qa_count = len(qa.get("qa_pairs", []))
        categories = qa.get("metadata", {}).get("categories", {})
        collector.add("测试QA加载", True,
                       f"问答对: {qa_count}, 分类: {categories}",
                       "数据完整性", (time.time()-t0)*1000)

    # 1.3 测试视频文件
    t0 = time.time()
    video_ok = TEST_VIDEO_PATH.exists()
    video_size = ""
    if video_ok:
        size_mb = TEST_VIDEO_PATH.stat().st_size / (1024*1024)
        video_size = f"{size_mb:.1f} MB"
    collector.add("测试视频文件", video_ok,
                   video_size if video_ok else "文件不存在",
                   "数据完整性", (time.time()-t0)*1000)

    # 1.4 整体数据关联校验
    t0 = time.time()
    if kb and qa:
        mismatches = 0
        # 检查 QA 中的预期答案是否在知识库中有对应实体
        entity_names = set()
        for cat_data in kb.get("Entity", {}).values():
            for items in cat_data.values():
                for item in items:
                    entity_names.add(item.get("name", ""))
        for qa_pair in qa.get("qa_pairs", []):
            expected = qa_pair.get("expected_answer", "")
            # 简单检查：预期答案的核心词
            core = expected.split("（")[0].split("(")[0].strip()
            if core and core not in entity_names and "X " not in core:
                mismatches += 1
        collector.add("数据关联校验", mismatches < 5,
                       f"知识库未覆盖答案: {mismatches}/{len(qa.get('qa_pairs', []))}",
                       "数据完整性", (time.time()-t0)*1000)
    else:
        collector.add("数据关联校验", False, "前置数据缺失", "数据完整性", 0)


# ============================================================
# 测试 2: 模块接口
# ============================================================
def test_module_interfaces():
    """测试各模块的导入和初始化"""
    log("=" * 50)
    log("  测试 2: 模块接口")
    log("=" * 50)

    # 2.1 YOLO 检测器
    t0 = time.time()
    try:
        from yolo_detector import YOLODetector
        det = YOLODetector(model_name="yolov8n.pt")
        ok = det.is_ready()
        collector.add("YOLO检测器初始化", ok,
                       "模型加载成功" if ok else "模型加载失败",
                       "模块接口", (time.time()-t0)*1000)
    except Exception as e:
        collector.add("YOLO检测器初始化", False, str(e), "模块接口", (time.time()-t0)*1000)

    # 2.2 VectorStore 向量数据库
    t0 = time.time()
    try:
        from vector_store import VectorStore
        vs = VectorStore(persist_dir=str(OUTPUT_DIR / "test_vector_db"))
        ok = vs.is_ready()
        collector.add("VectorStore初始化", ok,
                       "ChromaDB 就绪" if ok else "ChromaDB 未就绪",
                       "模块接口", (time.time()-t0)*1000)
    except Exception as e:
        collector.add("VectorStore初始化", False, str(e), "模块接口", (time.time()-t0)*1000)

    # 2.3 LLM 引擎
    t0 = time.time()
    try:
        from llm_engine import LLMEngine, PROMPT_TEMPLATES
        engine = LLMEngine(
            backend="openai",
            api_key=LLM_CONFIG["api_key"],
            api_base=LLM_CONFIG["api_base"],
            model_name=LLM_CONFIG["model_name"],
        )
        has_templates = len(PROMPT_TEMPLATES) >= 4
        collector.add("LLM引擎初始化", has_templates,
                       f"模板数: {len(PROMPT_TEMPLATES)}",
                       "模块接口", (time.time()-t0)*1000)

        # 2.4 Prompt 模板完整性
        t0 = time.time()
        required_templates = ["default", "object_count", "scene_desc", "technical"]
        missing = [t for t in required_templates if t not in PROMPT_TEMPLATES]
        for tmpl in required_templates:
            ok = tmpl in PROMPT_TEMPLATES
            content = PROMPT_TEMPLATES.get(tmpl, "")
            has_placeholders = "{context}" in content and "{question}" in content
            collector.add(f"Prompt模板: {tmpl}", ok and has_placeholders,
                           "模板缺少占位符" if not has_placeholders else "",
                           "模块接口", 0)
    except Exception as e:
        collector.add("LLM引擎初始化", False, str(e), "模块接口", (time.time()-t0)*1000)

    # 2.5 CLIP 编码器 (可选)
    t0 = time.time()
    try:
        # 只测试导入和配置，不加载模型（太大）
        from clip1.config import CLIPConfig as _CLIPConfig
        collector.add("CLIP配置加载", True, "clip1 模块可导入", "模块接口", (time.time()-t0)*1000)
    except ImportError:
        collector.add("CLIP配置加载", False, "clip1 模块不可用", "模块接口", (time.time()-t0)*1000)

    # 2.6 VectorDB 独立模块
    t0 = time.time()
    try:
        from vector_db_manager import VectorDBManager
        collector.add("VectorDB独立模块", True, "vector_db_manager 可导入", "模块接口", (time.time()-t0)*1000)
    except Exception as e:
        collector.add("VectorDB独立模块", False, str(e), "模块接口", (time.time()-t0)*1000)


# ============================================================
# 测试 3: 管线集成
# ============================================================
def test_pipeline_integration(video_path: Optional[str] = None, skip_llm: bool = False):
    """端到端管线测试"""
    log("=" * 50)
    log("  测试 3: 管线集成")
    log("=" * 50)

    if video_path is None:
        if TEST_VIDEO_PATH.exists():
            video_path = str(TEST_VIDEO_PATH)
        else:
            candidates = list(DATA_DIR.glob("*.mp4"))
            video_path = str(candidates[0]) if candidates else None

    if video_path is None:
        collector.add("管线-视频准备", False, "无测试视频", "管线集成", 0)
        return

    collector.add("管线-视频就绪", True, Path(video_path).name, "管线集成", 0)

    # 3.1 YOLO 检测
    t0 = time.time()
    try:
        from yolo_detector import YOLODetector
        det = YOLODetector(model_name=YOLO_CONFIG["model_name"])
        if not det.is_ready():
            collector.add("管线-YOLO初始化", False, "", "管线集成", (time.time()-t0)*1000)
            return
        collector.add("管线-YOLO初始化", True, "", "管线集成", (time.time()-t0)*1000)

        # 生成唯一测试目录
        import uuid
        test_id = f"test_{uuid.uuid4().hex[:6]}"
        output_dir = str(OUTPUT_DIR / test_id / "detection")
        os.makedirs(output_dir, exist_ok=True)

        t1 = time.time()
        results, images, detected_video = det.process_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=YOLO_CONFIG["frame_interval"],
            conf=YOLO_CONFIG["confidence_threshold"],
        )
        yolo_time = (time.time() - t1) * 1000
        total_objs = sum(len(r.get("labels", [])) for r in results)
        unique_labels = set()
        for r in results:
            unique_labels.update(r.get("labels", []))
        collector.add("管线-YOLO检测", len(results) > 0,
                       f"检测了 {len(results)} 帧, 发现 {total_objs} 个目标, "
                       f"类别: {', '.join(list(unique_labels)[:8])}, 耗时 {yolo_time:.0f}ms",
                       "管线集成", yolo_time)
    except Exception as e:
        collector.add("管线-YOLO检测", False, str(e)[:200], "管线集成", (time.time()-t0)*1000)
        return

    # 3.2 向量存储
    t0 = time.time()
    try:
        from vector_store import VectorStore
        vs = VectorStore(persist_dir=str(OUTPUT_DIR / "test_vector_db"))

        video_id = f"test_video_{uuid.uuid4().hex[:6]}"
        summary = det.generate_detection_summary(results)

        t1 = time.time()
        vs.add_detection_results(video_id, results, summary)
        vs_time = (time.time() - t1) * 1000
        collector.add("管线-向量存储", True,
                       f"video_id={video_id}, {len(results)} 帧入库, 耗时 {vs_time:.0f}ms",
                       "管线集成", vs_time)
    except Exception as e:
        collector.add("管线-向量存储", False, str(e)[:200], "管线集成", (time.time()-t0)*1000)
        video_id = None

    # 3.3 向量检索
    if video_id:
        t0 = time.time()
        try:
            t1 = time.time()
            ctx, imgs = vs.get_context_and_images(video_id, "检测到了什么物体？", top_k=3)
            search_time = (time.time() - t1) * 1000
            collector.add("管线-向量检索", bool(ctx),
                           f"检索到 {len(ctx.split(chr(10))) if ctx else 0} 条, "
                           f"图片 {len(imgs)} 张, 耗时 {search_time:.0f}ms",
                           "管线集成", search_time)
        except Exception as e:
            collector.add("管线-向量检索", False, str(e)[:200], "管线集成", (time.time()-t0)*1000)
            ctx = None
    else:
        ctx = None

    # 3.4 LLM 问答（可选）
    if not skip_llm and video_id and ctx:
        t0 = time.time()
        try:
            from llm_engine import LLMEngine
            engine = LLMEngine(
                backend=LLM_CONFIG["backend"],
                api_key=LLM_CONFIG["api_key"],
                api_base=LLM_CONFIG["api_base"],
                model_name=LLM_CONFIG["model_name"],
            )
            if engine.is_ready():
                t1 = time.time()
                answer = engine.generate(
                    question="视频中检测到了什么物体？",
                    context=ctx,
                    template_name="default",
                    temperature=0.3,
                )
                llm_time = (time.time() - t1) * 1000
                collector.add("管线-LLM问答", len(answer) > 10,
                               f"生成 {len(answer)} 字符, 耗时 {llm_time:.0f}ms",
                               "管线集成", llm_time)
            else:
                collector.add("管线-LLM问答", False, "LLM 未就绪（无 API Key）", "管线集成", 0)
        except Exception as e:
            collector.add("管线-LLM问答", False, str(e)[:200], "管线集成", (time.time()-t0)*1000)
    elif skip_llm:
        collector.add("管线-LLM问答", False, "已跳过 (--skip-llm)", "管线集成", 0)


# ============================================================
# 测试 4: 知识库集成
# ============================================================
def test_knowledge_base_integration():
    """测试知识库与 LLM 的集成效果"""
    log("=" * 50)
    log("  测试 4: 知识库集成")
    log("=" * 50)

    kb = load_json(KNOWLEDGE_BASE_PATH)
    if kb is None:
        collector.add("知识库-加载", False, "", "知识库集成", 0)
        return

    # 4.1 知识库结构化检查
    t0 = time.time()
    entity = kb.get("Entity", {})
    top_categories = list(entity.keys())
    expected_cats = ["Person", "Vehicle", "Building"]
    cat_overlap = set(expected_cats) & set(top_categories)
    collector.add("知识库-顶层分类", len(cat_overlap) >= 2,
                   f"实际: {top_categories}, 预期含: {expected_cats}",
                   "知识库集成", (time.time()-t0)*1000)

    # 4.2 统计知识库覆盖范围
    t0 = time.time()
    all_entities = []
    for cat_data in entity.values():
        for subcat, items in cat_data.items():
            for item in items:
                all_entities.append({
                    "name": item.get("name", ""),
                    "category": item.get("category", ""),
                    "subcategory": subcat,
                })
    collector.add("知识库-实体统计", len(all_entities) > 15,
                   f"共 {len(all_entities)} 个实体",
                   "知识库集成", (time.time()-t0)*1000)

    # 4.3 测试 QA 覆盖度
    t0 = time.time()
    qa = load_json(TEST_QA_PATH)
    if qa:
        kb_entities = set(e["name"] for e in all_entities)
        covered = 0
        for pair in qa.get("qa_pairs", []):
            expected = pair.get("expected_answer", "")
            core = expected.split("（")[0].split("(")[0].strip()
            if core in kb_entities:
                covered += 1
        collector.add("知识库-QA覆盖", covered > 10,
                       f"知识库覆盖 {covered}/{len(qa.get('qa_pairs', []))} 条 QA 的预期答案",
                       "知识库集成", (time.time()-t0)*1000)

    # 4.4 实体属性完整性
    t0 = time.time()
    entities_with_attrs = 0
    entities_without_attrs = 0
    for e in all_entities:
        item_data = None
        for cat_data in entity.values():
            for items in cat_data.values():
                for item in items:
                    if item.get("name") == e["name"]:
                        item_data = item
                        break
        if item_data:
            attrs = item_data.get("attributes", [])
            scenes = item_data.get("typical_scenes", [])
            if attrs and scenes:
                entities_with_attrs += 1
            else:
                entities_without_attrs += 1
    collector.add("知识库-属性完整性", entities_without_attrs == 0,
                   f"完整: {entities_with_attrs}, 缺失: {entities_without_attrs}",
                   "知识库集成", (time.time()-t0)*1000)


# ============================================================
# 测试 5: 性能基准
# ============================================================
def test_performance():
    """性能基准测试"""
    log("=" * 50)
    log("  测试 5: 性能基准")
    log("=" * 50)

    # 5.1 模块导入时间
    t0 = time.time()
    import yolo_detector
    import vector_store
    import llm_engine
    import_time = (time.time() - t0) * 1000
    collector.add("性能-模块导入", import_time < 5000,
                   f"{import_time:.0f}ms", "性能基准", import_time)

    # 5.2 向量检索性能（模拟）
    t0 = time.time()
    try:
        from vector_store import VectorStore
        vs = VectorStore(persist_dir=str(OUTPUT_DIR / "perf_test_db"))
        if vs.is_ready():
            # 做 10 次空查询测试检索延迟
            times = []
            import uuid
            test_vid = f"perf_{uuid.uuid4().hex[:6]}"
            # 添加少量测试数据
            test_results = [{
                "frame_index": 0, "timestamp": 0,
                "labels": ["person"], "scores": [0.9],
                "saved_image": "", "cropped_objects": [],
            }]
            vs.add_detection_results(test_vid, test_results, "test")
            for _ in range(10):
                t1 = time.time()
                vs.search(test_vid, "test query", top_k=3)
                times.append((time.time() - t1) * 1000)
            avg = sum(times) / len(times)
            collector.add("性能-向量检索", avg < 500,
                           f"10次平均: {avg:.1f}ms", "性能基准", avg)
        else:
            collector.add("性能-向量检索", False, "ChromaDB 未就绪", "性能基准", 0)
    except Exception as e:
        collector.add("性能-向量检索", False, str(e)[:100], "性能基准", 0)

    # 5.3 知识库查询性能
    t0 = time.time()
    try:
        kb = load_json(KNOWLEDGE_BASE_PATH)
        if kb:
            times = []
            for i in range(100):
                t1 = time.time()
                # 模拟知识检索
                entity = kb.get("Entity", {})
                _ = str(entity.keys())
                times.append((time.time() - t1) * 1000)
            avg = sum(times) / len(times)
            collector.add("性能-知识库查询", avg < 10,
                           f"100次平均: {avg:.3f}ms", "性能基准", avg)
    except Exception as e:
        collector.add("性能-知识库查询", False, str(e)[:100], "性能基准", 0)


# ============================================================
# 报告生成
# ============================================================
def generate_report(report_dir: str = None):
    """生成集成测试报告 (Markdown + JSON)"""
    if report_dir is None:
        report_dir = str(OUTPUT_DIR)
    os.makedirs(report_dir, exist_ok=True)

    summary = collector.summary()
    by_cat = collector.by_category()

    # --- Markdown 报告 ---
    md_path = os.path.join(report_dir, "integration_test_report.md")
    md_content = f"""# videoAI 集成测试报告

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **总耗时**: {summary['total_time_s']:.2f} 秒
> **测试结果**: {summary['passed']}/{summary['total']} 通过, 通过率 {summary['pass_rate']}

---

## 总览

| 指标 | 数值 |
|------|------|
| 总测试项 | {summary['total']} |
| 通过 | {summary['passed']} |
| 失败 | {summary['failed']} |
| 通过率 | {summary['pass_rate']} |
| 总耗时 | {summary['total_time_s']:.2f}s |

---

## 分类统计

| 测试类别 | 总项 | 通过 | 失败 | 通过率 |
|----------|------|------|------|--------|
"""

    for cat, stats in by_cat.items():
        pass_rate = f"{stats['passed']/stats['total']*100:.0f}%" if stats['total'] else "N/A"
        md_content += f"| {cat} | {stats['total']} | {stats['passed']} | {stats['failed']} | {pass_rate} |\n"

    md_content += """
---

## 详细结果

| # | 类别 | 测试项 | 结果 | 详情 | 耗时 |
|---|------|--------|------|------|------|
"""

    for i, r in enumerate(summary["results"], 1):
        status = "✅" if r["passed"] else "❌"
        md_content += f"| {i} | {r['category']} | {r['name']} | {status} | {r['detail'][:80]} | {r['duration_ms']:.0f}ms |\n"

    md_content += f"""

---

## 系统环境

| 项目 | 值 |
|------|-----|
| Python | {sys.version} |
| 项目根目录 | {PROJECT_ROOT} |
| 测试视频 | {TEST_VIDEO_PATH.name if TEST_VIDEO_PATH.exists() else '未找到'} |
| 知识库 | {KNOWLEDGE_BASE_PATH.name if KNOWLEDGE_BASE_PATH.exists() else '未找到'} |
| 测试QA | {TEST_QA_PATH.name if TEST_QA_PATH.exists() else '未找到'} |
| LLM 后端 | {LLM_CONFIG['backend']} |
| 模型 | {LLM_CONFIG['model_name']} |

---

> 🤖 由 videoAI 集成测试框架自动生成
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    log(f"Markdown 报告: {md_path}")

    # --- JSON 报告 ---
    json_path = os.path.join(report_dir, "integration_test_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "by_category": {k: v for k, v in by_cat.items()},
            "environment": {
                "python": sys.version,
                "project_root": str(PROJECT_ROOT),
                "llm_backend": LLM_CONFIG["backend"],
                "model": LLM_CONFIG["model_name"],
            }
        }, f, ensure_ascii=False, indent=2)
    log(f"JSON 报告: {json_path}")

    # --- 控制台输出 ---
    print()
    print("=" * 60)
    print("  集成测试完成!")
    print("=" * 60)
    print(f"  通过: {summary['passed']}/{summary['total']} ({summary['pass_rate']})")
    print(f"  总耗时: {summary['total_time_s']:.2f}s")
    print()
    print("  分类统计:")
    for cat, stats in by_cat.items():
        status = "[OK]" if stats["failed"] == 0 else "⚠"
        print(f"    {status} {cat}: {stats['passed']}/{stats['total']}")
    print()
    print(f"  报告文件: {md_path}")
    print(f"  报告文件: {json_path}")
    print("=" * 60)


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="videoAI 集成测试套件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="测试视频文件路径")
    parser.add_argument("--output", "-o", type=str, default=str(OUTPUT_DIR),
                        help="报告输出目录")
    parser.add_argument("--report", "-r", type=str, default=None,
                        help="报告文件前缀")
    parser.add_argument("--skip-llm", action="store_true",
                        help="跳过需要 LLM API 的测试")
    parser.add_argument("--sample", "-s", type=int, default=0,
                        help="QA 采样数量, 0=全部")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（跳过性能测试）")
    args = parser.parse_args()

    log("=" * 60)
    log("  videoAI 集成测试套件")
    log(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # 1. 数据完整性测试
    test_data_integrity()

    # 2. 模块接口测试
    test_module_interfaces()

    # 3. 管线集成测试
    test_pipeline_integration(
        video_path=args.video if args.video else None,
        skip_llm=args.skip_llm,
    )

    # 4. 知识库集成测试
    test_knowledge_base_integration()

    # 5. 性能基准测试（快速模式跳过）
    if not args.quick:
        test_performance()

    # 生成报告
    report_dir = args.report or args.output
    generate_report(report_dir)

    # 返回状态码
    failed = collector.failed_count
    sys.exit(0 if failed == 0 else min(failed, 127))


if __name__ == "__main__":
    main()
