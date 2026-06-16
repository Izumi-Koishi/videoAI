"""
Mock 模块
模拟队友的 YOLO 检测器、CLIP 特征提取器和向量数据库检索器。
用于独立演示和测试，待队友模块完成后替换为真实实现。
"""

import os
import random
from typing import List
from dataclasses import dataclass
from PIL import Image, ImageDraw


@dataclass
class DetectionResult:
    """检测结果"""
    frame_id: int
    class_name: str
    confidence: float
    bbox: tuple
    crop_path: str


@dataclass
class SearchResult:
    """向量检索结果"""
    target_id: int
    class_name: str
    confidence: float
    similarity: float
    description: str
    image_path: str
    bbox: tuple


# ================================================================
# Mock YOLO 检测器
# ================================================================

class MockYOLODetector:
    """模拟 YOLO 检测器，接口与真实检测器一致"""

    MOCK_CLASSES = [
        "行人", "汽车", "自行车", "摩托车", "公交车",
        "卡车", "交通灯", "停车标志", "长椅", "背包",
        "手提包", "手机", "笔记本电脑", "书本", "瓶子",
        "杯子", "椅子", "桌子", "显示器", "键盘",
    ]

    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        print(f"[Mock] YOLO检测器已初始化")

    def detect_video(self, video_path: str,
                     sample_interval: int = 30) -> List[DetectionResult]:
        """模拟视频目标检测"""
        print(f"[Mock] 正在检测视频: {video_path}")

        total_frames = random.randint(60, 300)
        results = []

        for frame_id in range(1, total_frames + 1, sample_interval):
            n_objects = random.randint(1, 5)
            for _ in range(n_objects):
                cls = random.choice(self.MOCK_CLASSES)
                x1 = random.randint(0, 500)
                y1 = random.randint(0, 300)
                w = random.randint(50, 200)
                h = random.randint(50, 200)

                results.append(DetectionResult(
                    frame_id=frame_id,
                    class_name=cls,
                    confidence=round(random.uniform(0.65, 0.98), 4),
                    bbox=(x1, y1, x1 + w, y1 + h),
                    crop_path=f"./data/outputs/crop_{frame_id}_{cls}.jpg",
                ))

        print(f"[Mock] 检测到 {len(results)} 个目标")

        # 生成模拟裁剪图
        self._generate_mock_crops(results)
        return results

    def _generate_mock_crops(self, results: List[DetectionResult]):
        """生成模拟裁剪图像"""
        os.makedirs("./data/outputs", exist_ok=True)
        for r in results[:20]:
            img = Image.new('RGB', (100, 100), color=(
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200),
            ))
            draw = ImageDraw.Draw(img)
            draw.text((10, 40), r.class_name, fill='white')
            path = os.path.join(
                "./data/outputs",
                f"crop_{r.frame_id}_{r.class_name}.jpg"
            )
            img.save(path)
            r.crop_path = path


# ================================================================
# Mock CLIP + 向量数据库检索器
# ================================================================

class MockCLIPRetriever:
    """模拟 CLIP 特征提取 + 向量数据库检索"""

    KNOWLEDGE_BASE = {
        "行人": [
            "成年人在人行道上行走",
            "穿深色衣服的行人正在过马路",
            "学生背着书包走在校园道路上",
            "行人正在使用手机",
        ],
        "汽车": [
            "红色轿车停在路边停车位",
            "白色SUV在道路上行驶",
            "黑色轿车等红灯",
            "银色轿车正在转弯",
        ],
        "自行车": [
            "共享单车停放在指定区域",
            "骑自行车的人戴着头盔",
            "自行车在非机动车道行驶",
        ],
        "交通灯": [
            "红灯亮起，车辆停止等待",
            "绿灯亮起，车辆开始通行",
            "黄灯闪烁，提醒减速",
        ],
        "公交车": [
            "绿色公交车停靠在站台",
            "公交车正在上下乘客",
            "校车停在学校门口",
        ],
        "长椅": [
            "公园里的木质长椅",
            "有人坐在长椅上休息",
            "校园路边的铁质长椅",
        ],
        "背包": [
            "黑色双肩背包",
            "学生背着书包",
            "旅行背包放在地上",
        ],
        "笔记本电脑": [
            "银色笔记本电脑在桌面上",
            "学生使用笔记本电脑学习",
            "笔记本电脑连接着外接显示器",
        ],
        "手机": [
            "智能手机在手中",
            "路人正在低头看手机",
            "手机放在桌面上充电",
        ],
        "书本": [
            "桌上放着打开的教科书",
            "学生在图书馆看书",
            "书架上整齐排列的书籍",
        ],
    }

    def __init__(self, db_path: str = ""):
        self.db_path = db_path
        self._detections: List[DetectionResult] = []
        print(f"[Mock] CLIP检索器已初始化")

    def set_detections(self, detections: List[DetectionResult]):
        """设置检测结果，用于建立检索索引"""
        self._detections = detections
        print(f"[Mock] 索引已建立: {len(detections)} 个目标")

    def search(self, query_text: str, top_k: int = 5) -> List[SearchResult]:
        """基于文本查询检索相关目标"""
        print(f"[Mock] 检索: '{query_text[:50]}...'")

        if not self._detections:
            return self._mock_search(query_text, top_k)

        results = []
        for i, det in enumerate(self._detections[:top_k]):
            sim = round(random.uniform(0.45, 0.95), 4)
            desc = self._get_description(det.class_name)
            results.append(SearchResult(
                target_id=i + 1,
                class_name=det.class_name,
                confidence=det.confidence,
                similarity=sim,
                description=desc,
                image_path=det.crop_path,
                bbox=det.bbox,
            ))

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def _mock_search(self, query_text: str,
                     top_k: int = 5) -> List[SearchResult]:
        """使用内置知识库生成模拟检索结果"""
        matched_classes = []
        for cls in self.KNOWLEDGE_BASE:
            if cls in query_text or any(w in query_text for w in cls):
                matched_classes.append(cls)

        if not matched_classes:
            matched_classes = random.sample(
                list(self.KNOWLEDGE_BASE.keys()),
                min(3, len(self.KNOWLEDGE_BASE))
            )

        results = []
        for i, cls in enumerate(matched_classes[:top_k]):
            desc = random.choice(self.KNOWLEDGE_BASE[cls])
            results.append(SearchResult(
                target_id=i + 1,
                class_name=cls,
                confidence=round(random.uniform(0.70, 0.95), 4),
                similarity=round(random.uniform(0.65, 0.92), 4),
                description=desc,
                image_path=self._create_mock_image(cls, i),
                bbox=(random.randint(0, 300), random.randint(0, 200),
                      random.randint(350, 600), random.randint(250, 450)),
            ))

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def _get_description(self, class_name: str) -> str:
        if class_name in self.KNOWLEDGE_BASE:
            return random.choice(self.KNOWLEDGE_BASE[class_name])
        return f"视频中检测到的{class_name}"

    def _create_mock_image(self, class_name: str, idx: int) -> str:
        os.makedirs("./data/outputs", exist_ok=True)
        path = f"./data/outputs/mock_target_{idx}_{class_name}.jpg"
        img = Image.new('RGB', (200, 200), color=(
            random.randint(30, 180),
            random.randint(30, 180),
            random.randint(30, 180),
        ))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 180, 180], outline='white', width=3)
        draw.text((50, 90), class_name, fill='white')
        img.save(path)
        return path

    def get_search_results_as_contexts(self,
                                       results: List[SearchResult]):
        """将检索结果转换为 PromptBuilder 可用的上下文格式"""
        from prompts.templates import RetrievedContext
        contexts = []
        for r in results:
            contexts.append(RetrievedContext(
                target_id=r.target_id,
                class_name=r.class_name,
                confidence=r.confidence,
                similarity=r.similarity,
                description=r.description,
                image_path=r.image_path,
            ))
        return contexts
