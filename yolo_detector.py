"""
YOLO 检测模块 - 负责视频帧的目标检测
"""
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional


class YOLODetector:
    """YOLO 目标检测器，基于 ultralytics"""

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        初始化 YOLO 检测器
        Args:
            model_name: 模型名称，默认使用 yolov8n（nano 版本，速度快）
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"[YOLO] 模型 {self.model_name} 加载成功")
        except Exception as e:
            print(f"[YOLO] 模型加载失败: {e}")
            self.model = None

    def is_ready(self) -> bool:
        """检查模型是否就绪"""
        return self.model is not None

    def detect_image(self, image: np.ndarray, conf: float = 0.25) -> Dict:
        """
        对单张图片进行目标检测
        Args:
            image: numpy 数组格式的图片 (BGR)
            conf: 置信度阈值
        Returns:
            包含检测结果的字典
        """
        if not self.is_ready():
            return {"boxes": [], "labels": [], "scores": [], "annotated": image}

        results = self.model(image, conf=conf, verbose=False)
        result = results[0]

        boxes = []
        labels = []
        scores = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = result.names[cls_id]

            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            scores.append(score)

        # 生成标注图
        annotated = result.plot()

        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "annotated": annotated
        }

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        frame_interval: int = 30,
        conf: float = 0.25
    ) -> Tuple[List[Dict], List[str], str]:
        """
        处理视频文件，按帧间隔进行检测
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            frame_interval: 每隔多少帧检测一次
            conf: 置信度阈值
        Returns:
            (detection_results, saved_image_paths, detection_video_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # 默认帧率，防止 VideoWriter 用 0 fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detection_results = []
        saved_image_paths = []
        frame_count = 0
        detection_count = 0

        # 用于保存检测可视化视频
        video_name = Path(video_path).stem
        detection_video_path = os.path.join(output_dir, f"{video_name}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(detection_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                result = self.detect_image(frame, conf=conf)
                result["frame_index"] = frame_count
                result["timestamp"] = frame_count / fps if fps > 0 else 0

                # 保存标注图片
                img_filename = f"frame_{detection_count:04d}.jpg"
                img_path = os.path.join(output_dir, img_filename)
                cv2.imwrite(img_path, result["annotated"])
                saved_image_paths.append(img_path)

                # 保存原始裁剪目标
                result["saved_image"] = img_path
                result["cropped_objects"] = self._crop_objects(
                    frame, result["boxes"], result["labels"], output_dir, detection_count
                )

                detection_results.append(result)
                detection_count += 1

                # 写入检测视频
                writer.write(result["annotated"])
            else:
                # 非检测帧：直接写入原始帧，避免对每帧跑YOLO导致极慢
                writer.write(frame)

            frame_count += 1

        cap.release()
        writer.release()

        print(f"[YOLO] 视频处理完成: {frame_count} 帧, {detection_count} 次检测")
        return detection_results, saved_image_paths, detection_video_path

    def _crop_objects(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
        labels: List[str],
        output_dir: str,
        frame_idx: int
    ) -> List[Dict]:
        """
        裁剪检测到的目标对象
        """
        cropped = []
        crop_dir = os.path.join(output_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)

        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = [int(v) for v in box]
            # 确保坐标在图片范围内
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_filename = f"crop_{frame_idx:04d}_{i}_{label}.jpg"
            crop_path = os.path.join(crop_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            cropped.append({"label": label, "image_path": crop_path, "bbox": box})

        return cropped

    def generate_detection_summary(self, detection_results: List[Dict]) -> str:
        """
        根据检测结果生成文本摘要，用于向量检索
        """
        summaries = []
        for result in detection_results:
            frame_idx = result.get("frame_index", 0)
            timestamp = result.get("timestamp", 0)
            labels = result.get("labels", [])
            scores = result.get("scores", [])

            if not labels:
                continue

            # 统计每个类别的出现次数
            label_count = {}
            for label, score in zip(labels, scores):
                if label not in label_count:
                    label_count[label] = {"count": 0, "max_score": 0}
                label_count[label]["count"] += 1
                label_count[label]["max_score"] = max(
                    label_count[label]["max_score"], score
                )

            # 生成描述文本
            objects_desc = ", ".join(
                [f"{label}({info['count']}个, 最高置信度{info['max_score']:.2f})"
                 for label, info in label_count.items()]
            )
            summary = f"时间戳{timestamp:.1f}秒(第{frame_idx}帧): 检测到 {objects_desc}"
            summaries.append(summary)

        return "\n".join(summaries)
