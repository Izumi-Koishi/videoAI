import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detector import Detector
from src.visualizer import (
    create_annotated_video,
    create_detection_summary,
    draw_detections,
    frame_to_base64,
)
from src.config import VIDEO_DIR


def main():
    print("=" * 50)
    print("    YOLO目标检测组 - 任务一实现")
    print("=" * 50)
    print("\n功能:")
    print("  1. 部署YOLOv8预训练模型，完成环境适配")
    print("  2. 开发视频批量处理脚本，逐帧检测目标")
    print("  3. 输出边界框坐标、类别标签、置信度")
    print("  4. 封装检测结果可视化函数，供前端调用")
    print(f"\n视频目录: {VIDEO_DIR}")

    print("\n初始化YOLO检测器...")
    detector = Detector()

    print("\n开始处理视频文件...")
    results = detector.process_all_videos()

    if not results:
        print("\n提示: 请将视频文件放入 data/videos 目录后重新运行")
        return

    for result in results:
        print("\n" + create_detection_summary(result))

    print("\n" + "=" * 50)
    print("任务一完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
