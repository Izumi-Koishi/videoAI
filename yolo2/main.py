import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.yolo_detector import YoloDetector
from src.config import VIDEO_DIR, IMAGE_DIR, OUTPUT_DIR

def main():
    print("="*50)
    print("    YOLO目标检测组 - 任务二实现")
    print("="*50)
    print("\n功能:")
    print("  1. 检测目标自动裁剪")
    print("  2. 按规范命名生成目标图像块")
    print("  3. 构建本地目标图像库")
    print("  4. 整理检测结果元数据文件")
    
    print(f"\n视频目录: {VIDEO_DIR}")
    print(f"图像库目录: {IMAGE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    print("\n初始化YOLO检测器...")
    detector = YoloDetector()
    
    print("\n开始处理视频文件...")
    total_detected = detector.process_all_videos()
    
    if total_detected == 0:
        print("\n提示: 请将视频文件放入 data/videos 目录后重新运行")
        return
    
    print("\n" + "="*50)
    print("任务二完成!")
    print("="*50)
    print(f"目标图像库: {IMAGE_DIR}")
    print(f"元数据文件: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()