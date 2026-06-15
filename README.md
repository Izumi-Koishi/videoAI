# YOLO目标检测组 - 任务二实现

## 项目简介

本项目是视频AI模型项目中YOLO目标检测组的**任务二**实现，专注于检测目标自动裁剪和图像库构建。

## 任务二需求

| 序号 | 功能 | 状态 |
|------|------|------|
| 1 | 实现检测目标自动裁剪功能 | ✅ |
| 2 | 按规范命名生成目标图像块 | ✅ |
| 3 | 构建本地目标图像库 | ✅ |
| 4 | 整理检测结果元数据文件 | ✅ |

## 项目结构

```
├── main.py                    # 项目主入口
├── requirements.txt           # Python依赖列表
├── README.md                  # 项目说明文档
├── data/
│   ├── videos/               # 输入视频文件目录
│   ├── images/               # 目标图像库（按类别分类）
│   └── output/               # 检测结果元数据输出目录
├── models/                   # YOLO模型存储目录
└── src/
    ├── __init__.py           # 包初始化
    ├── config.py             # 配置文件
    └── yolo_detector.py      # 核心检测模块
```

## 安装依赖

```bash
pip install ultralytics pillow numpy
```

## 使用方法

### 1. 准备视频文件

将视频放入 `data/videos/` 目录，支持格式：MP4、AVI、MOV、MKV、FLV

### 2. 运行主程序

```bash
python main.py
```

## 交付物

### 目标图像库
- 位置：`data/images/{类别名称}/`
- 命名格式：`{类别}_{视频ID}_frame{帧号}_{UUID}_{时间戳}.jpg`

### 检测结果元数据
- 位置：`data/output/detection_metadata_*.json`
- 字段：image_id、image_path、class_name、confidence、bbox

## 配置参数

`src/config.py`:
```python
CONFIDENCE_THRESHOLD = 0.5    # 置信度阈值
IOU_THRESHOLD = 0.45          # IOU阈值
DEFAULT_FRAME_SKIP = 5        # 帧跳步
```

## 代码示例

```python
from src.yolo_detector import YoloDetector

detector = YoloDetector()
detector.process_all_videos()  # 处理所有视频并自动保存元数据
```

## 技术栈

- Python 3.12+
- Ultralytics YOLOv8
- PIL (Pillow)
- NumPy

---
