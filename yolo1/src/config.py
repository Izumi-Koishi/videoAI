import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'yolov8n.pt')

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

DEFAULT_FRAME_SKIP = 5
