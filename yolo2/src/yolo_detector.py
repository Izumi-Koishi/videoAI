import os
import json
import uuid
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from .config import (
    YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    VIDEO_DIR, IMAGE_DIR, OUTPUT_DIR
)

class YoloDetector:
    def __init__(self):
        self.model = None
        self.detection_metadata = []
        self.load_model()

    def load_model(self):
        if not os.path.exists(YOLO_MODEL_PATH):
            print("Downloading YOLOv8n model...")
            self.model = YOLO('yolov8n.pt')
            self.model.save(YOLO_MODEL_PATH)
        else:
            self.model = YOLO(YOLO_MODEL_PATH)
        print("YOLOv8 model loaded successfully")

    def process_video(self, video_path, frame_skip=5):
        video_name = os.path.basename(video_path)
        video_id = os.path.splitext(video_name)[0]
        
        print(f"Processing video: {video_name}")
        
        frame_count = 0
        detection_count = 0
        
        for result in self.model.track(video_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, stream=True):
            if frame_count % frame_skip == 0:
                frame = result.orig_img
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = self.model.names[cls]
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': label
                    }
                    
                    self._process_detection(frame, detection, video_id, frame_count)
                    detection_count += 1
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames, {detection_count} detections")
        
        print(f"Processing complete. Total detections: {detection_count}")
        return detection_count

    def _process_detection(self, frame, detection, video_id, frame_index):
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        frame_pil = Image.fromarray(frame[..., ::-1])
        target_img = frame_pil.crop((x1, y1, x2, y2))
        
        target_uuid = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        class_dir = os.path.join(IMAGE_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        image_filename = f"{class_name}_{video_id}_frame{frame_index}_{target_uuid}_{timestamp}.jpg"
        image_path = os.path.join(class_dir, image_filename)
        
        target_img.save(image_path)
        
        metadata = {
            'image_id': target_uuid,
            'image_path': image_path,
            'class_name': class_name,
            'confidence': confidence,
            'bbox': detection['bbox']
        }
        
        self.detection_metadata.append(metadata)

    def save_metadata(self, output_filename=None):
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"detection_metadata_{timestamp}.json"
        
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.detection_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Metadata saved to: {output_path}")
        return output_path

    def process_all_videos(self, frame_skip=5):
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
        videos_found = []
        
        for file in os.listdir(VIDEO_DIR):
            if file.lower().endswith(video_extensions):
                videos_found.append(file)
        
        if not videos_found:
            print("未找到视频文件")
            return 0
        
        total_detections = 0
        for video in videos_found:
            video_path = os.path.join(VIDEO_DIR, video)
            detections = self.process_video(video_path, frame_skip)
            total_detections += detections
        
        self.save_metadata()
        return total_detections

def main():
    detector = YoloDetector()
    
    print("\n=== YOLO目标检测组 - 任务二 ===")
    print("功能: 检测目标自动裁剪、图像库构建、元数据管理")
    print("="*50)
    
    detector.process_all_videos()
    
    print("\n目标图像库已构建完成，元数据文件已保存")

if __name__ == "__main__":
    main()
