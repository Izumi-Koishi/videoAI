import os
import json
from datetime import datetime
from ultralytics import YOLO
from .config import (
    YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    VIDEO_DIR, OUTPUT_DIR
)


class Detector:
    def __init__(self, model_path=None):
        self.model_path = model_path or YOLO_MODEL_PATH
        self.model = None
        self.results = []
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print("Downloading YOLOv8n model...")
            self.model = YOLO('yolov8n.pt')
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
        else:
            self.model = YOLO(self.model_path)
        print("YOLOv8n model loaded successfully")

    def detect_frame(self, frame):
        detections = []
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': round(conf, 4),
                    'class_id': cls_id,
                    'class_name': cls_name
                })

        return detections

    def process_video(self, video_path, frame_skip=None):
        if frame_skip is None:
            from .config import DEFAULT_FRAME_SKIP
            frame_skip = DEFAULT_FRAME_SKIP

        video_name = os.path.basename(video_path)
        video_id = os.path.splitext(video_name)[0]
        print(f"Processing video: {video_name}")

        frame_results = []
        frame_count = 0

        for result in self.model.track(
            video_path,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            stream=True,
            verbose=False
        ):
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            boxes = result.boxes
            if boxes is None:
                frame_count += 1
                continue

            detections = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': round(conf, 4),
                    'class_id': cls_id,
                    'class_name': cls_name
                })

            frame_info = {
                'frame_index': frame_count,
                'video_id': video_id,
                'detections': detections
            }
            frame_results.append(frame_info)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames")

        total_detections = sum(len(fr['detections']) for fr in frame_results)
        print(f"Finished: {len(frame_results)} frames, {total_detections} detections")

        video_result = {
            'video_id': video_id,
            'video_path': video_path,
            'total_frames_processed': len(frame_results),
            'total_detections': total_detections,
            'frame_results': frame_results
        }

        self.results.append(video_result)
        return video_result

    def process_all_videos(self, frame_skip=None):
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')
        videos_found = [
            f for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith(video_extensions)
        ]

        if not videos_found:
            print("No video files found in", VIDEO_DIR)
            return []

        all_results = []
        for video in videos_found:
            video_path = os.path.join(VIDEO_DIR, video)
            result = self.process_video(video_path, frame_skip)
            all_results.append(result)

        self.save_results()
        return all_results

    def save_results(self, output_filename=None):
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"detection_results_{timestamp}.json"

        output_path = os.path.join(OUTPUT_DIR, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"Detection results saved to: {output_path}")
        return output_path

    def get_detections_flat(self):
        flat = []
        for vr in self.results:
            for fr in vr['frame_results']:
                for det in fr['detections']:
                    flat.append({
                        'video_id': vr['video_id'],
                        'frame_index': fr['frame_index'],
                        **det
                    })
        return flat
