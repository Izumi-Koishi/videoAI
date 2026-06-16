import os
import cv2
import base64
import numpy as np
from pathlib import Path
from .config import OUTPUT_DIR

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (192, 192, 192), (128, 128, 128),
]


def get_color(class_id, colors=None):
    if colors is None:
        colors = COLORS
    return colors[class_id % len(colors)]


def draw_detections(frame, detections, show_conf=True):
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['class_name']
        cls_id = det.get('class_id', 0)
        conf = det.get('confidence', 0)
        color = get_color(cls_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if show_conf:
            label = f"{cls_name} {conf:.2f}"
        else:
            label = cls_name

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def numpy_to_pil(frame):
    from PIL import Image
    return Image.fromarray(frame[..., ::-1])


def create_annotated_video(video_path, detections_by_frame, output_path=None, fps=30):
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_annotated.mp4")

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps > 0:
        fps = orig_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    det_map = {}
    for fr in detections_by_frame:
        det_map[fr['frame_index']] = fr['detections']

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in det_map:
            frame = draw_detections(frame, det_map[frame_idx])

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Annotated video saved to: {output_path}")
    return output_path


def visualize_detection_result(frame, detections):
    annotated = draw_detections(frame, detections)
    b64 = frame_to_base64(annotated)
    return annotated, b64


def create_detection_summary(detection_result, top_n=10):
    video_id = detection_result.get('video_id', 'unknown')
    total_dets = detection_result.get('total_detections', 0)
    total_frames = detection_result.get('total_frames_processed', 0)

    class_counts = {}
    for fr in detection_result.get('frame_results', []):
        for det in fr['detections']:
            cls = det['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    summary = f"## Detection Summary: {video_id}\n\n"
    summary += f"- Total frames processed: {total_frames}\n"
    summary += f"- Total detections: {total_dets}\n"
    summary += f"- Unique classes detected: {len(class_counts)}\n\n"
    summary += "### Top Detected Classes\n\n"

    for cls_name, count in sorted_classes[:top_n]:
        summary += f"- **{cls_name}**: {count}\n"

    if len(sorted_classes) > top_n:
        summary += f"\n... and {len(sorted_classes) - top_n} more classes\n"

    return summary
