# The code directory I will probably never look at again. YOLOv8 and MiDaS studied the dynamics of their models.

import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

def load_models(device):
    yolo = YOLO("yolov8m.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    return yolo, midas, transform

def get_depth(frame_rgb, midas, transform, device):
    input_tensor = transform(frame_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    norm = cv2.normalize(prediction.cpu().numpy(), None, 0, 1, cv2.NORM_MINMAX)
    color_map = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return norm, color_map

def process_boxes(boxes, names, timers, depth_map, marked_frame):
    marked = []
    current_ids = set()
    for box in boxes:
        if names[int(box.cls[0])] != "person":
            continue
        if box.id is None:
            continue
        track_id = int(box.id[0].item())
        current_ids.add(track_id)

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        center = (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))

        if track_id not in timers:
            timers[track_id] = time.time()
            print(f"Kişi {track_id} sahneye girdi.")

        duration = time.time() - timers[track_id]

        radius = 50
        mask = np.zeros_like(depth_map, dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, -1)
        masked_depths = depth_map[mask == 1]
        if masked_depths.size > 0:
            depth = np.nanmean(masked_depths)
        else:
            depth = -1

        cv2.circle(marked_frame, center, radius, (255, 255, 0), 1)

        marked.append((track_id, center, duration, depth))
    return marked, current_ids

def process_boxes_photo(boxes, names, depth_map, marked_frame):
    marked = []
    for box in boxes:
        if names[int(box.cls[0])] != "person":
            continue
        if box.id is None:
            track_id = 0  
        else:
            track_id = int(box.id[0].item())

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        center = (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))

        # Derinlik
        radius = 50
        mask = np.zeros_like(depth_map, dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, -1)
        masked_depths = depth_map[mask == 1]
        depth = np.nanmean(masked_depths) if masked_depths.size > 0 else -1

        cv2.circle(marked_frame, center, radius, (255, 255, 0), 1)
        marked.append((track_id, center, depth))
    return marked

def video_cap():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo, midas, transform = load_models(device)
    cap = cv2.VideoCapture(0)
    timers = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_norm, depth_color = get_depth(rgb, midas, transform, device)

        results = yolo.track(frame, persist=True, conf=0.5, device=device, tracker="ultralytics/cfg/trackers/bytetrack.yaml")[0]
        boxes = results.boxes if results.boxes is not None else []
        marked_frame = frame.copy()

        marked_data, current_ids = process_boxes(boxes, yolo.names, timers, depth_norm, marked_frame)

        for tid, center, duration, depth in marked_data:
            cv2.circle(marked_frame, center, 5, (132, 112, 255), -1)
            cv2.putText(marked_frame, f"ID:{tid} T:{duration:.1f}s D:{depth:.2f}",
                        (center[0] - 40, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 0, 211), 2)

        for lost_id in set(timers) - current_ids:
            del timers[lost_id]

        fps = 1 / (time.time() - start)
        cv2.putText(marked_frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 - Multi-Person Timer", marked_frame)
        cv2.imshow("Depth Map", depth_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def photo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo, midas, transform = load_models(device)

    image_path = "2person.png"  
    frame = cv2.imread(image_path)
    if frame is None:
        print("Görsel yüklenemedi.")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_norm, depth_color = get_depth(rgb, midas, transform, device)

    results = yolo.track(frame, persist=True, conf=0.5, device=device, tracker="ultralytics/cfg/trackers/bytetrack.yaml")[0]
    boxes = results.boxes if results.boxes is not None else []
    marked_frame = frame.copy()

    marked_data = process_boxes_photo(boxes, yolo.names, depth_norm, marked_frame)

    for tid, center, depth in marked_data:
        cv2.circle(marked_frame, center, 10, (132, 112, 255), -1)
        cv2.putText(marked_frame, f"ID:{tid} D:{depth:.2f}", (center[0] - 40, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (148, 0, 211), 5)

    resized_marked = cv2.resize(marked_frame, None, fx=0.2, fy=0.2)
    resized_depth = cv2.resize(depth_color, None, fx=0.2, fy=0.2)

    cv2.imshow("YOLOv8 - Image", resized_marked)
    cv2.imshow("Depth Map", resized_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# video_cap()
photo()