"""UDP inference server — telefon kamerasından gelen JPEG kareleri alır,
YOLO çalıştırır ve tespitleri JSON olarak geri gönderir."""

import socket
import struct
import json
import time

import cv2
import numpy as np
from ultralytics import YOLO

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9999
BUFFER_SIZE = 131072  # 128 KB — 320×320 JPEG için fazlasıyla yeterli

def load_model(weights: str = "yolo26n.pt") -> YOLO:
    print(f"Model yükleniyor: {weights}")
    model = YOLO(weights)
    # Warm-up
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False, imgsz=640)
    print("Model hazır.")
    return model

def run_server():
    model = load_model()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_HOST, LISTEN_PORT))
    print(f"Sunucu dinleniyor: {LISTEN_HOST}:{LISTEN_PORT}")

    frame_count = 0
    fps_start = time.time()

    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
        except Exception as e:
            print(f"Alma hatası: {e}")
            continue

        if len(data) < 4:
            continue

        frame_id = struct.unpack(">I", data[:4])[0]
        jpg_bytes = data[4:]

        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        t0 = time.perf_counter()
        results = model.predict(img, conf=0.35, iou=0.45, verbose=False, imgsz=640)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxyn = boxes.xyxyn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            names = results[0].names
            for (x1, y1, x2, y2), conf, cls_id in zip(xyxyn, confs, cls_ids):
                detections.append({
                    "class_id": int(cls_id),
                    "class_name": names[cls_id],
                    "confidence": round(float(conf), 3),
                    "x1": round(float(x1), 4),
                    "y1": round(float(y1), 4),
                    "x2": round(float(x2), 4),
                    "y2": round(float(y2), 4),
                })

        response = struct.pack(">I", frame_id) + json.dumps(detections).encode()
        sock.sendto(response, addr)

        frame_count += 1
        now = time.time()
        if now - fps_start >= 2.0:
            fps = frame_count / (now - fps_start)
            print(f"FPS: {fps:.1f} | Inference: {elapsed_ms:.0f}ms | Tespitler: {len(detections)}")
            frame_count = 0
            fps_start = now

if __name__ == "__main__":
    run_server()
