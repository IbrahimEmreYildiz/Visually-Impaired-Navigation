"""Kamera odak uzunluğu kalibrasyon aracı.

Kullanım:
    python -m visually_impaired_navigation.src.calibrate_focal

Adımlar:
    1. Kameranın önüne bilinen yükseklikte bir nesne koyun (örn. 30 cm'lik kitap).
    2. Nesneyi tam olarak bilinen bir mesafeye yerleştirin (örn. 1.00 metre).
    3. Programı başlatın, nesne etrafına çerçeve çizin, 'c' tuşuna basın.
    4. Hesaplanan focal_length değerini config.py'deki DistanceConfig'e yazın.
"""

from __future__ import annotations

import sys
import cv2
import numpy as np

# --- Kalibrasyon parametreleri ---
KNOWN_REAL_HEIGHT_M = 0.30   # Nesnenin gerçek yüksekliği (metre)
KNOWN_DISTANCE_M    = 1.00   # Kameradan nesneye gerçek mesafe (metre)
# ---------------------------------


def run_calibration() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı.")
        sys.exit(1)

    roi: list = []
    drawing = False
    ix, iy = -1, -1
    frame_snapshot = None

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, ix, iy, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            roi.clear()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            roi[:] = [ix, iy, x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi[:] = [ix, iy, x, y]

    cv2.namedWindow("Kalibrasyon")
    cv2.setMouseCallback("Kalibrasyon", mouse_cb)

    print("=" * 55)
    print("KAMERA KALİBRASYONU")
    print(f"  Nesne yüksekliği : {KNOWN_REAL_HEIGHT_M*100:.0f} cm")
    print(f"  Nesme mesafesi   : {KNOWN_DISTANCE_M*100:.0f} cm")
    print("Nesne etrafını fareyle seçin, ardından 'c' tuşuna basın.")
    print("Çıkmak için 'q' tuşuna basın.")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if len(roi) == 4:
            x1, y1, x2, y2 = roi
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            pixel_height = abs(y2 - y1)
            if pixel_height > 0:
                fl = (KNOWN_REAL_HEIGHT_M * pixel_height) / KNOWN_DISTANCE_M * (1 / KNOWN_REAL_HEIGHT_M) * KNOWN_DISTANCE_M / KNOWN_REAL_HEIGHT_M
                # Doğru formül: fl = (pixel_height * KNOWN_DISTANCE_M) / KNOWN_REAL_HEIGHT_M
                fl_correct = (pixel_height * KNOWN_DISTANCE_M) / KNOWN_REAL_HEIGHT_M
                cv2.putText(display, f"Piksel yuksekligi: {pixel_height}px",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Tahmini focal: {fl_correct:.1f}px",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(display, "Nesneyi sec, 'c' ile kaydet, 'q' cikis",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Kalibrasyon", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and len(roi) == 4:
            x1, y1, x2, y2 = roi
            pixel_height = abs(y2 - y1)
            if pixel_height <= 0:
                print("Geçersiz seçim, tekrar deneyin.")
                continue
            focal_length = (pixel_height * KNOWN_DISTANCE_M) / KNOWN_REAL_HEIGHT_M
            print("\n" + "=" * 55)
            print(f"KALİBRASYON SONUCU")
            print(f"  Piksel yüksekliği : {pixel_height} px")
            print(f"  Hesaplanan focal  : {focal_length:.2f} px")
            print()
            print("config.py dosyasında şu satırı güncelleyin:")
            print(f"  focal_length_pixels: float = {focal_length:.2f}")
            print("=" * 55)
            frame_snapshot = frame.copy()
            cv2.imwrite("calibration_snapshot.jpg", frame_snapshot)
            print("Anlık görüntü 'calibration_snapshot.jpg' olarak kaydedildi.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration()
