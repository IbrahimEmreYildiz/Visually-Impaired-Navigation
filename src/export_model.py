"""Export YOLO26n to TFLite format for on-device Flutter inference.

YOLO26 supports two export modes:

  * **end2end=True** (default) – bakes NMS into the graph so the output is a
    clean ``(1, 300, 6)`` tensor with ``[x1, y1, x2, y2, conf, class_id]``.
    This is the recommended mode because it removes the need for a separate
    NMS step on the mobile device, giving a significant FPS boost.

  * **end2end=False** – produces the legacy ``(1, 84, 8400)`` raw-grid
    output identical to YOLOv8.  Use this only as a fallback if the end-to-end
    graph contains TFLite-unsupported operators on your target device.
"""

from ultralytics import YOLO


def main() -> None:
    print("YOLO26n modelini TFLite formatina donusturme islemi basliyor...")
    # Modeli yukle (yolo26n.pt ilk calistirmada otomatik indirilir)
    model = YOLO("yolo26n.pt")

    # ── Birincil: End-to-end (NMS-free) export ──────────────────────────
    # Cikti tensoru: [1, 300, 6]  →  [x1, y1, x2, y2, confidence, class_id]
    # Flutter tarafinda NMS gerekmez, dogrudan parse edilir.
    model.export(format="tflite", imgsz=640, end2end=True)
    print("End-to-end TFLite donusturme tamamlandi! (yolo26n.tflite)")

    # ── Yedek: Legacy (NMS gerektirir) export ───────────────────────────
    # Asagidaki satiri aktif etmek istersen end2end satirini yoruma al.
    # model.export(format="tflite", imgsz=640, end2end=False)
    # print("Legacy TFLite donusturme tamamlandi! (yolo26n.tflite)")


if __name__ == "__main__":
    main()
