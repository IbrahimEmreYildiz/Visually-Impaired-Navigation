# Görme Engelliler İçin Yapay Zeka Tabanlı Navigasyon Sistemi

Görme engelli bireylerin çevrelerindeki engelleri gerçek zamanlı olarak tespit edip sesli bildirim almasını sağlayan iki modüllü bir sistem. YOLO26 nesne algılama modeli kullanılarak hem bilgisayar kamerası (Python) hem de mobil cihaz (Flutter) üzerinde çalışır.

---

## 📂 Proje Yapısı

```text
visually_impaired_navigation/
│
├── python_server/                        # PC Sunucu & Standalone Modülü
│   ├── src/
│   │   ├── config.py                     # Yapılandırma (frozen dataclass'lar)
│   │   ├── detector.py                   # YOLO26 sarmalayıcısı (half=True, imgsz=416)
│   │   ├── distance_estimator.py         # Pinhole kamera mesafe tahmini + yön belirleme
│   │   ├── audio_feedback.py             # pyttsx3 TTS — daemon thread + queue
│   │   ├── visualizer.py                 # Debug bounding box görselleştirme
│   │   ├── main.py                       # Standalone mod (webcam + İngilizce TTS)
│   │   ├── inference_server.py           # UDP çıkarım sunucusu — port 9999
│   │   ├── calibrate_focal.py            # Odak uzaklığı kalibrasyon aracı
│   │   └── export_model.py               # YOLO26 → TFLite dışa aktarma
│   ├── requirements.txt
│   └── calibration_image_sample_data_20x128x128x3_float32.npy
│
└── flutter_app/                          # Android / iOS Mobil Uygulama
    ├── lib/main.dart                     # Tüm uygulama mantığı tek dosyada
    ├── assets/
    │   ├── yolo26n.tflite                # YOLO26 TFLite modeli (float16, 9.9 MB)
    │   └── labels.txt                    # 80 COCO sınıf etiketi
    ├── android/
    ├── ios/
    └── pubspec.yaml
```

---

## 📦 Model Ağırlıklarını İndirme

`yolo26n.pt` ve `yolo26n.onnx` dosyaları boyutları nedeniyle repoya dahil edilmemiştir.
Python sunucusunu çalıştırmak için `python_server/` klasörüne indirin:

```bash
# Ultralytics ile otomatik indirme (requirements kurulduktan sonra):
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

> Flutter uygulaması için gerekli TFLite modeli (`flutter_app/assets/yolo26n.tflite`) repoda mevcuttur — ayrıca indirme gerekmez.

---

## 🚀 Kurulum ve Çalıştırma

### 🐍 1. Python Modülü

Python modülü iki farklı modda çalışabilir:

**Gerekli kütüphaneleri kurun:**
```bash
cd python_server
pip install -r requirements.txt
```

**a) Standalone mod — bilgisayar kamerası ile:**
```bash
python src/main.py
```
Webcam görüntüsü açılır, tespit edilen en yakın nesne İngilizce olarak seslendirilir (`"chair ahead, 1.2 meters"`). Çıkmak için `q`.

**b) UDP sunucu modu — harici istemci için:**
```bash
python src/inference_server.py
```
Port `9999`'da UDP dinlemeye başlar. İstemci `4 bayt frame_id + JPEG` gönderir; sunucu `frame_id + JSON tespitler` döner.

---

### 📱 2. Flutter Mobil Uygulaması

Flutter uygulaması YOLO26 TFLite modelini doğrudan telefon üzerinde çalıştırır — internet bağlantısı veya Python sunucu gerektirmez.

**Gereksinimler:**
- Flutter SDK 3.x veya üzeri
- Android 5.0+ cihaz (iOS da desteklenir)
- USB Hata Ayıklama modunun açık olması

**Derleme ve yükleme:**
```bash
cd flutter_app
flutter pub get
flutter run --release
```

> [!IMPORTANT]
> Mutlaka `--release` modunda çalıştırın. Debug modunda YUV→RGB dönüşümü 5-10× yavaş çalışır ve kamera donabilir.

---

## 🔍 Nasıl Çalışır?

### Python Standalone Modu

```
Webcam → ObjectDetector (YOLO26, conf=0.40, imgsz=416, half=True)
       → DistanceEstimator (pinhole model: d = H_real × f / pixel_height)
       → AudioFeedback (pyttsx3, İngilizce, en yakın nesne, 2.5s cooldown/sınıf)
       → Visualizer (bounding box + mesafe etiketi, FPS göstergesi)
```

İki ayrı thread çalışır: ana thread kamera karelerini okur, `inference_worker` thread YOLO çıkarımı ve TTS kuyruğunu yönetir. Kamera buffer'ı `CAP_PROP_BUFFERSIZE=1` ile 1 kareye kısıtlanmıştır — her zaman en güncel kare işlenir.

### Flutter Mobil Uygulaması

```
Kamera (YUV420) → Optimized YUV→RGB (bit-shift, LUT) → YOLO26 TFLite
                → Pinhole mesafe tahmini → Türkçe TTS (flutter_tts, tr-TR)
```

Her kare için:
1. `_isProcessing` bayrağı ile çift çıkarım engellenir
2. YUV→RGB dönüşümü önceden hesaplanmış offset dizileri (`_yOffsets`, `_uvOffsets`) ile tek geçişte tamamlanır
3. YOLO26 çıktısı `(300, 6)` formatındadır: `[x1, y1, x2, y2, güven, sınıf_id]` — NMS modelin içinde

---

## 🔊 Sesli Geri Bildirim

### Python (İngilizce)
- En yakın **tek** nesne seslendirilir: `"chair ahead, 1.2 meters"`
- Yön: `"left"` (merkez < 0.35), `"ahead"` (0.35–0.65), `"right"` (merkez > 0.65)
- Her sınıf için bağımsız **2.5 saniyelik cooldown** (aynı anda birden fazla sınıf duyurulabilir)

### Flutter (Türkçe)
- 0.05–12 metre aralığındaki tespitler değerlendirilir
- En yakın **3 benzersiz nesne** sırayla seslendirilir: `"sandalye önünde, 1.2 metre. insan solunda, 2.8 metre"`
- Yön: `"solunda"` (< 0.35), `"önünde"` (0.35–0.65), `"sağında"` (> 0.65)
- **Global 4 saniyelik cooldown** — ancak kadrajdaki en yakın kişi **0.3 metre** daha yaklaşırsa veya yön değiştirirse hemen yeniden seslendirme yapılır
- Ses tamamen kapatılabilir (sağ üst köşedeki 🔇 düğmesi)

---

## 🛠️ Performans Optimizasyonları (Flutter)

| Optimizasyon | Detay |
|---|---|
| **Bit-shift YUV→RGB** | `float` çarpımı yerine `(298*c + 409*e + 128) >> 8` tamsayı aritmetiği |
| **Normalizasyon LUT** | 256 elemanlı önceden hesaplanmış tablo — bölme işlemi kaldırıldı |
| **Önceden ayrılmış buffer** | `Float32List(1*640*640*3)` uygulama başlarken bir kez ayrılır |
| **Offset dizisi önbelleği** | `_yOffsets` / `_uvOffsets` dizileri yalnızca kamera boyutu değiştiğinde yeniden hesaplanır |
| **Orta-kare kırpma** | Kameranın kısa kenarı kadar kare ortadan alınır → aspect ratio bozulması olmadan 640×640'a ölçeklenir |
| **NMS-Free YOLO26** | YOLOv8'in 8–12ms CPU NMS adımı modelin içine gömüldü, ayrı post-processing yok |
| **Çok çekirdek + GPU** | Android: 4 thread CPU (NNAPI devre dışı — end-to-end NMS operatörü ile uyumsuz). iOS: GPU delegate |

---

## 📊 Canlı Performans Paneli (Flutter)

Ekranın alt kısmındaki yarı saydam panel:

| Gösterge | Açıklama | Hedef |
|---|---|---|
| **YUV → RGB** | Kamera ham verisinin dönüşüm süresi | < 15 ms |
| **YOLO26** | Model çıkarım süresi | < 120 ms |
| **Gerçek FPS** | Saniyede işlenen kare sayısı | > 8 FPS |

Ayrıca bounding box rengi mesafeye göre değişir: **kırmızı** (< 1.5 m tehlike bölgesi) / **mor** (uzak).

---

## ⚙️ Temel Parametreler

| Parametre | Python | Flutter |
|---|---|---|
| Model | `yolo26n.pt` | `yolo26n.tflite` (float16) |
| Güven eşiği | 0.40 | 0.40 |
| Giriş boyutu | 416×416 | 640×640 |
| TTS dili | İngilizce (pyttsx3) | Türkçe (flutter_tts, tr-TR) |
| Cooldown | 2.5 s / sınıf | 4 s global |
| Duyurulan nesne | En yakın 1 | En yakın 3 benzersiz |

---

## 📋 Gereksinimler

**Python:**
```
Python 3.10+
ultralytics, opencv-python, numpy, pyttsx3
```

**Flutter:**
```
Flutter 3.x+
tflite_flutter, camera, flutter_tts
```
