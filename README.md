# Görme Engelliler İçin Yapay Zeka Tabanlı Navigasyon ve Mesafe Tahmin Sistemi

Bu proje, görme engelli bireylerin günlük yaşamlarında önlerindeki engelleri algılayıp güvenli bir şekilde hareket edebilmelerini sağlamak amacıyla geliştirilmiştir. Proje, gerçek zamanlı nesne tespiti yapan bir mobil uygulama (Flutter) ve bu sistemin temelini oluşturan bilgisayarlı görü / kalibrasyon test ortamını (Python) içermektedir.

---

## 📂 Proje Yapısı

Proje iki ana modülden oluşmaktadır:

```text
visually_impaired_navigation/
│
├── visually_impaired_navigation/     # Python Kalibrasyon & Prototip Modülü
│   ├── src/                         # Python kaynak kodları (görselleştirici, kalibrasyon vb.)
│   ├── requirements.txt             # Python bağımlılık listesi
│   └── yolo26n.pt                   # YOLO26 PyTorch model ağırlıkları
│
└── visually_impaired_navigation_app/ # Flutter Mobil Uygulama Modülü
    ├── lib/main.dart                # Uygulama ana kaynak kodu
    ├── assets/                      # YOLO26n.tflite modeli ve etiket dosyaları
    └── pubspec.yaml                 # Flutter bağımlılık tanımları
```

---

## 🚀 Kurulum ve Çalıştırma

### 🐍 1. Python Modülü (Kalibrasyon & Test)

Python modülü, kameranın odak uzaklığını (`focal_length`) belirlemek ve mesafe ölçüm modelini simüle etmek için kullanılır.

1. **Gerekli kütüphaneleri yükleyin:**
   ```bash
   cd visually_impaired_navigation
   pip install -r requirements.txt
   ```
2. **Uygulamayı çalıştırın:**
   ```bash
   python src/visualizer.py
   ```

---

### 📱 2. Flutter Mobil Uygulaması

Mobil uygulama, YOLO26n TFLite modelini kullanarak Android/iOS cihazlarda tamamen çevrimdışı (offline) çalışır. YOLO26'nın NMS-free (end-to-end) mimarisi sayesinde geleneksel Non-Maximum Suppression adımı modelin içine gömülüdür ve cihaz tarafında ek hesaplama gerektirmez.

#### Gereksinimler:
* Flutter SDK (3.x veya üzeri)
* Android SDK ve USB kablosu
* Telefon üzerinde **Geliştirici Seçenekleri** ve **USB Hata Ayıklama** modunun açık olması
* YOLO26n TFLite modeli (`assets/yolo26n.tflite`) — `export_model.py` ile oluşturulabilir

#### Derleme ve Çalıştırma:
1. Proje dizinine geçiş yapın:
   ```bash
   cd visually_impaired_navigation_app
   ```
2. Bağımlılıkları indirin:
   ```bash
   flutter pub get
   ```
3. Uygulamayı **Release (Maksimum Performans) Modunda** cihazınıza kurun:
   ```bash
   flutter run --release
   ```

> [!IMPORTANT]
> Uygulamanın akıcı çalışması ve gecikme (lag) yapmaması için mutlaka `--release` parametresiyle çalıştırılması gerekmektedir. Debug modunda Dart kodları interpret edildiği için piksel dönüşümleri yavaş kalacaktır.

---

## 🛠️ Yapılan Performans Optimizasyonları

Düşük ve orta segment mobil cihazlarda yapay zeka modelinin kamerayı dondurmaması için aşağıdaki donanım seviyesi optimizasyonlar uygulanmıştır:

1. **Hızlı Tam Sayı Aritmetiği (Integer Bit-Shifting):** Kamera YUV formatından RGB'ye geçilirken CPU'yu yoran kayan noktalı sayı çarpımları (`float/double`) yerine donanım seviyesinde çalışan **Bit Kaydırma (`>> 8`) ve Tam Sayı (Integer)** aritmetiği kullanılmıştır.
2. **Normalizasyon Arama Tablosu (Lookup Table - LUT):** Her piksel değeri normalize edilirken yapılan ve saniyede milyonlarca kez çağrılan bölme işlemi kaldırılmıştır. Bunun yerine 256 elemanlı hazır bir tablo üzerinden doğrudan bellek okuması (O(1)) yapılmaktadır.
3. **Düzleştirilmiş Bellek ve Önbellekleme (O(1) Çıktı Ayrıştırma):** YOLO26'nın end-to-end çıktısı `(1, 300, 6)` formatında doğrudan `[x1, y1, x2, y2, confidence, class_id]` döndürdüğü için, eski `(1, 84, 8400)` formatındaki karmaşık parsing ve NMS adımı tamamen kaldırılmıştır. Bu sayede çıktı ayrıştırma süresi ihmal edilebilir seviyeye düşmüştür.
4. **Gerçek Zamanlı Çıkarım (5 FPS / 200 ms Cooldown):** Model çıkarım ve ayrıştırma hızının optimize edilmesiyle, dinlenme süresi (cooldown) 1.5 saniyeden **200 milisaniyeye** düşürülmüş, sistem neredeyse anlık ve akıcı çalışır hale getirilmiştir.
5. **Donanımsal Hızlandırma:** TFLite Interpreter ayarlarında çoklu çekirdek desteği (4 Thread) ve Android cihazlar için **NNAPI** (Neural Networks API) donanımsal GPU delegeleri aktif edilmiştir.
6. **NMS-Free Mimari (YOLO26):** YOLOv8'den YOLO26'ya geçişle birlikte, Non-Maximum Suppression (NMS) post-processing adımı model grafiğinin içine gömülmüştür. Bu sayede Dart tarafında çalışan O(n²) NMS algoritması tamamen kaldırılmış, hem CPU yükü hem de gecikme süresi azaltılmıştır.

---

## 🔊 Sesli Geri Bildirim ve Kalibrasyon

* **Türkçe TTS (Metinden Sese):** Tespit edilen nesneler ve konumları sesli olarak *"Sandalye, önünde, 1.2 metre"* şeklinde Türkçe olarak seslendirilir.
* **Öncelikli İnsan Takibi (Closest Person Priority):** Kadrajda insan varsa, sistem diğer nesneleri filtreler ve doğrudan en yakın insana odaklanır.
* **0.2 Metre (20 cm) Değişim Kuralı:** Seslerin üst üste binip gürültü yapmaması için, en yakın insanın mesafesi her **0.2 metre (20 cm)** değiştiğinde yeni mesafe sesli söylenir; eğer mesafe değişmezse 4 saniyede bir hatırlatma yapılır.
* **Mesafe Kalibrasyonu (Odak Uzaklığı):** Ekranda bulunan kaydırıcı (Slider) yardımıyla kameranın odak uzaklığı (`focal_length`) gerçek dünyadaki 1 metrelik referans bir nesneye göre kolayca kalibre edilebilir.

---

## 📊 Performans Gösterge Paneli (Dashboard)

Uygulamanın en üst kısmında yer alan bilgi paneli ile aşağıdaki metrikler anlık olarak takip edilebilir:
* **YUV->RGB Dönüşüm Süresi (ms):** Kameranın ham verisinin görsele dönüştürülme süresi (Hedef: < 20 ms).
* **YOLO26 Çıkarım Süresi (ms):** Modelin nesneleri tespit etme süresi (Hedef: < 150 ms). NMS modelin içinde olduğu için ayrı bir post-processing süresi yoktur.
* **Kapasite (FPS):** Cihazın saniyede işleyebileceği maksimum kare sayısı.
