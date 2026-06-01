import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Set preferred orientations to portrait only
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Görme Engelli Navigasyon',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        primaryColor: const Color(0xFF6C63FF),
        scaffoldBackgroundColor: const Color(0xFF0F0E17),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF6C63FF),
          secondary: Color(0xFFFF8906),
          surface: Color(0xFF16161A),
          background: Color(0xFF0F0E17),
        ),
        useMaterial3: true,
      ),
      home: const MainNavigationScreen(),
    );
  }
}

class Detection {
  final int classId;
  final String className;
  final double confidence;
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  Detection({
    required this.classId,
    required this.className,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });

  double get boxHeight => y2 - y1;
  double get boxWidth => x2 - x1;
  double get centerX => (x1 + x2) / 2.0;
}

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  CameraController? _cameraController;
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isModelLoading = true;
  bool _isProcessing = false;
  List<Detection> _detections = [];
  
  // TTS Settings
  final FlutterTts _flutterTts = FlutterTts();
  bool _isMuted = false;
  final Map<String, DateTime> _ttsCooldowns = {};
  final Duration _cooldownDuration = const Duration(seconds: 3);

  // Calibration Settings
  double _focalLength = 600.0; // Default focal length in pixels

  // Performance Metrics
  double _conversionTimeMs = 0.0;
  double _inferenceTimeMs = 0.0;
  double _totalTimeMs = 0.0;
  DateTime? _lastFrameTime;

  // Person Tracking for TTS
  double? _lastSpokenPersonDistance;
  DateTime? _lastSpokenPersonTime;



  // ── Performance: Pre-allocated input buffer (avoids ~5MB allocation per frame) ──
  final Float32List _inputBuffer = Float32List(1 * 640 * 640 * 3);

  // ── Performance: Pre-allocated index mapping arrays ──
  Int32List? _srcXArray;
  Int32List? _srcYArray;
  Int32List? _yOffsets;
  Int32List? _uvOffsets;
  int _lastImageWidth = 0;
  int _lastImageHeight = 0;
  int _lastRowStrideY = 0;
  int _lastRowStrideUV = 0;
  int _lastPixelStrideUV = 0;

  // Center-square crop state (set once per camera init)
  int _cropSize = 640;
  int _cropXOffset = 0;
  int _cropYOffset = 0;

  // Real FPS counter
  DateTime? _fpsCounterTime;
  int _fpsFrameCount = 0;
  double _realFps = 0.0;

  // Lookup table for fast normalization (division by 255.0)
  static final Float32List _normalizeTable = Float32List.fromList(
    List.generate(256, (i) => i / 255.0),
  );

  // Average real-world heights (meters) — tüm 80 COCO sınıfı tanımlanmış
  final Map<String, double> _knownHeights = {
    // İnsanlar
    "person": 1.70,
    // Araçlar
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.10,
    "airplane": 5.00,
    "bus": 3.20,
    "train": 4.00,
    "truck": 3.50,
    "boat": 2.50,
    // Sokak
    "traffic light": 0.75,
    "fire hydrant": 0.60,
    "stop sign": 0.75,
    "parking meter": 1.20,
    "bench": 0.90,
    // Hayvanlar
    "bird": 0.30,
    "cat": 0.30,
    "dog": 0.50,
    "horse": 1.60,
    "sheep": 0.90,
    "cow": 1.50,
    "elephant": 3.00,
    "bear": 1.20,
    "zebra": 1.50,
    "giraffe": 5.50,
    // Kişisel eşya
    "backpack": 0.45,
    "umbrella": 1.00,
    "handbag": 0.30,
    "tie": 0.60,
    "suitcase": 0.70,
    // Spor & outdoor
    "frisbee": 0.03,
    "skis": 1.50,
    "snowboard": 1.50,
    "sports ball": 0.22,
    "kite": 0.60,
    "baseball bat": 1.00,
    "baseball glove": 0.25,
    "skateboard": 0.15,
    "surfboard": 2.00,
    "tennis racket": 0.70,
    // Mutfak
    "bottle": 0.25,
    "wine glass": 0.20,
    "cup": 0.10,
    "fork": 0.18,
    "knife": 0.18,
    "spoon": 0.16,
    "bowl": 0.10,
    "banana": 0.20,
    "apple": 0.08,
    "sandwich": 0.08,
    "orange": 0.08,
    "broccoli": 0.20,
    "carrot": 0.20,
    "hot dog": 0.12,
    "pizza": 0.05,
    "donut": 0.05,
    "cake": 0.12,
    // Mobilya & ev
    "chair": 0.90,
    "couch": 0.85,
    "potted plant": 0.50,
    "bed": 0.60,
    "dining table": 0.75,
    "toilet": 0.70,
    "tv": 0.60,
    "laptop": 0.25,
    "mouse": 0.05,
    "remote": 0.05,
    "keyboard": 0.05,
    "cell phone": 0.15,
    "microwave": 0.30,
    "oven": 0.90,
    "toaster": 0.20,
    "sink": 0.85,
    "refrigerator": 1.80,
    "book": 0.25,
    "clock": 0.30,
    "vase": 0.30,
    "scissors": 0.15,
    "teddy bear": 0.35,
    "hair drier": 0.20,
    "toothbrush": 0.18,
  };

  // Class translation to Turkish for voice guidance
  final Map<String, String> _turkishLabels = {
    "person": "insan",
    "bicycle": "bisiklet",
    "car": "araba",
    "motorcycle": "motosiklet",
    "bus": "otobüs",
    "truck": "kamyon",
    "traffic light": "trafik ışığı",
    "stop sign": "dur tabelası",
    "bench": "bank",
    "chair": "sandalye",
    "couch": "koltuk",
    "dining table": "masa",
    "bed": "yatak",
    "toilet": "tuvalet",
    "laptop": "dizüstü bilgisayar",
    "cell phone": "telefon",
    "book": "kitap",
    "clock": "saat",
    "vase": "vazo",
    "potted plant": "saksı bitkisi",
    "backpack": "sırt çantası",
    "bottle": "şişe",
    "cup": "bardak",
    "fork": "çatal",
    "knife": "bıçak",
    "spoon": "kaşık",
    "bowl": "kase",
    "handbag": "çanta",
    "umbrella": "şemsiye",
    "tv": "televizyon",
    "mouse": "fare",
    "remote": "kumanda",
    "keyboard": "klavye",
    "microwave": "mikrodalga",
    "oven": "fırın",
    "toaster": "tost makinası",
    "sink": "lavabo",
    "refrigerator": "buzdolabı",
    "scissors": "makas",
    "teddy bear": "oyuncak ayı",
    "hair drier": "saç kurutma makinası",
    "toothbrush": "diş fırçası",
    "dog": "köpek",
    "cat": "kedi",
    "horse": "at",
    "bird": "kuş",
    "airplane": "uçak",
    "train": "tren",
    "boat": "tekne",
    "fire hydrant": "yangın musluğu",
    "suitcase": "bavul",
    "banana": "muz",
    "apple": "elma",
    "sandwich": "sandviç",
    "orange": "portakal",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "pasta",
  };

  @override
  void initState() {
    super.initState();
    _initializeTTS();
    _loadModel();
  }

  Future<void> _initializeTTS() async {
    await _flutterTts.setLanguage("tr-TR");
    await _flutterTts.setSpeechRate(0.55);
    await _flutterTts.setVolume(1.0);
    
    // Play welcome sound/text
    _speakNow("Navigasyon sistemi başlatıldı. Model yükleniyor.");
  }

  Future<void> _loadModel() async {
    try {
      setState(() {
        _isModelLoading = true;
      });

      // Load labels
      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n').map((l) => l.trim()).where((l) => l.isNotEmpty).toList();

      // Load interpreter
      final options = InterpreterOptions();
      options.threads = 4;
      
      // On Android, we run on CPU with 4 threads and XNNPACK (enabled by default) for max stability.
      // NNAPI delegates often freeze/hang on custom NMS operators of end-to-end YOLO models.
      if (Platform.isIOS) {
        options.addDelegate(GpuDelegate());
      }
      debugPrint("Loading TFLite model on CPU (threads: 4)...");
      _interpreter = await Interpreter.fromAsset('assets/yolo26n.tflite', options: options);
      _interpreter!.allocateTensors();
      


      _speakNow("Model başarıyla yüklendi. Kamera başlatılıyor.");
      await _initializeCamera();
      
    } catch (e) {
      debugPrint("Error loading model: $e");
      _speakNow("Model yükleme hatası.");
    } finally {
      setState(() {
        _isModelLoading = false;
      });
    }
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      _speakNow("Kamera bulunamadı.");
      return;
    }

    // Find the back-facing camera
    final backCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _cameraController = CameraController(
      backCamera,
      ResolutionPreset.medium, // ~640x480 — 3x daha az piksel, YUV dönüşümü çok daha hızlı
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.yuv420 : ImageFormatGroup.bgra8888,
    );

    try {
      await _cameraController!.initialize();
      // Lock sensor orientation if needed, default is fine
      await _cameraController!.startImageStream(_processCameraFrame);
      setState(() {});
    } catch (e) {
      debugPrint("Error initializing camera: $e");
      _speakNow("Kamera açma hatası.");
    }
  }

  void _speakNow(String text) async {
    if (_isMuted) return;
    await _flutterTts.speak(text);
  }

  // Camera frame streaming callback
  void _processCameraFrame(CameraImage image) async {
    if (_interpreter == null || _isProcessing) return;
    _isProcessing = true;

    // Gerçek FPS ölçümü
    _fpsFrameCount++;
    final now = DateTime.now();
    _fpsCounterTime ??= now;
    final elapsed = now.difference(_fpsCounterTime!).inMilliseconds;
    if (elapsed >= 1000) {
      _realFps = _fpsFrameCount * 1000.0 / elapsed;
      _fpsFrameCount = 0;
      _fpsCounterTime = now;
    }
    final stopwatch = Stopwatch()..start();

    try {
      // ── Step 1: Optimized YUV→RGB conversion on main thread ──
      final convStart = stopwatch.elapsedMilliseconds;
      _convertImageToBuffer(image);
      final convEnd = stopwatch.elapsedMilliseconds;
      final conversionTime = convEnd - convStart;

      // ── Step 2: TFLite inference using manual zero-copy memory transfers ──
      final inferStart = stopwatch.elapsedMilliseconds;
      _interpreter!.getInputTensor(0).data = _inputBuffer.buffer.asUint8List();
      _interpreter!.invoke();
      
      final outputBytes = _interpreter!.getOutputTensor(0).data;
      final outputFloats = Float32List.sublistView(outputBytes);
      final finalDetections = _parseYolo26Output(outputFloats);
      final inferEnd = stopwatch.elapsedMilliseconds;
      final inferenceTime = inferEnd - inferStart;
      final totalTime = stopwatch.elapsedMilliseconds;

      if (mounted) {
        setState(() {
          _detections = finalDetections;
          _conversionTimeMs = conversionTime.toDouble();
          _inferenceTimeMs = inferenceTime.toDouble();
          _totalTimeMs = totalTime.toDouble();
        });
        debugPrint("PERF: YUV->RGB: ${conversionTime}ms, YOLO: ${inferenceTime}ms, Total: ${totalTime}ms | Gerçek FPS: ${_realFps.toStringAsFixed(1)}");

        _handleAudioFeedback(finalDetections);
      }
    } catch (e) {
      debugPrint("Inference error: $e");
    } finally {
      _isProcessing = false;
    }
  }

  // Pre-calculate flat offsets for YUV→RGB conversion.
  // Center-square crop: kameranın kısa kenarı kadar kare alan ortadan kırpılır,
  // sonra 640×640'a ölçeklenir. Bu sayede aspect ratio bozulması olmaz ve
  // mesafe tahminleri doğru kalır (nesneler ezilmiş görünmez).
  void _rebuildIndexArrays(int width, int height, int rowStrideY, int rowStrideUV, int pixelStrideUV) {
    if (width == _lastImageWidth &&
        height == _lastImageHeight &&
        rowStrideY == _lastRowStrideY &&
        rowStrideUV == _lastRowStrideUV &&
        pixelStrideUV == _lastPixelStrideUV) {
      return;
    }

    _lastImageWidth = width;
    _lastImageHeight = height;
    _lastRowStrideY = rowStrideY;
    _lastRowStrideUV = rowStrideUV;
    _lastPixelStrideUV = pixelStrideUV;

    // Kameranın kısa kenarı kadar kare al, ortadan kırp
    final cropSize = width < height ? width : height;
    final xCropOffset = (width - cropSize) ~/ 2;
    final yCropOffset = (height - cropSize) ~/ 2;
    _cropSize = cropSize;
    _cropXOffset = xCropOffset;
    _cropYOffset = yCropOffset;

    _yOffsets = Int32List(640 * 640);
    _uvOffsets = Int32List(640 * 640);

    int index = 0;
    for (int y = 0; y < 640; y++) {
      final srcYi = yCropOffset + (y * cropSize) ~/ 640;
      final srcYuv = srcYi ~/ 2;
      final yRowOff = srcYi * rowStrideY;
      final uvRowOff = srcYuv * rowStrideUV;

      for (int x = 0; x < 640; x++) {
        final srcXi = xCropOffset + (x * cropSize) ~/ 640;
        _yOffsets![index] = yRowOff + srcXi;

        final srcXuv = srcXi ~/ 2;
        _uvOffsets![index] = uvRowOff + srcXuv * pixelStrideUV;
        index++;
      }
    }

    // BGRA (iOS) için lookup tabloları
    _srcXArray = Int32List(640);
    _srcYArray = Int32List(640);
    for (int x = 0; x < 640; x++) {
      _srcXArray![x] = xCropOffset + (x * cropSize) ~/ 640;
    }
    for (int y = 0; y < 640; y++) {
      _srcYArray![y] = yCropOffset + (y * cropSize) ~/ 640;
    }
  }

  // Optimized YUV-to-RGB conversion using flat pre-calculated lookup arrays.
  // This performs the conversion in ~8-12ms on the main thread without triggering isolates or copying planes.
  void _convertImageToBuffer(CameraImage image) {
    final width = image.width;
    final height = image.height;

    if (image.format.group == ImageFormatGroup.yuv420) {
      final planeY = image.planes[0].bytes;
      final planeU = image.planes[1].bytes;
      final planeV = image.planes[2].bytes;

      final rowStrideY = image.planes[0].bytesPerRow;
      final rowStrideUV = image.planes[1].bytesPerRow;
      final pixelStrideUV = image.planes[1].bytesPerPixel ?? 1;

      _rebuildIndexArrays(width, height, rowStrideY, rowStrideUV, pixelStrideUV);

      final yOffsets = _yOffsets!;
      final uvOffsets = _uvOffsets!;
      final normalizeTable = _normalizeTable;
      final inputBuf = _inputBuffer;

      int pixelIndex = 0;
      final lenY = planeY.length;
      final lenU = planeU.length;
      final lenV = planeV.length;

      for (int i = 0; i < 409600; i++) {
        final idxY = yOffsets[i];
        final idxUV = uvOffsets[i];

        final yv = idxY < lenY ? planeY[idxY] : 0;
        final uv = idxUV < lenU ? planeU[idxUV] : 128;
        final vv = idxUV < lenV ? planeV[idxUV] : 128;

        final c = yv - 16;
        final d = uv - 128;
        final e = vv - 128;

        final r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255);
        final g = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255);
        final b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255);

        inputBuf[pixelIndex++] = normalizeTable[r];
        inputBuf[pixelIndex++] = normalizeTable[g];
        inputBuf[pixelIndex++] = normalizeTable[b];
      }
    } else {
      // iOS BGRA fallback
      _rebuildIndexArrays(width, height, 0, 0, 0);
      final bytes = image.planes[0].bytes;
      final rowStride = image.planes[0].bytesPerRow;
      final pixelStride = image.planes[0].bytesPerPixel ?? 4;
      final normalizeTable = _normalizeTable;
      final inputBuf = _inputBuffer;

      int pixelIndex = 0;
      for (int y = 0; y < 640; y++) {
        final srcY = _srcYArray![y];
        final yRowOffset = srcY * rowStride;
        for (int x = 0; x < 640; x++) {
          final srcX = _srcXArray![x];
          final index = yRowOffset + srcX * pixelStride;
          if (index + 2 >= bytes.length) {
            pixelIndex += 3;
            continue;
          }
          inputBuf[pixelIndex++] = normalizeTable[bytes[index + 2]];
          inputBuf[pixelIndex++] = normalizeTable[bytes[index + 1]];
          inputBuf[pixelIndex++] = normalizeTable[bytes[index]];
        }
      }
    }
  }

  // Parse YOLO26 end2end output: [300, 6] → [x1, y1, x2, y2, confidence, class_id]
  // NMS is already handled inside the model graph – no post-processing needed!
  List<Detection> _parseYolo26Output(Float32List output) {
    final List<Detection> detections = [];
    const double confidenceThreshold = 0.40;

    for (int i = 0; i < 300; i++) {
      final offset = i * 6;
      if (offset + 5 >= output.length) break;
      final confidence = output[offset + 4];

      // Skip empty/low-confidence slots (YOLO26 pads to 300)
      if (confidence < confidenceThreshold) continue;

      final classId = output[offset + 5].toInt();
      if (classId < 0 || classId >= _labels.length) continue;

      // Scale coordinates from normalized range [0.0, 1.0] to model input resolution [0.0, 640.0]
      detections.add(Detection(
        classId: classId,
        className: _labels[classId],
        confidence: confidence,
        x1: output[offset + 0] * 640.0,
        y1: output[offset + 1] * 640.0,
        x2: output[offset + 2] * 640.0,
        y2: output[offset + 3] * 640.0,
      ));
    }

    return detections;
  }

  // Estimate distance using the pinhole model
  double _estimateDistance(Detection detection) {
    final realHeight = _knownHeights[detection.className] ?? 0.50;
    final pixelHeight = detection.boxHeight;
    if (pixelHeight <= 0) return double.infinity;
    return (realHeight * _focalLength) / pixelHeight;
  }

  // Estimate direction (left, center/ahead, right)
  String _estimateDirection(Detection detection) {
    final normalizedCenter = detection.centerX / 640.0;
    if (normalizedCenter < 0.35) {
      return "solunda";
    } else if (normalizedCenter > 0.65) {
      return "sağında";
    } else {
      return "önünde";
    }
  }

  DateTime? _lastAnnouncementTime;
  String? _lastSpokenPersonDirection;

  // Handle TTS cooldowns and trigger voice commands
  void _handleAudioFeedback(List<Detection> detections) {
    if (_isMuted || detections.isEmpty) return;

    // Filter detections to those within 12 meters
    final validDetections = detections.where((d) {
      final dist = _estimateDistance(d);
      return dist < 12.0 && dist > 0.05;
    }).toList();

    if (validDetections.isEmpty) {
      return;
    }

    final now = DateTime.now();

    // Check if there is a person closest to us
    final people = validDetections.where((d) => d.className == 'person').toList();
    bool forceSpeak = false;
    
    if (people.isNotEmpty) {
      people.sort((a, b) => _estimateDistance(a).compareTo(_estimateDistance(b)));
      final closestPerson = people.first;
      final distance = _estimateDistance(closestPerson);
      final direction = _estimateDirection(closestPerson);

      // Force an update if a person gets significantly closer (by 0.3m or more) or changes direction
      if (_lastSpokenPersonDistance != null) {
        final distDiff = _lastSpokenPersonDistance! - distance; // positive if getting closer
        if (distDiff >= 0.3 || _lastSpokenPersonDirection != direction) {
          forceSpeak = true;
          debugPrint("TTS Force Speak: Person distance/direction changed significantly");
        }
      }
    }

    // Global cooldown check: 4 seconds between announcements unless forced (like a person getting closer)
    if (!forceSpeak && 
        _lastAnnouncementTime != null && 
        now.difference(_lastAnnouncementTime!) < const Duration(seconds: 4)) {
      return;
    }

    // Sort valid detections by proximity
    validDetections.sort((a, b) => _estimateDistance(a).compareTo(_estimateDistance(b)));

    final List<String> currentUniqueObjects = [];
    final List<String> speechParts = [];

    for (final d in validDetections) {
      final trName = _turkishLabels[d.className] ?? d.className;
      if (!currentUniqueObjects.contains(trName)) {
        currentUniqueObjects.add(trName);
        final dist = _estimateDistance(d);
        final dir = _estimateDirection(d);
        speechParts.add("$trName $dir, ${dist.toStringAsFixed(1)} metre");
      }
      // Limit to 3 closest unique objects to avoid long spoken lists
      if (currentUniqueObjects.length >= 3) break;
    }

    if (speechParts.isNotEmpty) {
      final textToSpeak = speechParts.join(". ");
      _lastAnnouncementTime = now;
      
      // Update closest person tracking state
      if (people.isNotEmpty) {
        people.sort((a, b) => _estimateDistance(a).compareTo(_estimateDistance(b)));
        _lastSpokenPersonDistance = _estimateDistance(people.first);
        _lastSpokenPersonDirection = _estimateDirection(people.first);
      } else {
        _lastSpokenPersonDistance = null;
        _lastSpokenPersonDirection = null;
      }
      
      debugPrint("TTS Speaking: $textToSpeak");
      _speakNow(textToSpeak);
    }
  }

  Widget _buildStatCard(String label, String value, IconData icon) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: const Color(0xFF6C63FF)),
            const SizedBox(width: 4),
            Text(
              label,
              style: const TextStyle(
                fontSize: 10,
                color: Colors.white54,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _interpreter?.close();
    _flutterTts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final isCameraReady = _cameraController != null && _cameraController!.value.isInitialized;

    return Scaffold(
      body: Stack(
        children: [
          // Camera Preview
          if (isCameraReady)
            SizedBox(
              width: size.width,
              height: size.height,
              child: AspectRatio(
                aspectRatio: _cameraController!.value.aspectRatio,
                child: CameraPreview(_cameraController!),
              ),
            )
          else
            const Center(
              child: CircularProgressIndicator(
                color: Color(0xFF6C63FF),
              ),
            ),

          // Detection Bounding Boxes Overlay
          if (isCameraReady && _detections.isNotEmpty)
            ..._detections.map((d) {
              final dist = _estimateDistance(d);
              final dir = _estimateDirection(d);
              
              // Scale factor from 640x640 model size to actual screen size
              final scaleX = size.width / 640.0;
              final scaleY = size.height / 640.0;

              return Positioned(
                left: d.x1 * scaleX,
                top: d.y1 * scaleY,
                width: d.boxWidth * scaleX,
                height: d.boxHeight * scaleY,
                child: Container(
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: dist < 1.5 ? Colors.redAccent : const Color(0xFF6C63FF),
                      width: 3,
                    ),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Align(
                    alignment: Alignment.topLeft,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      color: dist < 1.5 ? Colors.redAccent : const Color(0xFF6C63FF),
                      child: Text(
                        "${_turkishLabels[d.className] ?? d.className} (${dist.toStringAsFixed(1)}m)",
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                ),
              );
            }).toList(),

          // Glassmorphism Dashboard Overlay
          Positioned(
            left: 20,
            right: 20,
            bottom: 30,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(24),
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: const Color(0xFF16161A).withOpacity(0.85),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(
                    color: Colors.white.withOpacity(0.1),
                    width: 1.5,
                  ),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Title and Mute Toggle
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Row(
                          children: [
                            Container(
                              width: 12,
                              height: 12,
                              decoration: BoxDecoration(
                                color: _isModelLoading ? Colors.amber : Colors.greenAccent,
                                shape: BoxShape.circle,
                              ),
                            ),
                            const SizedBox(width: 8),
                            Text(
                              _isModelLoading ? 'Model Yükleniyor...' : 'Sistem Aktif',
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                          ],
                        ),
                        IconButton(
                          icon: Icon(
                            _isMuted ? Icons.volume_off_rounded : Icons.volume_up_rounded,
                            color: _isMuted ? Colors.redAccent : const Color(0xFF6C63FF),
                          ),
                          iconSize: 28,
                          onPressed: () {
                            setState(() {
                              _isMuted = !_isMuted;
                            });
                            _speakNow(_isMuted ? "Ses kapatıldı." : "Ses açıldı.");
                          },
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    
                    // FPS and Latency Stats Row
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildStatCard("YUV ➔ RGB", "${_conversionTimeMs.toStringAsFixed(0)} ms", Icons.transform_rounded),
                        _buildStatCard("YOLO26", "${_inferenceTimeMs.toStringAsFixed(0)} ms", Icons.memory_rounded),
                        _buildStatCard("Gerçek FPS", "${_realFps.toStringAsFixed(1)}", Icons.speed_rounded),
                      ],
                    ),
                    
                    const Divider(color: Colors.white12, height: 20),
                    
                    // Focal Length / Calibration Slider
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              "Mesafe Kalibrasyonu (Odak Uzaklığı)",
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.white70,
                              ),
                            ),
                            Text(
                              "${_focalLength.round()} px",
                              style: const TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFFFF8906),
                              ),
                            ),
                          ],
                        ),
                        SliderTheme(
                          data: SliderTheme.of(context).copyWith(
                            activeTrackColor: const Color(0xFF6C63FF),
                            inactiveTrackColor: Colors.white12,
                            thumbColor: const Color(0xFFFF8906),
                            overlayColor: const Color(0xFFFF8906).withOpacity(0.2),
                          ),
                          child: Slider(
                            value: _focalLength,
                            min: 200.0,
                            max: 1500.0,
                            onChanged: (value) {
                              setState(() {
                                _focalLength = value;
                              });
                            },
                          ),
                        ),
                      ],
                    ),
                    
                    // Detected Objects Summary
                    if (_detections.isNotEmpty) ...[
                      const SizedBox(height: 10),
                      const Align(
                        alignment: Alignment.centerLeft,
                        child: Text(
                          "Algılanan Nesneler:",
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.white60,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      const SizedBox(height: 6),
                      SizedBox(
                        height: 50,
                        child: ListView(
                          scrollDirection: Axis.horizontal,
                          children: _detections.map((d) {
                            final dist = _estimateDistance(d);
                            final trName = _turkishLabels[d.className] ?? d.className;
                            return Container(
                              margin: const EdgeInsets.only(right: 8),
                              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.06),
                                borderRadius: BorderRadius.circular(12),
                                border: Border.all(
                                  color: Colors.white12,
                                ),
                              ),
                              child: Row(
                                children: [
                                  Icon(
                                    Icons.radar_rounded,
                                    size: 14,
                                    color: dist < 1.5 ? Colors.redAccent : const Color(0xFF6C63FF),
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    "$trName (${dist.toStringAsFixed(1)}m)",
                                    style: TextStyle(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w500,
                                      color: dist < 1.5 ? Colors.redAccent : Colors.white,
                                    ),
                                  ),
                                ],
                              ),
                            );
                          }).toList(),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
