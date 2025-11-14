# YAMNet-Lite + Custom Head (17-Class Home Sound Classifier)

이 프로젝트는 Google YAMNet의 임베딩을 기반으로 **17개 생활 소리만 분류하는 초경량 오디오 분류기**를 구현한 것입니다.  
라즈베리파이·임베디드 IoT 환경에서도 실시간으로 실행 가능하도록 모델을 최적화했습니다.

---

## 🚀 Features

- **YAMNet 256-dim 경량화 백본 사용**
- **17-class custom head TFLite 모델 (FP16, ~1.3MB)**
- WAV 파일 분류 / 실시간 마이크 입력 분류 지원
- TFLite Runtime 기반 Edge 디바이스 최적화
- 클래스별 **AUC / AUPR 성능 제공**

---

## 🔧 Installation

### 1) 가상환경 생성
```bash
python3 -m venv .venv
source .venv/bin/activate

### 2) 패키지 설치
```bash
pip install -r requirements.txt

---

## 📁 Project Structure
YAMNET/
│
├── models/
│   ├── yamnet/
│   │   ├── yamnet.tflite
│   │   └── yamnet-256.tflite
│   └── head/
│       ├── head_1024_fp16.tflite
│       └── head_256_fp16.tflite
│
├── runs_multi/
│   └── per_class_eval_1024.json
│
├── scripts/
│   ├── eval_per_class.py
│   ├── inspect_yamnet_tflite.py
│   ├── realtime_infer_mic.py
│   ├── run_yamnet_plus_head_tflite.py
│   └── train_head_1024.py
│
├── scripts/data/
│   ├── balanced_train_segments.csv
│   ├── class_labels_indices.csv
│   └── ontology.json
│
├── requirements.txt
└── README.md

---

## ▶️ How to Run

### 1) WAV 파일 분류
python scripts/run_yamnet_plus_head_tflite.py

### 2) 실시간 마이크 기반 분류
python scripts/realtime_infer_mic.py

---

## 🧠 Model Overview

### 🔹 YAMNet Backbone (256-dim)
Google AudioSet 기반 모델  
원본 1024-dim → 256-dim 경량화 버전 제공  
Custom Head의 입력 임베딩으로 사용

### 🔹 Custom Head (17-Class, FP16)
YAMNet 임베딩을 입력으로 받아 17개 클래스 분류  
FP16 TFLite (~1.3MB)  
Raspberry Pi 4/5 등 Edge 디바이스에서 실시간 가능

---

## 📊 Evaluation Results (AUC / AUPR)

### 전체 요약
{
  "num_samples": 22212,
  "macro_auc": 0.9898577788296867,
  "macro_aupr": 0.9705225053955527
}

### 클래스별 성능 요약
Class | AUC | AUPR
door | 0.9878 | 0.9579
dishes | 0.9897 | 0.9614
cutlery | 0.9830 | 0.9533
chopping | 0.9796 | 0.9547
frying | 0.9913 | 0.9761
microwave | 0.9941 | 0.9743
blender | 0.9947 | 0.9886
water_tap | 0.9897 | 0.9615
sink | 0.9935 | 0.9665
toilet_flush | 0.9962 | 0.9911
telephone | 0.9953 | 0.9869
chewing | 0.9849 | 0.9659
speech | 0.9902 | 0.9693
television | 0.9819 | 0.9566
footsteps | 0.9788 | 0.9445
vacuum | 0.9980 | 0.9957
hair_dryer | 0.9981 | 0.9940

---

## 📜 Scripts Description

train_head_1024.py  
YAMNet 임베딩 기반 17-class head 학습 및 FP16 TFLite 변환

run_yamnet_plus_head_tflite.py  
WAV → YAMNet → Head TFLite 단일 파이프라인 실행

realtime_infer_mic.py  
마이크 스트림(16kHz) 기반 실시간 분류

inspect_yamnet_tflite.py  
YAMNet TFLite 구조 및 tensor index 자동 분석

eval_per_class.py  
클래스별 AUC / AUPR 계산

---

## 📦 Requirements
tensorflow==2.15.0  
tensorflow-hub  
numpy  
soundfile  
sounddevice  
tflite-runtime  
scikit-learn  

---

## 📄 License
MIT License

---

## 📬 Contact
문의: your-email@example.com  
GitHub Issues로 문의 가능

