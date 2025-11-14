YAMNet-Lite + Custom Head (17-Class Home Sound Classifier)
==========================================================

이 프로젝트는 **Google YAMNet**의 임베딩을 기반으로 **17개 생활 소리만 분류하는 초경량 오디오 분류기**를 구현한 것입니다.라즈베리파이·임베디드 IoT 환경에서도 실시간으로 실행 가능하도록 모델을 최적화했습니다.

📌 Features
-----------

*   **YAMNet 256-dim 경량화 모델** 사용
    
*   **17-class custom head TFLite 모델 (FP16, ~1.3MB)**
    
*   **WAV 파일 분류 / 실시간 마이크 입력 분류** 지원
    
*   **TFLite Runtime 기반 Edge 디바이스 실행 최적화**
    
*   **클래스별 AUC / AUPR 검증 평가 제공**
    

📁 Project Structure
--------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   YAMNET/  │  ├── models/  │   ├── yamnet/  │   │   ├── yamnet.tflite          # 원본 1024-dim YAMNet  │   │   └── yamnet-256.tflite      # 경량화된 256-dim YAMNet  │   │  │   └── head/  │       ├── head_1024_fp16.tflite  # 1024-dim head (초기 모델)  │       └── head_256_fp16.tflite   # 최종 256-dim head (17-class)  │  ├── runs_multi/  │   └── per_class_eval_1024.json   # 클래스별 AUC/AUPR 평가 결과  │  ├── scripts/  │   │  │   ├── train_head_1024.py               # Head 학습 스크립트  │   ├── eval_per_class.py                # AUC/AUPR 평가  │   ├── inspect_yamnet_tflite.py         # TFLite 구조 확인  │   ├── run_yamnet_plus_head_tflite.py   # WAV 파일 분류 실행  │   ├── realtime_infer_mic.py            # 실시간 마이크 추론  │   │  │   └── data/  │       ├── balanced_train_segments.csv  # 학습용 AudioSet 라벨링 파일  │       ├── class_labels_indices.csv     # 521개 원본 레이블 목록  │       └── ontology.json                # AudioSet 레이블 계층 구조  │  ├── requirements.txt               # Python 패키지 목록  └── README.md                      # (현재 문서)   `

🔧 Installation
---------------

### 1) Create Virtual Environment

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   cd YAMNET  python3 -m venv .venv  source .venv/bin/activate   `

### 2) Install Requirements

#### 일반 환경 (TF 사용)

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

#### 라즈베리파이 (TensorFlow Lite Runtime 환경)

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install tflite-runtime  pip install numpy soundfile pyaudio   `

🧪 Usage
--------

### ▶️ 1. WAV 파일 분류

run\_yamnet\_plus\_head\_tflite.py 내부에서 WAV 경로를 수정한 뒤 실행:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python scripts/run_yamnet_plus_head_tflite.py   `

출력 예:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   door         0.002  sink         0.932  <-- 가장 가능성 높은 클래스  microwave    0.010   `

### 🎤 2. 실시간 마이크 스트리밍

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python scripts/realtime_infer_mic.py   `

실행하면 100ms 간격으로 현재 소리를 분류합니다.

예:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Listening...  [ sink ] 0.91  [ footsteps ] 0.07   `

📊 Model Evaluation (AUC / AUPR)
--------------------------------

runs\_multi/per\_class\_eval\_1024.json 파일은 17개 클래스에 대한**AUC(Area Under Curve)** 및**AUPR(Area Under Precision-Recall curve)** 평가 결과입니다.

### Summary

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   {    "num_samples": 22212,    "macro_auc": 0.9898,    "macro_aupr": 0.9705  }   `

*   **macro\_auc ≈ 0.99** → 모델이 전체적으로 매우 잘 분류함
    
*   **macro\_aupr ≈ 0.97** → 클래스 불균형에도 뛰어난 성능
    
*   17개 클래스 모두 AUC 0.97~0.998 수준의 우수한 분류 성능 확보
    

### Per-Class Example (일부 발췌)

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   "door": { "auc": 0.9878, "aupr": 0.9579 },  "dishes": { "auc": 0.9897, "aupr": 0.9614 },  "footsteps": { "auc": 0.9788, "aupr": 0.9445 },  "vacuum": { "auc": 0.9980, "aupr": 0.9957 }   `

*   기계음(vacuum, hair\_dryer 등)은 거의 완벽
    
*   난이도 높은 소리(발걸음, 식기류 등)도 0.94~0.97의 높은 성능
    

🧠 Model Architecture
---------------------

### 1) YAMNet (Frozen)

*   Google's YAMNet 구조 유지
    
*   오디오 파형 → 256-dim 임베딩 출력
    

### 2) Custom Head (Trainable)

*   Input: (256,)
    
*   Dense → ReLU → Dropout → Dense → Softmax
    
*   최종 Output: (17,)
    

🚀 Edge Deployment (라즈베리파이)
---------------------------

필요 파일:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   models/yamnet/yamnet-256.tflite  models/head/head_256_fp16.tflite  scripts/realtime_infer_mic.py   `

라즈베리파이 설정:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install tflite-runtime soundfile pyaudio numpy  python realtime_infer_mic.py   `

🧾 License
----------

모델 및 코드는 MIT 라이선스를 따릅니다.YAMNet은 Google Research의 오픈소스를 기반으로 합니다.

🙌 Acknowledgements
-------------------

*   Google YAMNet
    
*   AudioSet Dataset
    
*   TF Lite Team
