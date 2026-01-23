# CLAUDE.md - Desktop Version

PySide6 기반 데스크톱 애플리케이션으로 손글씨 숫자 인식 기능을 제공합니다.

## Commands

```bash
# 환경 설정 (프로젝트 루트에서 실행)
uv venv && source .venv/bin/activate
uv pip install tensorflow pillow numpy PySide6 pyinstaller

# 데스크톱 앱 실행 (프로젝트 루트에서)
python desktop_version/digit_recognizer_pyside.py

# macOS 독립 실행 파일 빌드 (프로젝트 루트에서)
pyinstaller --name "DigitRecognizer" --windowed --onedir --add-data "digit_model.keras:." desktop_version/digit_recognizer_pyside.py
```

## Architecture

**주요 파일:**
- `digit_recognizer_pyside.py` - 메인 데스크톱 앱 (PySide6 GUI)
- `digit_model.keras` - 사전 학습된 CNN 모델 (~99% 정확도)

**ML Pipeline:**
1. `DigitRecognitionModel` 클래스가 모델 빌드, 학습, 추론 담당
2. CNN: 3 Conv2D layers → Dropout → Dense(64) → Dense(10, softmax)
3. MNIST로 학습 (60k 이미지, 5 epochs)

**이미지 전처리 (정확도에 중요):**
캔버스 드로잉 → 그레이스케일 → 바운딩 박스 찾기 → 패딩 추가 → 정사각형 크롭 → 28×28 리사이즈 → 색상 반전 → [0,1] 정규화

**GUI Architecture:**
- `DrawingCanvas` - 마우스 입력 및 드로잉 처리
- `MainWindow` - 레이아웃 및 예측 결과 표시 관리
- 마우스 이동 시 `on_drawing_changed()`에서 실시간 예측

**Model Bundling:**
PyInstaller가 `--add-data`로 `digit_model.keras` 번들링. 데스크톱 앱은 `get_resource_path()`로 스크립트 모드와 번들 모드 모두에서 모델 위치 탐색.

## Tech Stack

- **ML**: TensorFlow/Keras
- **Desktop GUI**: PySide6
- **Packaging**: PyInstaller
- **Package Manager**: uv
