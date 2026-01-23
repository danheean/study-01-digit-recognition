# CLAUDE.md - Web Version

Gradio 기반 웹 인터페이스로 손글씨 숫자 인식 기능을 제공합니다.

## Commands

```bash
# 환경 설정 (프로젝트 루트에서 실행)
uv venv && source .venv/bin/activate
uv pip install tensorflow pillow numpy gradio

# 웹 앱 실행 (프로젝트 루트에서)
python web_version/digit_recognizer_web.py  # http://localhost:7860
```

## Architecture

**주요 파일:**
- `digit_recognizer_web.py` - Gradio 웹 인터페이스
- `digit_model.keras` - 사전 학습된 CNN 모델 (~99% 정확도)

**ML Pipeline:**
1. `DigitRecognitionModel` 클래스가 모델 빌드, 학습, 추론 담당
2. CNN: 3 Conv2D layers → Dropout → Dense(64) → Dense(10, softmax)
3. MNIST로 학습 (60k 이미지, 5 epochs)

**이미지 전처리 (정확도에 중요):**
캔버스 드로잉 → 그레이스케일 → 바운딩 박스 찾기 → 패딩 추가 → 정사각형 크롭 → 28×28 리사이즈 → 색상 반전 → [0,1] 정규화

**웹 GUI:**
- Gradio의 스케치패드/이미지 입력 컴포넌트 사용
- 실시간 예측 결과 표시

## Tech Stack

- **ML**: TensorFlow/Keras
- **Web GUI**: Gradio
- **Package Manager**: uv
