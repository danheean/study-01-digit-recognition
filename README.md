# Handwritten Digit Recognition

A handwritten digit recognition application using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

> **Note**: This project was created using Claude Code. See [PROMPT.md](PROMPT.md) for the prompts used to generate this project.

## Features

- Draw digits (0-9) with mouse input
- Real-time digit recognition
- Display prediction confidence and probability distribution
- ~99% accuracy on test data

## Project Structure

```
study-01/
├── desktop_version/
│   ├── CLAUDE.md                    # Claude Code 가이드 (데스크톱)
│   ├── digit_recognizer_pyside.py   # PySide6 데스크톱 앱
│   └── digit_recognizer.py          # Tkinter 레거시 버전
├── web_version/
│   ├── CLAUDE.md                    # Claude Code 가이드 (웹)
│   └── digit_recognizer_web.py      # Gradio 웹 인터페이스
├── digit_model.keras                # 사전 학습된 CNN 모델
└── dist/
    └── DigitRecognizer.app          # macOS 독립 실행 파일
```

## Files

| File | Description |
|------|-------------|
| `desktop_version/digit_recognizer_pyside.py` | PySide6 desktop GUI application |
| `desktop_version/digit_recognizer.py` | Tkinter version (requires system Python) |
| `web_version/digit_recognizer_web.py` | Gradio web interface version |
| `digit_model.keras` | Pre-trained CNN model |
| `dist/DigitRecognizer.app` | Standalone macOS application |

## Requirements

- Python 3.10+
- TensorFlow
- PySide6 (for desktop version)
- Gradio (for web version)
- Pillow
- NumPy

## Installation

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install tensorflow pillow numpy PySide6
```

## Usage

### Desktop Application (PySide6)

```bash
python desktop_version/digit_recognizer_pyside.py
```

### Web Application (Gradio)

```bash
uv pip install gradio
python web_version/digit_recognizer_web.py
# Open http://localhost:7860 in browser
```

### Standalone macOS App

Double-click `dist/DigitRecognizer.app` or install to Applications:

```bash
cp -r dist/DigitRecognizer.app /Applications/
```

## Model Architecture

```
Input (28x28x1)
    |
Conv2D (32 filters, 3x3) + ReLU
    |
MaxPooling2D (2x2)
    |
Conv2D (64 filters, 3x3) + ReLU
    |
MaxPooling2D (2x2)
    |
Conv2D (64 filters, 3x3) + ReLU
    |
Flatten
    |
Dropout (0.5)
    |
Dense (64) + ReLU
    |
Dense (10) + Softmax
    |
Output (0-9 probabilities)
```

## Building macOS App

```bash
uv pip install pyinstaller
pyinstaller --name "DigitRecognizer" --windowed --onedir --add-data "digit_model.keras:." desktop_version/digit_recognizer_pyside.py
```

## License

MIT
