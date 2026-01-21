# Handwritten Digit Recognition

A handwritten digit recognition application using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

> **Note**: This project was created using Claude Code. See [PROMPT.md](PROMPT.md) for the prompts used to generate this project.

## Features

- Draw digits (0-9) with mouse input
- Real-time digit recognition
- Display prediction confidence and probability distribution
- ~99% accuracy on test data

## Files

| File | Description |
|------|-------------|
| `digit_recognizer_pyside.py` | PySide6 desktop GUI application |
| `digit_recognizer_web.py` | Gradio web interface version |
| `digit_recognizer.py` | Tkinter version (requires system Python) |
| `digit_model.keras` | Pre-trained CNN model |
| `DigitRecognizer.app` | Standalone macOS application (in `dist/`) |

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
python digit_recognizer_pyside.py
```

### Web Application (Gradio)

```bash
uv pip install gradio
python digit_recognizer_web.py
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
pyinstaller --name "DigitRecognizer" --windowed --onedir --add-data "digit_model.keras:." digit_recognizer_pyside.py
```

## License

MIT
