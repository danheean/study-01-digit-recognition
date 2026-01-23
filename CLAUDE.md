# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Handwritten digit recognition application using a CNN trained on MNIST. Offers desktop (PySide6), web (Gradio), and legacy (Tkinter) interfaces. Can be packaged as a standalone macOS app.

## Commands

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv pip install tensorflow pillow numpy PySide6 gradio pyinstaller

# Run desktop app (primary)
python desktop_version/digit_recognizer_pyside.py

# Run web app
python web_version/digit_recognizer_web.py  # opens http://localhost:7860

# Build macOS standalone app
pyinstaller --name "DigitRecognizer" --windowed --onedir --add-data "digit_model.keras:." desktop_version/digit_recognizer_pyside.py
```

## Architecture

**Key Files:**
- `desktop_version/digit_recognizer_pyside.py` - Main desktop app (PySide6 GUI)
- `web_version/digit_recognizer_web.py` - Web interface (Gradio)
- `digit_model.keras` - Pre-trained CNN model (~99% accuracy)

**ML Pipeline:**
1. `DigitRecognitionModel` class handles model building, training, and inference
2. CNN: 3 Conv2D layers → Dropout → Dense(64) → Dense(10, softmax)
3. Trained on MNIST (60k images, 5 epochs)

**Image Preprocessing (critical for accuracy):**
Canvas drawing → grayscale → find bounding box → add padding → crop to square → resize to 28×28 → invert colors → normalize [0,1]

**GUI Architecture:**
- `DrawingCanvas` - handles mouse input and drawing
- `MainWindow` - manages layout and prediction display
- Real-time prediction triggers on `on_drawing_changed()` during mouse movement

**Model Bundling:**
PyInstaller bundles `digit_model.keras` via `--add-data`. Desktop app uses `get_resource_path()` to locate model in both script and bundled modes.

## Tech Stack

- **ML**: TensorFlow/Keras
- **Desktop GUI**: PySide6
- **Web GUI**: Gradio
- **Packaging**: PyInstaller
- **Package Manager**: uv
