# Prompts Used to Create This Project

This document contains the prompts used to generate this handwritten digit recognition application with Claude Code.

## Initial Prompt

```
손글씨로 숫자를 입력하면 이것을 인식하는 코드를 만들어서 실행해 줘. 모든 코드와 주석을 영어로 작성해줘
```

(Translation: Create code that recognizes handwritten digit input and run it. Write all code and comments in English.)

## Environment Setup

```
uv로 가상환경을 만들어서 진행하면 어때?
```

(Translation: How about creating a virtual environment with uv?)

## GUI Framework Selection

```
PySide6로 가능할까?
```

(Translation: Is it possible with PySide6?)

## Creating Standalone App

```
이 프로그램을 마우스로 클릭했을 때 생성하게 할 수 있나요?
```

(Translation: Can you make this program launch when clicking with mouse?)

## Bug Fix

```
캔버스 우측에 까만게 모두 하얀색으로 되어야 할 것 같은데
```

(Translation: The black area on the right side of the canvas should be white.)

---

## Quick Recreation Guide

To recreate this project from scratch, use these prompts in order:

1. **Create the digit recognition app:**
   ```
   Create a handwritten digit recognition application using CNN trained on MNIST dataset.
   Use PySide6 for GUI. Include real-time prediction as user draws.
   Write all code and comments in English.
   ```

2. **Setup environment:**
   ```
   Use uv to create virtual environment and install dependencies:
   tensorflow, pillow, numpy, PySide6
   ```

3. **Create standalone macOS app:**
   ```
   Create a standalone macOS app using PyInstaller that can be launched by double-clicking.
   Bundle the trained model inside the app.
   ```

## Key Technical Decisions

- **Framework**: PySide6 (Qt-based, native look on macOS)
- **ML Library**: TensorFlow/Keras
- **Model**: CNN with 3 Conv2D layers
- **Dataset**: MNIST (60,000 training images)
- **Packaging**: PyInstaller for standalone app
- **Environment**: uv for fast Python package management
