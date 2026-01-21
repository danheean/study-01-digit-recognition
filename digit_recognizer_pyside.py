"""
Handwritten Digit Recognition Application using PySide6
Uses a CNN trained on MNIST dataset with Qt-based GUI
"""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QFrame, QGridLayout
)
from PySide6.QtCore import Qt, QPoint, QSize
from PySide6.QtGui import QPainter, QPen, QImage, QPixmap, QFont, QColor


def get_resource_path(filename):
    """Get the correct path for bundled resources"""
    if getattr(sys, 'frozen', False):
        # Running as bundled app
        base_path = sys._MEIPASS
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)


class DigitRecognitionModel:
    """CNN model for handwritten digit recognition"""

    def __init__(self, model_path='digit_model.keras'):
        self.model_path = get_resource_path(model_path)
        self.model = None

    def build_model(self):
        """Build the CNN architecture"""
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, epochs=5):
        """Train the model on MNIST dataset"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        print("Building model...")
        self.model = self.build_model()

        print(f"Training model for {epochs} epochs...")
        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=128,
            validation_split=0.1,
            verbose=1
        )

        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")

        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        return test_acc

    def load(self):
        """Load a pre-trained model"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        return False

    def predict(self, image_array):
        """Predict digit from preprocessed image array"""
        if self.model is None:
            raise ValueError("Model not loaded")

        if image_array.shape != (1, 28, 28, 1):
            image_array = image_array.reshape(1, 28, 28, 1)

        predictions = self.model.predict(image_array, verbose=0)
        return predictions[0]


class DrawingCanvas(QWidget):
    """Custom widget for drawing digits"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)
        self.setAutoFillBackground(True)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(QColor(255, 255, 255))
        self.drawing = False
        self.brush_size = 15
        self.last_point = QPoint()

    def paintEvent(self, event):
        """Paint the canvas"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        painter.drawImage(self.rect(), self.image)

    def mousePressEvent(self, event):
        """Handle mouse press"""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        """Handle mouse movement for drawing"""
        if self.drawing and event.buttons() & Qt.LeftButton:
            painter = QPainter(self.image)
            pen = QPen(Qt.black, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            current_point = event.position().toPoint()
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()

            # Emit signal to parent for real-time prediction
            if self.parent() and hasattr(self.parent(), 'on_drawing_changed'):
                self.parent().on_drawing_changed()

    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        """Clear the canvas"""
        self.image.fill(QColor(255, 255, 255))
        self.update()

    def get_image_array(self):
        """Convert canvas to numpy array for model input"""
        # Convert QImage to PIL Image
        width = self.image.width()
        height = self.image.height()

        # Get image data
        ptr = self.image.bits()
        arr = np.array(ptr).reshape(height, width, 4)  # BGRA format

        # Convert to grayscale using PIL
        pil_img = Image.fromarray(arr[:, :, :3][:, :, ::-1])  # BGR to RGB
        pil_img = pil_img.convert('L')

        # Find bounding box of the drawing
        img_array = np.array(pil_img)
        non_white = np.where(img_array < 250)

        if len(non_white[0]) == 0:
            return None

        # Get bounding box with padding
        top, bottom = non_white[0].min(), non_white[0].max()
        left, right = non_white[1].min(), non_white[1].max()

        padding = 20
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(height, bottom + padding)
        right = min(width, right + padding)

        # Crop to bounding box
        pil_img = pil_img.crop((left, top, right, bottom))

        # Make it square
        w, h = pil_img.size
        max_dim = max(w, h)
        square_img = Image.new('L', (max_dim, max_dim), 255)
        offset = ((max_dim - w) // 2, (max_dim - h) // 2)
        square_img.paste(pil_img, offset)

        # Resize to 28x28
        pil_img = square_img.resize((28, 28), Image.Resampling.LANCZOS)

        # Invert colors (MNIST has white digits on black background)
        pil_img = ImageOps.invert(pil_img)

        # Convert to numpy array and normalize
        img_array = np.array(pil_img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.setWindowTitle("Handwritten Digit Recognition")
        self.setFixedSize(550, 450)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left side - Drawing area
        left_layout = QVBoxLayout()

        title_label = QLabel("Draw a digit (0-9)")
        title_label.setFont(QFont("Helvetica", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)

        # Canvas with border
        self.canvas = DrawingCanvas(self)
        self.canvas.setStyleSheet("background-color: white; border: 2px solid gray;")
        left_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)

        # Buttons
        button_layout = QHBoxLayout()

        self.recognize_btn = QPushButton("Recognize")
        self.recognize_btn.setFixedHeight(35)
        self.recognize_btn.clicked.connect(self.recognize)
        button_layout.addWidget(self.recognize_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedHeight(35)
        self.clear_btn.clicked.connect(self.clear_canvas)
        button_layout.addWidget(self.clear_btn)

        left_layout.addLayout(button_layout)
        main_layout.addLayout(left_layout)

        # Right side - Results
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # Prediction label
        pred_title = QLabel("Prediction:")
        pred_title.setFont(QFont("Helvetica", 12))
        pred_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(pred_title)

        self.prediction_label = QLabel("-")
        self.prediction_label.setFont(QFont("Helvetica", 72, QFont.Bold))
        self.prediction_label.setStyleSheet("color: #0066cc;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.prediction_label)

        # Confidence label
        conf_title = QLabel("Confidence:")
        conf_title.setFont(QFont("Helvetica", 12))
        conf_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(conf_title)

        self.confidence_label = QLabel("-")
        self.confidence_label.setFont(QFont("Helvetica", 24))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.confidence_label)

        # Probability bars
        prob_title = QLabel("All Probabilities:")
        prob_title.setFont(QFont("Helvetica", 10))
        right_layout.addWidget(prob_title)

        prob_grid = QGridLayout()
        prob_grid.setSpacing(5)

        self.prob_bars = []
        self.prob_labels = []

        for i in range(10):
            digit_label = QLabel(f"{i}:")
            digit_label.setFixedWidth(20)
            prob_grid.addWidget(digit_label, i, 0)

            bar = QProgressBar()
            bar.setFixedHeight(18)
            bar.setFixedWidth(100)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            prob_grid.addWidget(bar, i, 1)

            value_label = QLabel("0%")
            value_label.setFixedWidth(40)
            prob_grid.addWidget(value_label, i, 2)

            self.prob_bars.append(bar)
            self.prob_labels.append(value_label)

        right_layout.addLayout(prob_grid)
        right_layout.addStretch()

        main_layout.addLayout(right_layout)

        # Instructions
        instructions = QLabel("Draw a digit and click 'Recognize' or just draw for real-time predictions")
        instructions.setStyleSheet("color: gray; font-size: 9px;")
        instructions.setWordWrap(True)

    def on_drawing_changed(self):
        """Called when drawing changes for real-time prediction"""
        self.recognize()

    def recognize(self):
        """Recognize the drawn digit"""
        img_array = self.canvas.get_image_array()

        if img_array is None:
            return

        predictions = self.model.predict(img_array)
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit]) * 100

        # Update UI
        self.prediction_label.setText(str(predicted_digit))
        self.confidence_label.setText(f"{confidence:.1f}%")

        # Update probability bars
        for i in range(10):
            prob = int(predictions[i] * 100)
            self.prob_bars[i].setValue(prob)
            self.prob_labels[i].setText(f"{prob}%")

    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.clear()
        self.prediction_label.setText("-")
        self.confidence_label.setText("-")

        for i in range(10):
            self.prob_bars[i].setValue(0)
            self.prob_labels[i].setText("0%")


def main():
    """Main entry point"""
    print("Initializing digit recognition model...")
    model = DigitRecognitionModel()

    if not model.load():
        print("No pre-trained model found. Training new model...")
        model.train(epochs=5)

    print("\nStarting PySide6 GUI application...")

    app = QApplication(sys.argv)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
