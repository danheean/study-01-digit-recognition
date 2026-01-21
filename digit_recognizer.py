"""
Handwritten Digit Recognition Application
Uses a Convolutional Neural Network (CNN) trained on the MNIST dataset
Provides a GUI for drawing digits and real-time recognition
"""

import os
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import threading


class DigitRecognitionModel:
    """CNN model for handwritten digit recognition"""

    def __init__(self, model_path='digit_model.keras'):
        self.model_path = model_path
        self.model = None

    def build_model(self):
        """Build the CNN architecture"""
        model = keras.Sequential([
            # Input layer - MNIST images are 28x28 grayscale
            layers.Input(shape=(28, 28, 1)),

            # First convolutional block
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Second convolutional block
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Third convolutional block
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),

            # Flatten and dense layers
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

        # Normalize and reshape data
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

        # Evaluate on test set
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")

        # Save the model
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
            raise ValueError("Model not loaded. Train or load a model first.")

        # Ensure correct shape
        if image_array.shape != (1, 28, 28, 1):
            image_array = image_array.reshape(1, 28, 28, 1)

        predictions = self.model.predict(image_array, verbose=0)
        return predictions[0]


class DrawingCanvas:
    """GUI application for drawing digits and recognition"""

    def __init__(self, model):
        self.model = model
        self.canvas_size = 280  # 10x the MNIST size for easier drawing
        self.brush_size = 15

        # Create main window
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognition")
        self.root.resizable(False, False)

        # Create PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Draw a digit (0-9)",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Canvas for drawing
        self.canvas = tk.Canvas(
            main_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            cursor='cross',
            highlightthickness=2,
            highlightbackground='gray'
        )
        self.canvas.grid(row=1, column=0, padx=(0, 20))

        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        # Right panel for results
        result_frame = ttk.Frame(main_frame)
        result_frame.grid(row=1, column=1, sticky=(tk.N, tk.S))

        # Prediction label
        pred_title = ttk.Label(
            result_frame,
            text="Prediction:",
            font=('Helvetica', 12)
        )
        pred_title.grid(row=0, column=0, pady=(0, 5))

        self.prediction_label = ttk.Label(
            result_frame,
            text="-",
            font=('Helvetica', 72, 'bold'),
            foreground='blue'
        )
        self.prediction_label.grid(row=1, column=0, pady=(0, 20))

        # Confidence label
        conf_title = ttk.Label(
            result_frame,
            text="Confidence:",
            font=('Helvetica', 12)
        )
        conf_title.grid(row=2, column=0, pady=(0, 5))

        self.confidence_label = ttk.Label(
            result_frame,
            text="-",
            font=('Helvetica', 24)
        )
        self.confidence_label.grid(row=3, column=0, pady=(0, 20))

        # Probability bars
        prob_title = ttk.Label(
            result_frame,
            text="All Probabilities:",
            font=('Helvetica', 10)
        )
        prob_title.grid(row=4, column=0, pady=(10, 5))

        self.prob_bars = []
        self.prob_labels = []

        for i in range(10):
            prob_frame = ttk.Frame(result_frame)
            prob_frame.grid(row=5+i, column=0, sticky=(tk.W, tk.E), pady=1)

            digit_label = ttk.Label(prob_frame, text=f"{i}:", width=2)
            digit_label.pack(side=tk.LEFT)

            bar = ttk.Progressbar(
                prob_frame,
                length=100,
                mode='determinate'
            )
            bar.pack(side=tk.LEFT, padx=(5, 5))

            value_label = ttk.Label(prob_frame, text="0%", width=5)
            value_label.pack(side=tk.LEFT)

            self.prob_bars.append(bar)
            self.prob_labels.append(value_label)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        self.recognize_btn = ttk.Button(
            button_frame,
            text="Recognize",
            command=self.recognize
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Draw a digit in the canvas and click 'Recognize' or just draw to see real-time predictions",
            font=('Helvetica', 9),
            foreground='gray'
        )
        instructions.grid(row=3, column=0, columnspan=2, pady=(10, 0))

    def paint(self, event):
        """Handle mouse painting on canvas"""
        x1 = event.x - self.brush_size
        y1 = event.y - self.brush_size
        x2 = event.x + self.brush_size
        y2 = event.y + self.brush_size

        # Draw on canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')

        # Draw on PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

        # Real-time prediction (debounced)
        self.root.after(100, self.recognize)

    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Reset labels
        self.prediction_label.config(text="-")
        self.confidence_label.config(text="-")

        for i in range(10):
            self.prob_bars[i]['value'] = 0
            self.prob_labels[i].config(text="0%")

    def preprocess_image(self):
        """Preprocess the drawn image for model input"""
        # Resize to 28x28
        img = self.image.copy()

        # Find bounding box of the drawing
        bbox = img.getbbox()

        if bbox is None:
            # Empty canvas
            return None

        # Crop to bounding box with padding
        padding = 20
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(self.canvas_size, bbox[2] + padding)
        bottom = min(self.canvas_size, bbox[3] + padding)

        img = img.crop((left, top, right, bottom))

        # Make it square
        width, height = img.size
        max_dim = max(width, height)
        square_img = Image.new('L', (max_dim, max_dim), 'white')
        offset = ((max_dim - width) // 2, (max_dim - height) // 2)
        square_img.paste(img, offset)

        # Resize to 28x28 (MNIST size)
        img = square_img.resize((28, 28), Image.Resampling.LANCZOS)

        # Invert colors (MNIST has white digits on black background)
        img = ImageOps.invert(img)

        # Convert to numpy array and normalize
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array

    def recognize(self):
        """Recognize the drawn digit"""
        img_array = self.preprocess_image()

        if img_array is None:
            return

        # Get prediction
        predictions = self.model.predict(img_array)
        predicted_digit = np.argmax(predictions)
        confidence = predictions[predicted_digit] * 100

        # Update UI
        self.prediction_label.config(text=str(predicted_digit))
        self.confidence_label.config(text=f"{confidence:.1f}%")

        # Update probability bars
        for i in range(10):
            prob = predictions[i] * 100
            self.prob_bars[i]['value'] = prob
            self.prob_labels[i].config(text=f"{prob:.0f}%")

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    model = DigitRecognitionModel()

    # Try to load existing model, otherwise train new one
    if not model.load():
        print("No pre-trained model found. Training new model...")
        print("=" * 50)
        model.train(epochs=5)
        print("=" * 50)

    print("\nStarting GUI application...")
    print("Draw a digit (0-9) in the canvas and see the prediction!")
    print("Close the window to exit.\n")

    # Create and run GUI
    app = DrawingCanvas(model)
    app.run()


if __name__ == "__main__":
    main()
