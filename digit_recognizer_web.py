"""
Handwritten Digit Recognition Web Application
Uses a CNN trained on MNIST dataset with Gradio web interface
"""

import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
import gradio as gr


class DigitRecognitionModel:
    """CNN model for handwritten digit recognition"""

    def __init__(self, model_path='digit_model.keras'):
        self.model_path = model_path
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


def preprocess_image(image_data):
    """Preprocess drawn image for model input"""
    if image_data is None:
        return None

    # Get the composite image from Gradio sketchpad
    if isinstance(image_data, dict):
        img = image_data.get('composite')
        if img is None:
            return None
    else:
        img = image_data

    # Convert to PIL Image if numpy array
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))

    # Convert to grayscale
    img = img.convert('L')

    # Get bounding box of the drawing
    img_array = np.array(img)

    # Find non-white pixels (drawing content)
    non_white = np.where(img_array < 250)

    if len(non_white[0]) == 0:
        return None

    # Get bounding box
    top, bottom = non_white[0].min(), non_white[0].max()
    left, right = non_white[1].min(), non_white[1].max()

    # Add padding
    padding = 20
    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(img.height, bottom + padding)
    right = min(img.width, right + padding)

    # Crop to bounding box
    img = img.crop((left, top, right, bottom))

    # Make it square
    width, height = img.size
    max_dim = max(width, height)
    square_img = Image.new('L', (max_dim, max_dim), 255)
    offset = ((max_dim - width) // 2, (max_dim - height) // 2)
    square_img.paste(img, offset)

    # Resize to 28x28
    img = square_img.resize((28, 28), Image.Resampling.LANCZOS)

    # Invert colors (MNIST has white digits on black background)
    img = ImageOps.invert(img)

    # Convert to numpy array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


# Initialize model
print("Initializing digit recognition model...")
model = DigitRecognitionModel()

if not model.load():
    print("No pre-trained model found. Training new model...")
    model.train(epochs=5)


def recognize_digit(image_data):
    """Recognize digit from sketchpad input"""
    if image_data is None:
        return "Please draw a digit", {}

    img_array = preprocess_image(image_data)

    if img_array is None:
        return "Please draw a digit", {}

    predictions = model.predict(img_array)
    predicted_digit = int(np.argmax(predictions))
    confidence = float(predictions[predicted_digit])

    # Create confidence dictionary for all digits
    confidences = {str(i): float(predictions[i]) for i in range(10)}

    result_text = f"Predicted: {predicted_digit} (Confidence: {confidence:.1%})"

    return result_text, confidences


# Create Gradio interface
with gr.Blocks(title="Handwritten Digit Recognition") as demo:
    gr.Markdown("# Handwritten Digit Recognition")
    gr.Markdown("Draw a digit (0-9) in the canvas below and see the prediction!")

    with gr.Row():
        with gr.Column(scale=1):
            sketchpad = gr.Sketchpad(
                label="Draw a digit here",
                brush=gr.Brush(default_size=12, colors=["#000000"]),
                canvas_size=(280, 280),
                layers=False
            )
            clear_btn = gr.ClearButton(sketchpad, value="Clear Canvas")

        with gr.Column(scale=1):
            result_text = gr.Textbox(label="Result", interactive=False)
            confidence_bars = gr.Label(label="Confidence for each digit", num_top_classes=10)

    # Real-time prediction on drawing
    sketchpad.change(
        fn=recognize_digit,
        inputs=[sketchpad],
        outputs=[result_text, confidence_bars]
    )

    gr.Markdown("---")
    gr.Markdown("This model uses a CNN trained on the MNIST dataset with ~99% accuracy.")


if __name__ == "__main__":
    print("\nStarting web application...")
    print("Open your browser and go to the URL shown below.\n")
    demo.launch()
