from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the trained model
model = tf.keras.models.load_model("malaria_cell_classifier.h5")  # Ensure "model.h5" is in the same folder

# Define class labels
CLASS_NAMES = ["Parasite", "Uninfected"]  # Ensure this matches your model training order

def preprocess_image(image):
    """Convert image to model input format (resize & normalize)"""
    image = image.resize((128, 128))  # Resize to match model input shape
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))  # Read image
    processed_image = preprocess_image(image)  # Preprocess

    # Make prediction
    prediction = model.predict(processed_image)[0][0]
    label = CLASS_NAMES[1] if prediction > 0.5 else CLASS_NAMES[0]  # Adjust if predictions are flipped

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
