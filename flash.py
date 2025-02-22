from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import cv2
import os

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend communication

# Load the trained model
MODEL_PATH = "plant_identification_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Label mapping
label_mapping = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array)
    return preprocessed_image

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files["image"]
    image_path = "temp.jpg"
    image_file.save(image_path)
    
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping[predicted_label_index]
    confidence = float(predictions[0][predicted_label_index])
    
    os.remove(image_path)  # Clean up
    
    return jsonify({"prediction": predicted_label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
