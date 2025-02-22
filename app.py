import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

st.title("🌿 Crop Disease Identification using AI")

# Load the trained model
model = tf.keras.models.load_model("plant_identification_model.h5")

# Label Mapping
label_mapping = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Additional medicinal information
medicinal_info = {
    'Healthy': 'No disease detected, plant is healthy!',
    'Powdery': 'Powdery mildew detected. Use fungicides.',
    'Rust': 'Rust disease detected. Remove infected leaves and use treatments.'
}

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array)
    return preprocessed_image

# Perform Prediction
def predict_plant(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    
    # Get the predicted label
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping[predicted_label_index]
    confidence = predictions[0][predicted_label_index]
    
    # Get medicinal information
    medicinal_value = medicinal_info.get(predicted_label, 'No  information available.')

    return predicted_label, confidence, medicinal_value

# OpenCV Webcam Capture
st.subheader("📸 Live Webcam")

# Create a button to start the webcam
if st.button("Open Webcam"):
    cap = cv2.VideoCapture(0)  # Open device webcam (0 for built-in webcam)
    
    # Create a placeholder for displaying the image
    image_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break
        
        # Convert BGR to RGB (OpenCV loads images in BGR format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the webcam feed in Streamlit
        image_placeholder.image(frame, caption="Live Webcam", use_column_width=True)

        # Capture frame on button press
        if st.button("Capture Image"):
            cap.release()
            image = Image.fromarray(frame)
            st.image(image, caption="Captured Image", use_column_width=True)
            
            # Perform prediction
            predicted_label, confidence, medicinal_value = predict_plant(image)

            # Display Results
            st.subheader("Prediction Result")
            st.write(f"**Predicted Class:** {predicted_label}")
            st.write(f"**Confidence Score:** {confidence:.2f}")
            st.write(f"** Information:** {medicinal_value}")

            break  # Stop the loop when image is captured

    cap.release()
