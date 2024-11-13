import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load the model
model = load_model("BrainTumordet10.h5")

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an MRI scan to detect if there's a tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI scan', use_column_width=True)

    # Preprocess the image
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))  # Adjust size as per model input
    img_resized = img_resized / 255.0  # Normalize if necessary
    img_reshaped = np.expand_dims(img_resized, axis=0)

    # Make a prediction
    prediction = model.predict(img_reshaped)

    # Display results
    st.write("Prediction:")
    if prediction[0] > 0.5:
        st.write("Tumor detected")
    else:
        st.write("No tumor detected")
