import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the model
model = load_model("BrainTumordet10.h5")

st.title("Brain Tumor Detection")
st.write("Upload an MRI scan to detect if there's a tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI scan', use_column_width=True)

    # Resize and preprocess the image for model prediction
    img = image.resize((224, 224))  # Resize as per model requirements
    img_array = img_to_array(img)  # Convert to numpy array
    img_normalized = img_array / 255.0  # Scale pixel values if required by the model
    img_reshaped = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_reshaped)

    # Display results
    st.write("Prediction:")
    if prediction[0][0] > 0.5:
        st.write("Tumor detected")
    else:
        st.write("No tumor detected")
