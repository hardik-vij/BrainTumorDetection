import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load your model
model = load_model("BrainTumordet10.h5")

# Set up Streamlit app
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to detect if there is a brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to a byte array, then load it with cv2
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # Reads as a color image
    
    # Display the uploaded image
    st.image(image, caption="Uploaded MRI Image", channels="BGR", use_column_width=True)
    
    # Resize the image to match the model's input shape, if known (e.g., (32, 32))
    # Replace (32, 32) with the input size the model was trained on
    img_resized = cv2.resize(image, (32, 32))  # Resize to the model's expected input size
    img_normalized = img_resized / 255.0  # Normalize pixel values
    img_reshaped = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    
    st.write("Classifying...")

    try:
        # Run the model prediction
        prediction = model.predict(img_reshaped)
        if prediction[0][0] > 0.5:
            st.write("Prediction: Positive for Brain Tumor")
        else:
            st.write("Prediction: Negative for Brain Tumor")
    except ValueError as e:
        st.error(f"Prediction error: {e}")

else:
    st.write("Please upload an image to classify.")
