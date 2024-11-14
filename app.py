import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np

# Load your model
model = load_model("BrainTumordet10.h5")

# Set up Streamlit app
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to detect if there is a brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    # Ensure consistent image resizing and check input shape
    img_resized = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)  # Resize to match model input
    img_array = img_to_array(img_resized)
    
    # Convert the image to a numpy array and normalize pixel values
    img_normalized = img_array / 255.0
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
