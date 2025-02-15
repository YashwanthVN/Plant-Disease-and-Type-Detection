import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Step 1: Load the saved model
loaded_model = tf.keras.models.load_model(r'Model\plant_disease_classification_keras.keras')

# Step 2: Define the dataset directory and get class labels dynamically from folder names
dataset_dir = r'Dataset'

# Get class labels from folder structure (directory names)
class_labels = sorted(os.listdir(dataset_dir))

# Create a dictionary to map the label names to indices
class_indices = {label: idx for idx, label in enumerate(class_labels)}

# Step 3: Preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    # Load the image from the file path
    img = image.load_img(img_path, target_size=target_size)
    # Convert the image to an array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model's input shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the pixel values to the range [0, 1]
    img_array = img_array / 255.0
    return img_array

# Step 4: Create the Streamlit Interface
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱", layout="centered")

# Title and Introduction
st.title("🌱 Plant Disease Detection")
st.markdown("""
Welcome to the **Plant Disease And Type Detection** app! Upload a plant image, and we will predict the **plant type** and **disease state** with confidence.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image of a plant", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Show loading spinner while processing
    with st.spinner('Processing the image...'):
        # Preprocess the image for prediction
        processed_image = preprocess_image(uploaded_file)

        # Step 5: Make a prediction
        predictions = loaded_model.predict(processed_image)

        # Step 6: Interpret the prediction
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class_index]
        confidence_score = predictions[0][predicted_class_index]

        # Split the label into plant type and disease state
        plant_type, disease_state = predicted_label.split("___")

        # Center the content with background and padding
        result_style = """
            <style>
                .result-container {
                    padding: 20px;
                    border-radius: 15px;
                    background-color: #f0f8ff;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                .confidence-container {
                    font-size: 22px;
                    font-weight: bold;
                    color: #333;
                }
                .large-text {
                    font-size: 36px;
                    font-weight: 600;
                    color: #2c3e50;
                    text-align: center;
                }
                .disease-text {
                    font-size: 32px;
                    font-weight: 600;
                    color: #e74c3c;
                    text-align: center;
                }
                .plant-text {
                    font-size: 32px;
                    font-weight: 600;
                    color: #27ae60;
                    text-align: center;
                }
            </style>
        """
        st.markdown(result_style, unsafe_allow_html=True)

        # Display results with background color and larger text
        st.markdown(f"""
            <div class="result-container">
                <div class="plant-text">🔍 {plant_type.capitalize()}</div>
                <div class="disease-text">🦠 {disease_state.capitalize()}</div>
            </div>
        """, unsafe_allow_html=True)

        # Display confidence score with a color indicator
        if confidence_score > 0.85:
            confidence_color = "green"
        elif confidence_score > 0.6:
            confidence_color = "yellow"
        else:
            confidence_color = "red"

        st.markdown(f"""
            <div class="confidence-container" style="color:{confidence_color};">
                Confidence Score: {confidence_score:.2f}
            </div>
        """, unsafe_allow_html=True)



st.markdown("""
---
<div style="text-align: center; font-size: 18px; color: #2c3e50;">
    <p><strong>Made by:</strong></p>
    <p style="font-size: 20px; font-weight: bold; color: #27ae60;">Shreyansh Khandelwal</p>
    <p style="font-size: 20px; font-weight: bold; color: #2980b9;">Yashwanth NV</p>
    <p style="font-size: 20px; font-weight: bold; color: #f39c12;">Aryavart Chandel</p>
    <p style="font-size: 20px; font-weight: bold; color: #e74c3c;">Harshit Notani</p>
    <br>
    <p style="font-size: 14px; color: #95a5a6;">For Plant Disease Detection</p>
    <p><a href="#" style="color: #3498db; text-decoration: none;">Visit Our Website</a></p>
</div>
""", unsafe_allow_html=True)

