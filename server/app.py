# server/app.py
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model(r'Model\plant_disease_classification_keras.keras')

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    r'D:\Yashwanth\College\study\Third Year\5th Sem\ML\Project\Dataset',  # Replace with your training data directory
    target_size=(224, 224),   # Adjust based on your input size
    batch_size=32,            # Batch size
    class_mode='categorical'  # or 'binary' for binary classification
)

def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction[0])
    class_labels = train_generator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}
    
    predicted_label = class_labels[predicted_class_index]
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence Score: {prediction[0][predicted_class_index]:.2f}")
    
    return jsonify({'Predicted Class:': str(predicted_label)}, {'Confidence Score:': float(prediction[0][predicted_class_index]:.2f)})

if __name__ == '__main__':
    app.run(debug=True)
