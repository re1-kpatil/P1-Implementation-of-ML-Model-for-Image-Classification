import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('fashion_mnist_cnn.h5')

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# App title and description
st.title("Image Classification Project")
st.text("Upload an image of clothing and get its prediction!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((28, 28))  # Resize to match the model's input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = image_array.reshape(1, 28, 28, 1)  # Add batch dimension

    # Predict the class
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display prediction results
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
