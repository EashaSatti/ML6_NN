import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved model
model = keras.models.load_model('mnist_digit_classifier.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((28, 28))  # Resize to 28x28
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize
    image = image.reshape(1, 28 * 28)  # Flatten
    return image

# Streamlit app title
st.title("MNIST Digit Classifier")

# Upload an image
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for prediction
    processed_image = preprocess_image(image)
    
    # Predict the digit using the model
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    
    # Display the predicted label
    st.write(f"Predicted Digit: {predicted_label}")
    
    # Display the confidence scores (optional)
    st.bar_chart(prediction[0])
