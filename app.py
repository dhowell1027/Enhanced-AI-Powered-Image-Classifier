# Install dependencies before running: 
# pip install tensorflow streamlit matplotlib pillow plotly

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px

# Load a pre-trained model (MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to the size MobileNetV2 expects
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Function to decode predictions and format them
def decode_predictions(predictions):
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    labels = [item[1] for item in decoded_preds]
    scores = [item[2] for item in decoded_preds]
    return labels, scores

# Streamlit app
st.title('AI-Powered Image Classifier')
st.write("Upload an image, and the AI will predict what's in it!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Preprocess the image for model input
    img_array = preprocess_image(image)
    
    # Get predictions from the model
    predictions = model.predict(img_array)
    labels, scores = decode_predictions(predictions)
    
    # Display the prediction results
    st.write("### Top Predictions:")
    
    # Display the predictions using Plotly
    fig = px.bar(x=scores, y=labels, orientation='h', labels={'x': 'Probability', 'y': 'Prediction'})
    st.plotly_chart(fig)

    # Also display the raw prediction scores using Matplotlib for variety
    fig, ax = plt.subplots()
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

