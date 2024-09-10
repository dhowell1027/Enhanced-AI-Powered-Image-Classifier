# Install dependencies before running: 
# pip install tensorflow streamlit matplotlib pillow plotly opencv-python

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import plotly.express as px
import cv2

# Load models: I chose MobileNetV2 and ResNet50 for a balance of speed and accuracy
# MobileNetV2 is lighter and faster, while ResNet50 is more accurate on complex images.
model_options = {
    "MobileNetV2": tf.keras.applications.MobileNetV2(weights='imagenet'),
    "ResNet50": tf.keras.applications.ResNet50(weights='imagenet')
}

# Sidebar option: Let the user decide which model to use
st.sidebar.title("Model Options")
model_choice = st.sidebar.selectbox("Choose a model", list(model_options.keys()), key="model_selector")
model = model_options[model_choice]

# This function resizes and preprocesses the image based on the chosen model
# MobileNetV2 expects different preprocessing compared to ResNet50
def preprocess_image(image, target_size=(224, 224)):
    # Ensure the image is in RGB format (even if it was converted to grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    if model_choice == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    else:
        return tf.keras.applications.resnet50.preprocess_input(img_array)

# The decode_predictions function interprets the model output into human-readable labels
# I found this useful because raw outputs from neural networks can be hard to interpret
def decode_predictions(predictions):
    if model_choice == "MobileNetV2":
        decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    else:
        decoded_preds = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)[0]
    labels = [item[1] for item in decoded_preds]
    scores = [item[2] for item in decoded_preds]
    return labels, scores

# This function implements Grad-CAM to visualize which parts of the image the model focuses on.
# Grad-CAM is critical for understanding why the model makes certain decisions.
def grad_cam(input_image, model, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    conv_outputs = conv_outputs * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# This function applies the Grad-CAM heatmap onto the original image.
# It's a great way to visualize what the neural network "sees."
def apply_gradcam(image, model, layer_name='block5_conv3'):
    img_array = preprocess_image(image)
    heatmap = grad_cam(img_array, model, layer_name)
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed_img)

# Sidebar for image filters: These were added to let the user explore how image processing affects predictions
st.sidebar.title("Image Filters")
apply_gray = st.sidebar.checkbox("Apply Grayscale", key="apply_gray")

# Add a slider to control blur intensity using Gaussian blur
blur_intensity = st.sidebar.slider("Blur Intensity (Gaussian)", min_value=0, max_value=10, value=0, step=1, key="blur_intensity")

# Initialize rotation state in Streamlit session
if "rotation_angle" not in st.session_state:
    st.session_state.rotation_angle = 0

# Button to rotate the image 90 degrees at a time
if st.sidebar.button("Rotate 90° Clockwise"):
    st.session_state.rotation_angle += 90
    st.session_state.rotation_angle %= 360  # Keep the angle between 0 and 360

# Main title for the app
st.title('Enhanced AI-Powered Image Classifier')
st.write("Upload an image, and the AI will predict what's in it! This app uses state-of-the-art models to classify images and explain its decisions using Grad-CAM.")

# Option to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="file_uploader")

# Checking for uploaded file
if uploaded_file:
    image = Image.open(uploaded_file)

    # Apply rotation
    if st.session_state.rotation_angle != 0:
        image = image.rotate(-st.session_state.rotation_angle)  # Rotate counterclockwise

    # Apply filters to the uploaded image
    if apply_gray:
        image = image.convert('L')  # Apply grayscale filter
    if blur_intensity > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur_intensity))  # Apply Gaussian blur filter with intensity

    # Ensure the image is in RGB format before passing to the model
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption='Filtered Image', use_column_width=True)

    # Preprocess the image and get model predictions
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    labels, scores = decode_predictions(predictions)
    
    # Display predictions in an interactive plot
    st.write("### Top Predictions:")
    fig = px.bar(x=scores, y=labels, orientation='h', labels={'x': 'Probability', 'y': 'Prediction'})
    st.plotly_chart(fig)
    
    # Grad-CAM visualization: This shows where the model "looks" in the image to make its prediction
    st.write("### Grad-CAM Explainability:")
    gradcam_image = apply_gradcam(image, model, layer_name='conv5_block3_out' if model_choice == "ResNet50" else 'block_16_project')
    st.image(gradcam_image, caption="Grad-CAM Explanation", use_column_width=True)
    
    # Also show predictions using a Matplotlib bar graph, just to provide different visual styles
    fig, ax = plt.subplots()
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)
