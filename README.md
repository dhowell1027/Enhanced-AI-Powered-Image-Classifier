# Enhanced AI-Powered Image Classifier with Grad-CAM

## Overview

This project is an advanced image classification tool that uses cutting-edge deep learning models, including MobileNetV2, ResNet50, and EfficientNetB0, to classify images with real-time explainability. By leveraging Grad-CAM (Gradient-weighted Class Activation Mapping), the app provides a visual explanation of the AI’s decision-making process, highlighting the areas of the image that influenced the model's predictions.

The application is designed to be flexible and user-friendly, allowing users to experiment with different models, apply filters, and analyze model predictions visually.

## Features

- **Model Selection**: Choose between multiple state-of-the-art pre-trained models:
  - **MobileNetV2** for fast and lightweight predictions.
  - **ResNet50** for more accurate, but computationally heavier classifications.
  - **EfficientNetB0** for a balance between performance and accuracy.
  
- **Grad-CAM Explainability**: Visualize which parts of the image the model focused on to make its predictions by generating heatmaps.

- **Image Filters**: Experiment with preprocessing by applying grayscale or Gaussian blur filters.

- **Image Rotation**: Rotate images in 90-degree increments to see how orientation impacts classification results.

- **Interactive Visualizations**: Predictions are presented with dynamic bar charts, making it easy to interpret the model’s confidence in its classifications.

## What's New?

This version introduces several new features and improvements:
- **Support for More Models**: Added support for MobileNetV2, ResNet50, and EfficientNetB0, providing flexibility in model choice based on speed and accuracy needs.
- **Grad-CAM for All Models**: Grad-CAM explainability works across all selected models, providing transparency into the AI’s decision-making process.
- **Improved Image Filters**: Now includes a slider to control the intensity of the Gaussian blur filter.
- **Image Rotation**: Rotate images using a simple button interface that rotates the image in 90-degree increments.

## How It Works

1. **Upload an Image**: Upload an image from your local machine. The app supports common image formats such as JPG and PNG.
   
2. **Select a Model**: From the sidebar, choose one of the available models. MobileNetV2 is optimized for speed, while ResNet50 and EfficientNetB0 offer better accuracy for complex images.

3. **Apply Filters (Optional)**: Use the sidebar to apply grayscale or Gaussian blur filters to preprocess the image before classification.

4. **Rotate the Image (Optional)**: Use the "Rotate 90°" button to rotate the image in 90-degree intervals.

5. **View Predictions**: The app will display the top predictions from the selected model, along with the probability scores. The results are shown in interactive bar charts.

6. **Understand the Model’s Focus**: Grad-CAM explainability highlights the important regions of the image that contributed to the model's predictions with a heatmap overlay.

## Setup and Installation

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)

### Installation
**Clone the Repository**:

   git clone https://github.com/dhowell1027/Enhanced-AI-Powered-Image-Classifier
Navigate to the Project Directory:

cd enhanced-ai-image-classifier

Install the Dependencies:

pip install -r requirements.txt

Run the Application:


    streamlit run app.py

    The app will open in your browser at http://localhost:8501.

Using the Classifier

    Upload an Image: Choose an image from your local machine.
    Model Selection: Select a model from MobileNetV2, ResNet50, or EfficientNetB0.
    Apply Filters: Apply optional filters like grayscale or blur.
    Rotate the Image: Use the button to rotate the image in 90-degree intervals.
    View Predictions: See the top predictions along with probability scores in an interactive bar chart.
    Grad-CAM Explanation: View a heatmap that highlights the important regions the model focused on for its prediction.

Example Use Cases

    Model Interpretability: Use Grad-CAM to understand why a model made a particular prediction and visualize how the model "sees" the image.

    Experimentation with Pre-trained Models: Quickly test different pre-trained models on various images and evaluate their performance.

    Educational Tool: Ideal for demonstrating how deep learning models process images and how different preprocessing steps (e.g., filters and rotations) can affect model predictions.

Technologies Used

    TensorFlow: For loading and managing pre-trained deep learning models like MobileNetV2, ResNet50, and EfficientNetB0.
    Streamlit: A fast and interactive web framework for creating the user interface.
    Pillow: For image manipulation (applying filters, rotating images).
    OpenCV: For image processing, used in Grad-CAM implementation.
    Plotly & Matplotlib: For visualizing prediction results through interactive and static plots.

DevContainer Setup

The project includes a devcontainer.json file to help you set up a development environment using GitHub Codespaces or VS Code Remote Containers.
DevContainer Features

    Python 3.10: The container comes pre-configured with Python 3.10, compatible with all required dependencies.
    VSCode Extensions: The container includes helpful extensions for Python development, such as Jupyter and Pylance.
    Automatic Dependency Installation: All Python dependencies are automatically installed from requirements.txt upon creating the container.
    Docker-in-Docker Support: The container includes support for Docker-in-Docker in case you need to run containers within your development environment.

Future Plans

Here are some planned features for future releases:

    Support for Additional Models: Add new models like Vision Transformers and InceptionV3.
    More Image Filters: Introduce advanced image transformations, such as edge detection and sharpen filters.
    Custom Model Uploads: Allow users to upload and test custom-trained models.

Contributing

Contributions are welcome! If you'd like to contribute, follow these steps:

    Fork the repository.
    Create a feature branch:

    

git checkout -b feature/my-new-feature

Commit your changes:



git commit -m 'Add new feature'

Push to the branch:



    git push origin feature/my-new-feature

    Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.
Final Thoughts

This Enhanced AI-Powered Image Classifier is designed to be a powerful yet accessible tool for experimenting with image classification and model explainability. Whether you’re exploring pre-trained models or diving into Grad-CAM to understand AI decision-making, this project offers a rich, interactive experience. Try it out and discover the power of AI image classification!