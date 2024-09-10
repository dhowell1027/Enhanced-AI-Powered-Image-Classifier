# Advanced AI-Powered Image Classifier with Grad-CAM

## Overview

This project is an advanced image classification tool built with cutting-edge deep learning models, enabling users to classify images with explainability. By utilizing Grad-CAM (Gradient-weighted Class Activation Mapping), this app provides a visual explanation of the AI’s decision-making process, offering a deeper understanding of how models perceive and classify images.

With support for multiple pre-trained models, this application is a powerful tool for both quick image classification tasks and exploring model interpretability.

## Features

- **Model Variety**: Choose from a range of pre-trained models including MobileNetV2, ResNet50, EfficientNet, InceptionV3, NASNet, Xception, DenseNet, and more. Each model provides different trade-offs between speed and accuracy.
  
- **Grad-CAM Explainability**: Visualize how the AI "sees" the image by generating heatmaps that highlight the regions the model focused on while making predictions. This feature enhances transparency in AI decision-making.

- **Image Filters**: Apply image transformations such as grayscale and Gaussian blur to see how the model reacts to pre-processed images.

- **Image Rotation**: Easily rotate images in 90-degree intervals to analyze how orientation affects classification results.

- **Interactive Visualizations**: View dynamic and intuitive bar charts showing the top predicted classes with their associated probabilities.

## What's New?

This version introduces several new models and enhanced functionality:
- **New Models**: Added support for EfficientNet, InceptionV3, NASNet, Xception, DenseNet, and Vision Transformers. Each model brings unique capabilities, offering a variety of approaches to image classification.
- **Adjustable Blur Filter**: Users can now control the intensity of the Gaussian blur, allowing for more flexibility in pre-processing.
- **Expanded Explainability**: Grad-CAM now works with all models, offering visual insights for any selected model.
- **Streamlined Performance**: Improved backend handling of image transformations and model predictions to optimize the user experience.

## How It Works

1. **Upload an Image**: Start by uploading an image from your local machine. The app accepts most common formats such as JPG and PNG.
   
2. **Select a Model**: Choose from a variety of state-of-the-art models, each optimized for different tasks. MobileNetV2 is great for quick, lightweight predictions, while ResNet50, InceptionV3, and others offer higher accuracy for more complex images.

3. **Apply Filters (Optional)**: Modify your image by applying grayscale or blur filters. This allows you to experiment with how pre-processing impacts model predictions.

4. **View Predictions**: The app will show the top predictions from the model, including probability scores. These results are displayed in easy-to-understand, interactive bar charts.

5. **Understand the Model’s Focus**: Using Grad-CAM, a heatmap is overlaid on the image to highlight the regions that contributed most to the model’s prediction. This powerful tool provides transparency into how the model makes decisions.

## Example Use Cases

- **Model Explainability**: Use Grad-CAM to explore why a model made a particular prediction, making the AI more transparent and trustworthy.
  
- **Experimentation with Pre-trained Models**: Test how different pre-trained models perform on various images and see which ones offer the best balance of speed and accuracy for your tasks.

- **Educational Purposes**: Ideal for demonstrating how different deep learning models work and how they interpret image data. Great for AI researchers, students, and educators.

## Setup and Installation

### Prerequisites

- **Python 3.7+**
- **pip** (Python package manager)

### Installation

**Clone the Repository**:

   git clone https://github.com/dhowell1027/Enhanced-AI-Powered-Image-Classifier
Navigate to the Project Directory:

cd advanced-image-classifier

Install the Dependencies:

pip install -r requirements.txt

Run the Application:

    streamlit run app.py

    The app will launch in your browser, typically at http://localhost:8501.

Future Plans

Here are some features and enhancements planned for future releases:

    Custom Model Upload: Allow users to upload and test custom-trained models.
    Advanced Image Filters: Introduce new filters such as sharpen, edge detection, and noise reduction.
    Performance Metrics: Display additional metrics such as inference time, model size, and memory usage.
    Extended Model Support: Add new models like Vision Transformers for even more cutting-edge classification tasks.

Technologies Used

    TensorFlow: For loading and managing pre-trained deep learning models.
    Streamlit: A fast and interactive UI framework for building data applications.
    Pillow: Used for image manipulation, such as applying filters and rotating images.
    OpenCV: For handling advanced image processing tasks.
    Plotly & Matplotlib: For generating interactive and static visualizations of classification results.

Contributing

Contributions are welcome! If you want to enhance the project, feel free to submit a pull request.

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

This advanced AI-Powered Image Classifier provides a robust, user-friendly platform for image classification, enhanced by Grad-CAM visualization. Whether you're using it for quick predictions or as a learning tool for AI explainability, this app showcases the power of modern deep learning models in a transparent and intuitive way. Dive in and explore the world of AI image classification!