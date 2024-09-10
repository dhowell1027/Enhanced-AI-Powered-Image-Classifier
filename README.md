Enhanced AI-Powered Image Classifier with Grad-CAM

Welcome to the Enhanced AI-Powered Image Classifier! This tool leverages cutting-edge deep learning models to classify images and provide real-time visual feedback on how the AI makes its decisions. With an intuitive user interface, users can upload or capture images, apply filters, and gain insight into model predictions using Grad-CAM for explainability.
Why This Project?

Understanding how and why an AI model makes decisions is as important as the decision itself. Most image classifiers treat models like black boxes, but this app introduces transparency by offering visual insights into the AI's thought process using Grad-CAM. Whether you're looking to classify objects quickly or to explore the inner workings of deep learning models, this project is designed to be both powerful and accessible.
Key Features

    Upload or Capture Images: Choose to upload an image from your device or capture one using your webcam.
    Model Selection: Switch between two state-of-the-art models:
        MobileNetV2 for lightweight and fast predictions.
        ResNet50 for more accurate, but computationally heavier, classifications.
    Apply Filters: Experiment with image preprocessing by applying grayscale or blur filters.
    Grad-CAM Visualization: Understand why the model makes a specific prediction by viewing a heatmap that highlights the important regions in the image.
    Interactive Visualizations: Predictions are shown through interactive bar charts and visual explanations of model behavior.

How to Set Up and Run Locally
Prerequisites

Make sure you have the following installed:

    Python 3.7+
    pip (Python package manager)

Installation

    Clone the repository:

    bash

git clone https://github.com/yourusername/enhanced-image-classifier.git

Navigate to the project folder:

bash

cd enhanced-image-classifier

Install the necessary dependencies:

bash

pip install -r requirements.txt

Run the application:

bash

    streamlit run app.py

This will open the app in your browser, typically at http://localhost:8501.
Using the Classifier

    Choose an Image: Upload an image file from your local machine or capture an image using your webcam.

    Select a Model: From the sidebar, choose between MobileNetV2 for faster results or ResNet50 for more accurate predictions.

    Apply Filters (Optional): If you want to experiment, apply grayscale or blur filters to modify the image.

    View Predictions: Once the image is processed, the app will display the top predictions with probability scores. These predictions will be visualized using dynamic bar charts for easy interpretation.

    Grad-CAM Explainability: The heatmap displayed below the prediction results provides insights into which parts of the image the model focused on when making its prediction.

Example Use Cases

    Educational Tool: For those learning about neural networks, this project serves as a practical way to visualize how models classify images and how Grad-CAM can enhance interpretability.
    Rapid Prototyping: Quickly test different pre-trained models on a new set of images and compare performance between them.
    Model Explainability: This tool emphasizes transparency, making it perfect for those wanting to understand and explain AI decision-making.

Technologies Used

    Streamlit: For building an interactive, real-time web interface.
    TensorFlow: For handling pre-trained models (MobileNetV2 and ResNet50).
    Pillow: For image processing and applying filters (grayscale, blur).
    OpenCV: For handling webcam image capture and processing.
    Plotly & Matplotlib: For generating interactive and static visualizations of the classification results.

Future Improvements

This is just the beginning! Here's what I'd like to add moving forward:

    Custom Models: Allow users to upload and test custom-trained models.
    Additional Filters: Integrate more complex image transformations, such as edge detection and sharpen.
    Extended Model Support: Add support for models like EfficientNet or InceptionV3 to compare performance.
    Performance Metrics: Display additional metrics like inference time and memory usage for each model.

Contributing

Contributions are welcome! Whether it's fixing bugs, adding new features, or suggesting improvements, feel free to fork the repository and create a pull request.

    Fork the repository.
    Create your feature branch (git checkout -b feature/new-feature).
    Commit your changes (git commit -m 'Add some feature').
    Push to the branch (git push origin feature/new-feature).
    Open a pull request.

License

This project is licensed under the MIT License - see the LICENSE file for details.
Closing Notes

Thanks for checking out the Enhanced AI-Powered Image Classifier! My goal was to create an intuitive tool that not only performs image classification but also offers insights into why the model made its predictions. By incorporating Grad-CAM explainability, I hope this tool serves as both a practical classifier and a learning tool for anyone curious about AI.