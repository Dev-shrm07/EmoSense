# EmoSense

This project is a Flask web application that utilizes computer vision to detect emotions in a person's face. It provides two modes of operation: live webcam streaming and file upload mode. The frontend of the web app is built using Bootstrap, and the emotion classification model is implemented using Convolutional Neural Networks (CNNs). The CNN model is trained on the FER 2013 dataset, which consists of grayscale face images labeled with seven different emotions. The project leverages popular libraries such as NumPy, Matplotlib, TensorFlow, Keras, OpenCV, and Flask.



## Methodology

The methodology of this project can be summarized as follows:

1. **Data Collection:** The emotion classification model is trained on the FER 2013 dataset, which contains a large number of grayscale face images labeled with different emotions.

2. **Model Architecture:** A Convolutional Neural Network (CNN) architecture is used to train the emotion classification model. CNNs are well-suited for image classification tasks and can learn spatial hierarchies from image data.

3. **Model Training:** The CNN model is trained on the FER 2013 dataset using techniques such as backpropagation and gradient descent. The model is trained to minimize the classification error on the training data and generalize well to unseen data.

4. **Web App Development:** The Flask web application is developed to provide a user-friendly interface for emotion detection. The frontend is built using Bootstrap, which ensures an attractive and responsive design. Flask handles the backend logic and serves the web pages.

5. **Emotion Detection:** The web app supports two modes of operation. In live webcam streaming mode, OpenCV is used to capture video frames from the user's webcam. These frames are then passed through the trained CNN model to detect emotions in real-time. In file upload mode, users can upload an image file, which is processed using the trained model to detect emotions.

6. **Result Visualization:** The detected emotions are displayed to the user through the web app's frontend. Matplotlib can be used to visualize the emotions using graphs or charts.

## Requirements

Before running the Emotion Detection Web App, ensure that the following dependencies are installed on your system:

- Python (3.6 or later)
- Flask
- OpenCV
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Bootstrap (frontend)

