# Lab Assignment 4: Deployed Image Classification System Report

## 1. Introduction
This report details the deployment of a Convolutional Neural Network (CNN) for a 10-class clothing image classification task. The objective was to take the trained PyTorch model and deploy it via a web-based application using the Streamlit framework.

## 2. Dataset and Preprocessing
The model was trained on a clothing dataset consisting of 10 classes:
- Dress, Hat, Longsleeve, Outwear, Pants, Shirt, Shoes, Shorts, Skirt, T-shirt

To ensure compatibility with the trained model, the deployment system preprocesses all uploaded images by:
1. Resizing the image to 128x128 pixels.
2. Converting the image to a PyTorch tensor (scaling pixel values between 0 and 1).

## 3. Model Architecture
The deployed model (`ModifiedModel`) is a custom CNN architecture implemented in PyTorch. It consists of:
- **Feature Extractor (features):**
  - Three blocks of Convolutional Layers (16, 32, and 64 filters of size 3x3).
  - Batch Normalization after each convolution for faster and more stable convergence.
  - ReLU activation functions.
  - MaxPooling (2x2) after each block to reduce spatial dimensions.
- **Classifier (classifier):**
  - A Flatten layer.
  - A fully connected (Linear) layer with 128 units, followed by Batch Normalization, ReLU, and Dropout (50%) for regularization.
  - An output layer with 10 units corresponding to the 10 clothing classes.

## 4. Web Application (Streamlit)
The application was built using **Streamlit**, which provides a user-friendly web interface for machine learning models. 

### Key Features of the Deployment:
- **Model Caching:** The Streamlit `@st.cache_resource` decorator is used to load the model into memory only once, avoiding unnecessary reloading during user interactions.
- **Image Upload:** The user can upload images in JPG, JPEG, or PNG formats using the `st.file_uploader` widget.
- **Inference:** Once an image is uploaded, the app applies the preprocessing pipeline and feeds the tensor into the model. We use `torch.no_grad()` to disable gradient calculation for faster inference.
- **Output:** The model's raw outputs are passed through a Softmax function to obtain prediction probabilities. The application then displays:
  - The top predicted class and its confidence score.
  - A ranked list of the top-3 predictions and their respective probabilities.

## 5. Conclusion
The deployment successfully bridges the gap between model development and end-user accessibility. By utilizing Streamlit and PyTorch, we developed a fast, interactive web interface where users can easily test the trained clothing classification model on new images.
