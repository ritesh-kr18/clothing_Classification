# 👕 Clothing Image Classification App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clothingclassification-nnf63h5cmnsx8g5gxwavmc.streamlit.app)

A web-based machine learning application that predicts the category of clothing from an uploaded image. This project was developed as part of Lab Assignment-4, deploying a PyTorch Convolutional Neural Network (CNN) through a Streamlit interface.

## 🚀 Live Demo
Try the application here: **[Clothing Classification App](https://clothingclassification-nnf63h5cmnsx8g5gxwavmc.streamlit.app)**

## 🧠 About the Model
The underlying model is a custom **Convolutional Neural Network (CNN)** built with PyTorch, trained on a 10-class clothing dataset. 
The 10 categories are:
`Dress` | `Hat` | `Longsleeve` | `Outwear` | `Pants` | `Shirt` | `Shoes` | `Shorts` | `Skirt` | `T-shirt`

### Architecture:
* **Feature Extraction:** 3 Convolutional Blocks (Conv2d, BatchNorm2d, ReLU, MaxPool2d)
* **Classification:** Flatten layer followed by Fully Connected layers with Dropout (0.5) for regularization.
* **Input Size:** Images are preprocessed and resized to 128x128.

## 🛠️ Built With
* **[PyTorch](https://pytorch.org/)** - For building and training the CNN model
* **[Streamlit](https://streamlit.io/)** - For the web application frontend
* **[Torchvision](https://pytorch.org/vision/stable/index.html)** - For image transformations and preprocessing
* **[Pillow (PIL)](https://python-pillow.org/)** - For image handling

## 💻 How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/ritesh-kr18/clothing_Classification.git
cd clothing_Classification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📝 Features
* **Image Upload:** Supports JPG, JPEG, and PNG formats.
* **Fast Inference:** Automatically preprocesses the image and passes it through the model.
* **Top-3 Predictions:** Outputs the primary prediction along with a confidence percentage, as well as the runner-up classifications.
