import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os

class ModifiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

CLASSES = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

st.set_page_config(page_title="Clothing Image Classification", layout="centered")

st.title("Clothing Image Classification App")
st.write("Upload an image of clothing, and the model will predict its category.")

@st.cache_resource
def load_model():
    model = ModifiedModel()
    model_path = 'clothing_model.pth'
    
    if not os.path.exists(model_path):
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

if model is None:
    st.error("⚠️ Model file `clothing_model.pth` not found!")
    st.info("Please make sure you have saved your model from the Jupyter Notebook using `torch.save(model.state_dict(), 'clothing_model.pth')` in the same directory.")
else:
    # Image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Processing...")
        
        # Preprocess the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Perform inference
        with torch.no_grad():
            output = model(input_batch)
            
        # Get probabilities using Softmax
        probabilities = F.softmax(output, dim=1)[0]
        
        # Get top 3 predictions
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        
        st.subheader("Results:")
        
        # Top 1 Prediction
        predicted_class = CLASSES[top3_catid[0]]
        confidence = top3_prob[0].item() * 100
        
        st.success(f"**Prediction:** {predicted_class.capitalize()} ({confidence:.2f}% confidence)")
        
        st.subheader("Top-3 Predictions:")
        for i in range(3):
            cat_name = CLASSES[top3_catid[i]]
            prob = top3_prob[i].item() * 100
            st.write(f"{i+1}. **{cat_name.capitalize()}**: {prob:.2f}%")
