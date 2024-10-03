"""
1. Install the following packages:
    pip install streamlit torch torchvision scikit-learn numpy pillow
2. Prepare Dataset
    - Put all images in a folder called "dataset" in the same directory as this file
3. Run the following command to start the Streamlit app:
    streamlit run app.py
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import ssl

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


# Load Pretrained Model
@st.cache(allow_output_mutation=True)
def load_model():
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last layer (classification)
    model.eval()
    return model

# Function to extract features
def extract_features(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image).squeeze().numpy()  # Extract feature vector
    return features

# Function to load dataset images and index them
@st.cache_data
def index_dataset(_model, dataset_folder="dataset/"):
    features_list = []
    image_paths = []
    
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(dataset_folder, filename)
            image = Image.open(img_path).convert('RGB')
            features = extract_features(image, model)
            features_list.append(features)
            image_paths.append(img_path)
    
    features_list = np.array(features_list)
    return features_list, image_paths

# Function to find top-N similar images
def find_similar_images(query_features, features_list, image_paths, top_n=5):
    similarities = cosine_similarity([query_features], features_list)[0]
    top_n_indices = similarities.argsort()[-top_n:][::-1]
    similar_images = [(image_paths[i], similarities[i]) for i in top_n_indices]
    return similar_images

# Streamlit UI
st.title("Visual Search System")

# Load model and index dataset
model = load_model()
features_list, image_paths = index_dataset(model)

# User uploads an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Extract features of uploaded image
    query_features = extract_features(image, model)
    
    # Find similar images
    st.write("Finding similar images...")
    similar_images = find_similar_images(query_features, features_list, image_paths)
    
    # Display similar images
    st.write(f"Top similar images:")
    for img_path, similarity in similar_images:
        st.image(Image.open(img_path), caption=f"Similarity: {similarity:.4f}", use_column_width=True)

