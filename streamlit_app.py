import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2

# Load YOLOv8 model
@st.cache_resource
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.eval()
    return model

model = load_model('https://github.com/ultralytics/ultralytics')

def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def detect_potholes(image):
    results = model(image)
    return results

def main():
    st.title("Pothole Detection App")
    st.write("Upload an image to detect potholes.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("Detecting potholes...")
        
        processed_image = preprocess_image(image)
        results = detect_potholes(processed_image)

        st.write("Detection Results:")
        st.write(results.pandas().xyxy[0].to_dict())

        # Display results on the image
        img_with_boxes = results.render()[0]
        st.image(img_with_boxes, caption='Detected Potholes', use_column_width=True)

if __name__ == "__main__":
    main()
