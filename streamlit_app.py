import streamlit as st
import torch
import os
import requests
import time
from PIL import Image
import numpy as np
from git import Repo

@st.experimental_singleton
def loadModel():
    start_dl = time.time()
    # Clone the YOLOv7 repository
    Repo.clone_from("https://github.com/WongKinYiu/yolov7", "yolov7")
    os.chdir('yolov7')
    # Download the YOLO model weights
    model_url = "https://path.to/your/custom_yolo_model_weights.pt"
    yolo_model = requests.get(model_url)
    with open("best.pt", 'wb') as file:
        file.write(yolo_model.content)
    finished_dl = time.time()
    print(f"Model Downloaded, ETA: {finished_dl - start_dl} seconds")
    # Load and return the YOLO model
    model = torch.hub.load(".", 'custom', 'best.pt', source='local')
    return model

def page1():
    # Load the model
    model = loadModel()

    st.title("Pothole Detection App")
    image_file = st.sidebar.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if image_file is not None:
        img = Image.open(image_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption='Uploaded Image', use_column_width='always')

        # Path for saving the output image
        outpath = os.path.join(os.getcwd(), f"out_{os.path.basename(image_file.name)}")
        
        if st.button('Detect Pothole'):
            # Perform detection
            pred = model(img)
            pred.render()  # Render bbox in image
            
            # Save and display the output image
            for im in pred.imgs:
                im_base64 = Image.fromarray(im)
                im_base64.save(outpath)
            
            img_ = Image.open(outpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')
            
            with st.expander("View Annotation Data"):
                tab1, tab2, tab3 = st.tabs(['Pascal VOC', 'COCO', 'YOLO'])
                
                with tab1:
                    df1 = pred.pandas().xyxy[0]
                    st.dataframe(df1)
                    st.download_button(
                        label="Download Annotation Data as CSV",
                        data=df1.to_csv(),
                        file_name=f"Annotation(Pascal VOC) Data For {image_file.name}.csv",
                        mime='text/csv'
                    )
                
                with tab2:
                    df1 = pred.pandas().xywh[0]
                    st.dataframe(df1)
                    st.download_button(
                        label="Download Annotation Data as CSV",
                        data=df1.to_csv(),
                        file_name=f"Annotation(COCO) Data For {image_file.name}.csv",
                        mime='text/csv'
                    )
                
                with tab3:
                    df1 = pred.pandas().xywhn[0]
                    st.dataframe(df1)
                    st.download_button(
                        label="Download Annotation Data as CSV",
                        data=df1.to_csv(),
                        file_name=f"Annotation(YOLO) Data For {image_file.name}.csv",
                        mime='text/csv'
                    )

if __name__ == '__main__':
    page1()
