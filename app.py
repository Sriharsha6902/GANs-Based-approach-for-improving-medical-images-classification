import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
import numpy as np

# Set page config
st.set_page_config(page_title="Medical Condition Detection App", page_icon=":microscope:", layout="centered", initial_sidebar_state="expanded")

@st.cache(allow_output_mutation=True)
def load_pneumonia_model():
    model = tf.keras.models.load_model('Classifier_pneumonia_syn.h5')
    return model

@st.cache(allow_output_mutation=True)
def load_alzheimers_model():
    model = tf.keras.models.load_model('Classifier_Alzheimer_syn.h5')
    return model

def preprocess_image(img):
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255
    return img
    
def preprocess_img_alz(img):
    img = Image.open(img).convert('RGB')
    img_size = (224, 224)
    channels = 3
    st.image(img)
    image = img.resize(img_size)
    image_array = np.array(image)
    # preprocessed_image = image_array.reshape(1, img_size[0], img_size[1], channels)
    return image_array

def predict_pneumonia(image, model):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    st.image(img)
    prediction = model.predict(img)
    return prediction

def predict_alzheimers(image, model):
    img = preprocess_img_alz(image)
    prediction = model.predict(img)
    return prediction

def main():
    st.title("Medical Condition Detection App")

    task = st.selectbox("Select Detection Task", ["Select Task", "Pneumonia Detection", "Alzheimer's Disease Detection"])

    if task != "Select Task":

        # Display the title and file uploader
        st.header("Upload an image for medical condition detection:")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="medical_condition")

        if uploaded_image is not None:
            if task == "Pneumonia Detection":
                st.subheader("Pneumonia Detection:")
                with st.spinner('Predicting...'):
                    pneumonia_model = load_pneumonia_model()
                    pneumonia_prediction = predict_pneumonia(uploaded_image, pneumonia_model)
                    print(pneumonia_prediction)
                    pneumonia_pred = tf.squeeze(pneumonia_prediction)
                    pneumonia_pred = pneumonia_pred >= 0.5
                    if pneumonia_pred:  
                        st.success("Prediction: Pneumonia")
                    else:
                        st.success("Prediction: Normal")

            elif task == "Alzheimer's Disease Detection":
                alz_classes=["Mild_Demented","Moderate_Demented","Non_Demented","Very_Mild_Demented"]
                st.subheader("Alzheimer's Detection:")
                with st.spinner('Predicting...'):
                    alzheimers_model = load_alzheimers_model()
                    alzheimers_prediction = predict_alzheimers(uploaded_image, alzheimers_model)
                    print(alzheimers_prediction)
                    pred = np.argmax(alzheimers_prediction,axis=1)
                    if pred==0:
                        st.write("Prediction: Mild_Demented")
                    elif pred==1:
                        st.write("Prediction: Moderate_Demented")
                    elif pred==2:
                        st.write("Prediction: Non_Demented")
                    else:
                        st.write("Prediction: Very_Mild_Demented")

        else:
            st.write("Please upload the correct file extension")

if __name__ == "__main__":
    main()
