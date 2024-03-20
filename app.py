import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# # Function to load and preprocess the image
def preprocess_image(img):
    img = Image.open(img).convert('RGB')
    img = resize(img,(224, 224))
    img = img / 255
    img = np.array(img)
    print(img.shape)
    return img

# # Load your pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('classifier_pneumonia_224x224.h5')
    return model

# # Function to make predictions
def predict(image, model):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction


def main():
    st.title("Image Classification for Pneumonia")
    st.write("Upload an image for Pneumonia:")
    uploaded_pneumonia_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="pneumonia")
    if uploaded_pneumonia_image is not None:
        model = load_model()
        prediction = predict(uploaded_pneumonia_image, model)
        y_pred = tf.squeeze(prediction)
        y_pred = y_pred >= 0.855
        if y_pred:  # Assuming the first class is brain tumor and second class is pneumonia
            st.write("Prediction: Normal")
        else:
            st.write("Prediction: Pneumonia")
    else :
        st.write("Please upload the correct file extension")

if __name__ == "__main__":
    main()
