from PIL import Image
import requests
from io import BytesIO
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
st.set_page_config(page_icon=":sunny:", layout="centered")

# # Function to load and preprocess the image
def preprocess_image(img):
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255
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
    

def main():
    st.title("Pneumonia Detection App")

    # Load the image from the URL
    response = requests.get('https://i.postimg.cc/mgcdTtQm/78205958-d81f-434a-8a70-7f5a00f12645.jpg')
    image = Image.open(BytesIO(response.content))

    # Create columns
    col1, col2 = st.columns(2)

    # Display the image in the left column
    col1.image(image, use_column_width=True)

    # Display the title and file uploader in the right column
    col2.header("Upload an image for Pneumonia:")
    uploaded_pneumonia_image = col2.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="pneumonia")

    if uploaded_pneumonia_image is not None:
        model = load_model()
        prediction = predict(uploaded_pneumonia_image, model)
        y_pred = tf.squeeze(prediction)
        print(y_pred)
        y_pred = y_pred >= 0.855
        if y_pred:  # Assuming the first class is brain tumor and second class is pneumonia
            col2.write("Prediction: Pneumonia")
        else:
            col2.write("Prediction: Normal")
    else :
        st.write("Please upload the correct file extension")
        

if __name__ == "__main__":
    main()
