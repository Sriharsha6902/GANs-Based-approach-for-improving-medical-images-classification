import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
import numpy as np

# Set page config
st.set_page_config(page_title="Pneumonia Detection App", page_icon=":microscope:", layout="centered", initial_sidebar_state="expanded")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('classifier_pneumonia_224x224.h5')
    return model

def preprocess_image(img):
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255
    return img

def predict(image, model):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

def main():
    st.title("Pneumonia Detection App")

    # Load the image from the URL
    response = requests.get('https://i.postimg.cc/mgcdTtQm/78205958-d81f-434a-8a70-7f5a00f12645.jpg')
    image = Image.open(BytesIO(response.content))

    # Set the image as the background using custom CSS
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/png;base64,{image});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the title and file uploader
    st.header("Upload an image for Pneumonia:")
    uploaded_pneumonia_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="pneumonia")

    if uploaded_pneumonia_image is not None:
        with st.spinner('Predicting...'):
            model = load_model()
            prediction = predict(uploaded_pneumonia_image, model)
            y_pred = tf.squeeze(prediction)
            y_pred = y_pred >= 0.88
            if y_pred:  # Assuming the first class is brain tumor and second class is pneumonia
                st.success("Prediction: Pneumonia")
            else:
                st.success("Prediction: Normal")
    else :
        st.write("Please upload the correct file extension")

if __name__ == "__main__":
    main()
