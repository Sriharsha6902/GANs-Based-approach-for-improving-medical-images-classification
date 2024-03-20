import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_icon=":sunny:", layout="wide")

# Function to load and preprocess the image
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255
    return img

# Load your pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('classifier_pneumonia_224x224.h5')
    return model

# Function to make predictions
def predict(image, model):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

def main():
    st.title("Image Classification for Pneumonia")

    # Load the image from URL
    image_url = "https://i.postimg.cc/mgcdTtQm/78205958-d81f-434a-8a70-7f5a00f12645.jpg"
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Display the image to the right of the title
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(" ")
    with col2:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    model = load_model()
    prediction = predict(img, model)
    y_pred = tf.squeeze(prediction)
    y_pred = y_pred >= 0.855
    if y_pred:  # Assuming the first class is normal and second class is pneumonia
        st.write("Prediction: Pneumonia")
    else:
        st.write("Prediction: Normal")

if __name__ == "__main__":
    main()
