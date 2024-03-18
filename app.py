import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import cv2

# # Function to load and preprocess the image
def preprocess_image(img):
    img = Image.open(img)
    img = img.resize((224, 224))  # Assuming the input size expected by your model
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    return img

# # Load your pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('classifier_pneumonia_224x224.h5')  # Replace 'your_model_path.h5' with your actual model path
    return model

# # Function to make predictions
def predict(image, model):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction


def main():
    st.title("Image Classification for Pneumonia")

    # st.write("Upload an image for Brain Tumor:")
    # uploaded_tumor_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="tumor")

    st.write("Upload an image for Pneumonia:")
    uploaded_pneumonia_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="pneumonia")

    # if uploaded_tumor_image is not None:
    #     tumor_image = Image.open(uploaded_tumor_image)
    #     st.image(tumor_image, caption="Uploaded Brain Tumor Image", use_column_width=True)
    #     st.write("Brain Tumor image uploaded successfully!")

    if uploaded_pneumonia_image is not None:
        # pneumonia_image = Image.open(uploaded_pneumonia_image)
        # st.image(pneumonia_image, caption="Uploaded Pneumonia Image", use_column_width=True)
        # st.write("Pneumonia image uploaded successfully!")
        st.write("Classifying...")

        model = load_model()
        # Perform prediction
        prediction = predict(uploaded_pneumonia_image, model)
        print(prediction)
        if prediction[0] < 0.85:  # Assuming the first class is brain tumor and second class is pneumonia
            st.write("Prediction: Normal")
        else:
            st.write("Prediction: Pneumonia")
    else :
        st.write("Please upload the correct file extension")

if __name__ == "__main__":
    main()
