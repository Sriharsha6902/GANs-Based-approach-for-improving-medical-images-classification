import streamlit as st
from PIL import Image
import requests
from io import BytesIO

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
    col2.header("Upload your image")
    uploaded_file = col2.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        user_image = Image.open(uploaded_file)
        col2.image(user_image, caption='Uploaded Image.', use_column_width=True)

if __name__ == "__main__":
    main()
