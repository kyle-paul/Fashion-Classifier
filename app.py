import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

st.title("Fashion Classifier")
uploaded_file = st.file_uploader("Choose an square image file", type=["jpg", "png"])
model = keras.models.load_model("CNN_model_2.h5")
labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((28, 28))
    image = image.convert("L")
    st.image(image, caption="Greyscale image")
    
    if st.button("Predict"):
        image = np.array(image)
        prediction = model.predict(image.reshape(1, 28, 28)).argmax(axis = 1)
        st.write(labels[int(prediction)])