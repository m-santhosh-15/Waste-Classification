import os
os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("♻️ Waste Classification System")
st.write("Upload an image and the model will classify the waste category.")

model = tf.keras.models.load_model("model/waste_model.h5")

with open("labels.txt") as f:
    classes = f.read().splitlines()

def predict(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0]
    label_idx = np.argmax(prediction)
    return classes[label_idx], prediction[label_idx]

uploaded = st.file_uploader("Upload Waste Image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=300)

    label, confidence = predict(img)
    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: {confidence:.2f}")
