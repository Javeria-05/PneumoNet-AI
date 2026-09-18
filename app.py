import streamlit as st
import numpy as np
import zipfile
import tempfile
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from PIL import Image

st.set_page_config(page_title="PneumoNet-AI", page_icon="🫁", layout="centered")

@st.cache_resource
def get_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    with zipfile.ZipFile('pneumonia_model.keras', 'r') as z:
        with z.open('model.weights.h5') as f:
            weights_bytes = f.read()

    with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False) as tmp:
        tmp.write(weights_bytes)
        tmp_path = tmp.name

    model.load_weights(tmp_path)
    os.remove(tmp_path)
    return model

model = get_model()

IMG_SIZE = (224, 224)

st.title("🫁 PneumoNet-AI")
st.write("A Deep Learning tool to detect Pneumonia from Chest X-Rays")

uploaded_file = st.file_uploader("Upload a Chest X-Ray image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-Ray', use_container_width=True)

    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction

    st.subheader("Result:")
    if predicted_class == 1:
        st.error(f"⚠️ PNEUMONIA Detected — Confidence: {confidence*100:.2f}%")
    else:
        st.success(f"✅ NORMAL — Confidence: {confidence*100:.2f}%")

    st.caption("Note: This tool is for research/educational purposes only. Please consult a qualified doctor for medical diagnosis.")