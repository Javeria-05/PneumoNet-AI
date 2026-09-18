import streamlit as st
import numpy as np
import zipfile
import tempfile
import os
import cv2
import tensorflow as tf
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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_image, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid

model = get_model()

IMG_SIZE = (224, 224)

st.title("🫁 PneumoNet-AI")
st.write("A Deep Learning tool to detect Pneumonia from Chest X-Rays")

uploaded_file = st.file_uploader("Upload a Chest X-Ray image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array_batch)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction

    heatmap = make_gradcam_heatmap(img_array_batch, model)
    original_uint8 = np.array(img_resized).astype(np.uint8)
    overlaid_image = overlay_heatmap(heatmap, original_uint8)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded X-Ray', use_container_width=True)
    with col2:
        st.image(overlaid_image, caption='Grad-CAM Heatmap', use_container_width=True)

    st.subheader("Result:")
    if predicted_class == 1:
        st.error(f"⚠️ PNEUMONIA Detected — Confidence: {confidence*100:.2f}%")
    else:
        st.success(f"✅ NORMAL — Confidence: {confidence*100:.2f}%")

    st.caption("The heatmap highlights regions the model focused on when making its prediction (red = high influence).")
    st.caption("Note: This tool is for research/educational purposes only. Please consult a qualified doctor for medical diagnosis.")