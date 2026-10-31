
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import math
from PIL import Image

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Brain MRI Classification",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #F5F7FA;
    color: #1E293B;
}

.block-container {
    max-width: 1400px;
    padding: 2rem 3rem;
}

[data-testid="stSidebar"] {
    background-color: #102A43;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    padding: 15px 0 30px 0;
}

.sidebar-caption {
    color: #B8C7D9 !important;
    font-size: 12px;
    letter-spacing: 1px;
    margin-top: 25px;
}

.page-title {
    font-size: 32px;
    font-weight: 750;
    color: #102A43;
    margin-bottom: 8px;
}

.page-subtitle {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #102A43;
    margin-bottom: 12px;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #94A3B8;
    border-radius: 10px;
    padding: 15px;
}

.stButton > button {
    background: #087E8B;
    color: white;
    border: none;
    border-radius: 8px;
    height: 46px;
    font-weight: 650;
    font-size: 15px;
}

.stButton > button:hover {
    background: #066570;
    color: white;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 10px;
}

.result-label {
    color: #64748B;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 12px;
}

.footer {
    color: #94A3B8;
    font-size: 12px;
    text-align: center;
    margin-top: 50px;
    padding: 15px;
}

hr {
    border-color: #E2E8F0;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD ANN MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "brain_tumor_ann.keras",
        compile=False
    )

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ---------------- LOAD CLASS NAMES ----------------

try:
    with open("classes.json", "r") as file:
        classes = json.load(file)

    if isinstance(classes, dict):
        classes = [
            classes[str(i)]
            for i in range(len(classes))
        ]

except Exception as e:
    st.error(f"Unable to load classes.json: {e}")
    st.stop()


if model.output_shape[-1] != len(classes):
    st.error("Model output classes and classes.json do not match.")
    st.stop()


# ---------------- CLASS DISPLAY NAMES ----------------

class_names = {
    "glioma": "GLIOMA TUMOR",
    "meningioma": "MENINGIOMA TUMOR",
    "notumor": "NO TUMOR DETECTED",
    "pituitary": "PITUITARY TUMOR"
}


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown(
        '<div class="brand">🧠 NeuroScan AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-caption">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    st.markdown("🏥 MRI Classification")
    st.markdown("📁 Image Prediction")

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-caption">MODEL INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.write("Artificial Neural Network")
    st.write("Four-Class Classification")

    st.markdown("---")

    st.caption("NeuroScan AI")


# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="page-title">
    Brain MRI Classification
</div>

<div class="page-subtitle">
    AI-Assisted Tumor Classification System
    | Artificial Neural Network
</div>
""", unsafe_allow_html=True)

st.divider()


# ---------------- IMAGE UPLOAD ----------------

st.markdown(
    '<div class="section-title">Upload MRI Images</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Select brain MRI images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# ---------------- IMAGE PREPROCESSING ----------------

def preprocess_image(image):

    if len(model.input_shape) == 2:

        n_features = model.input_shape[-1]

        if n_features % 3 != 0:
            raise ValueError("Expected flattened RGB image features.")

        pixels = n_features // 3
        img_size = math.isqrt(pixels)

        if img_size * img_size != pixels:
            raise ValueError("Cannot determine image dimensions.")

        image = image.convert("RGB")

        image = image.resize(
            (img_size, img_size),
            Image.Resampling.LANCZOS
        )

        arr = np.asarray(image, dtype=np.float32) / 255.0

        arr = arr.flatten()

        return np.expand_dims(arr, axis=0)

    elif len(model.input_shape) == 4:

        height = model.input_shape[1]
        width = model.input_shape[2]
        channels = model.input_shape[3]

        if height is None or width is None:
            raise ValueError("Dynamic image dimensions are unsupported.")

        if channels == 3:
            image = image.convert("RGB")

        elif channels == 1:
            image = image.convert("L")

        else:
            raise ValueError("Unsupported image channels.")

        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )

        arr = np.asarray(image, dtype=np.float32)

        if channels == 1:
            arr = np.expand_dims(arr, axis=-1)

        arr = arr / 255.0

        return np.expand_dims(arr, axis=0)

    else:
        raise ValueError("Unsupported model input shape.")


# ---------------- PREDICTION ----------------

if uploaded_files:

    st.info(f"{len(uploaded_files)} MRI image(s) selected.")

    if st.button("Predict", use_container_width=True):

        results = []

        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):

            try:

                image = Image.open(uploaded_file).convert("RGB")

                processed_image = preprocess_image(image)

                prediction = model.predict(
                    processed_image,
                    verbose=0
                )

                predicted_index = int(
                    np.argmax(prediction[0])
                )

                original_class = str(
                    classes[predicted_index]
                ).lower()

                predicted_disease = class_names.get(
                    original_class,
                    original_class.upper()
                )

                results.append({
                    "image": image.copy(),
                    "name": uploaded_file.name,
                    "disease": predicted_disease
                })

            except Exception as e:

                st.error(
                    f"Error processing {uploaded_file.name}: {e}"
                )

            progress.progress((i + 1) / len(uploaded_files))

        st.session_state["results"] = results


# ---------------- DISPLAY RESULTS ----------------

if st.session_state.get("results"):

    results = st.session_state["results"]

    st.divider()

    st.markdown(
        '<div class="section-title">MRI Classification Results</div>',
        unsafe_allow_html=True
    )

    st.write("Predicted classification of uploaded MRI images.")

    for start in range(0, len(results), 3):

        cols = st.columns(3, gap="large")

        for j, result in enumerate(results[start:start + 3]):

            with cols[j]:

                with st.container(border=True):

                    st.image(
                        result["image"],
                        width=180
                    )

                    st.caption(result["name"])

                    st.markdown(
                        '<div class="result-label">'
                        'PREDICTED CLASSIFICATION'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.subheader(result["disease"])


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    Brain MRI Classification System
    <br><br>
    Powered by Artificial Neural Networks
    <br>
    Educational use only — not intended for clinical diagnosis.
</div>
""", unsafe_allow_html=True)
