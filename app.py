import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
)

IMG_SIZE = (150, 150)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Map each model option to its saved file and the correct Keras preprocessing
# function (must match what was used during training).
MODEL_OPTIONS = {
    "VGG19 (Test Acc: 88.75%)": {
        "path": "models/vgg19_brain_tumor.keras",
        "preprocess": tf.keras.applications.vgg19.preprocess_input,
    },
    "ResNet152 (Test Acc: 86.56%)": {
        "path": "models/resnet152_brain_tumor.keras",
        "preprocess": tf.keras.applications.resnet.preprocess_input,
    },
    "InceptionV3 (Test Acc: 84.50%)": {
        "path": "models/inceptionv3_brain_tumor.keras",
        "preprocess": tf.keras.applications.inception_v3.preprocess_input,
    },
}

CLASS_INFO = {
    "glioma": "A tumor arising from glial cells in the brain or spinal cord.",
    "meningioma": "A typically slow-growing tumor forming in the meninges "
                  "(the membranes surrounding the brain and spinal cord).",
    "notumor": "No visible tumor detected in the scan.",
    "pituitary": "A tumor forming in the pituitary gland at the base of the brain.",
}


# ----------------------------------------------------------------------------
# MODEL LOADING (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image, preprocess_fn):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)          # add batch dimension
    array = preprocess_fn(array)                    # model-specific preprocessing
    return array


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload a brain MRI scan and a transfer-learning model (VGG19, ResNet152, "
    "or InceptionV3) will classify it into one of four categories: "
    "**glioma, meningioma, pituitary tumor, or no tumor**."
)

st.warning(
    "⚠️ This tool is for educational/demo purposes only and is **not** a "
    "medical diagnostic device. Do not use it for real clinical decisions.",
    icon="⚠️",
)

model_choice = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))
config = MODEL_OPTIONS[model_choice]

uploaded_file = st.file_uploader(
    "Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI scan", use_container_width=True)

    if st.button("Classify", type="primary"):
        try:
            with st.spinner("Loading model and running prediction..."):
                model = load_model(config["path"])
                processed = preprocess_image(image, config["preprocess"])
                predictions = model.predict(processed, verbose=0)[0]

            predicted_idx = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(predictions[predicted_idx]) * 100

            st.success(f"**Prediction: {predicted_class.upper()}** "
                       f"({confidence:.2f}% confidence)")
            st.caption(CLASS_INFO[predicted_class])

            # Probability breakdown
            st.subheader("Class probabilities")
            prob_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": predictions,
            }).sort_values("Probability", ascending=False)

            st.bar_chart(prob_df.set_index("Class"))
            st.dataframe(
                prob_df.style.format({"Probability": "{:.2%}"}),
                hide_index=True,
                use_container_width=True,
            )

        except FileNotFoundError:
            st.error(
                f"Model file not found at `{config['path']}`.\n\n"
                "Train the model in the notebook, then save it with:\n\n"
                "```python\nmodel.save('models/vgg19_brain_tumor.keras')\n```\n\n"
                "and place the resulting file in a `models/` folder next to "
                "this app."
            )
else:
    st.info("👆 Upload an MRI image to get started.")

st.divider()
st.caption(
    "Dataset: [Brain Tumor MRI Dataset — Kaggle]"
    "(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)"
)
