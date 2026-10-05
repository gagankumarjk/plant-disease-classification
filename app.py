import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# FILE NAMES
# ============================================================

MODEL_PATH = "best_plant_disease_224_finetuned.keras"
CLASS_NAMES_PATH = "plant_disease_class_names.json"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    color: #000000 !important;
}

/* Make all normal text black */
.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp div {
    color: #000000;
}

/* Main title */
.main-title {
    text-align: center;
    color: #000000 !important;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #000000 !important;
    font-size: 18px;
    margin-bottom: 30px;
}

/* All headings */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #e8f5e9;
}

section[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* Radio buttons */
div[data-testid="stRadio"] label {
    color: #000000 !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    color: #000000 !important;
}

div[data-testid="stFileUploader"] * {
    color: #000000 !important;
}

/* Prediction result box */
.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.10);
    margin-top: 20px;
    color: #000000 !important;
}

/* Predicted disease */
.disease {
    color: #000000 !important;
    font-size: 28px;
    font-weight: bold;
}

/* Confidence */
.confidence {
    color: #000000 !important;
    font-size: 22px;
    font-weight: bold;
}

/* Top predictions */
.top-prediction {
    color: #000000 !important;
    font-size: 18px;
}

/* Buttons */
button {
    color: #000000 !important;
}

/* Camera / upload text */
[data-testid="stCameraInput"] * {
    color: #00FF00 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #000000 !important;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"Model not found: {MODEL_PATH}"
    )

    st.info(
        "Make sure the .keras model is inside the same "
        "folder as app.py."
    )

    st.stop()


if not os.path.exists(CLASS_NAMES_PATH):

    st.error(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

    st.info(
        "Create plant_disease_class_names.json and "
        "place it in the same folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as file:

        class_names = json.load(file)

    return class_names


model = load_model()
class_names = load_class_names()


# ============================================================
# OPENCV PREPROCESSING
# Matches the Jupyter prediction pipeline
# ============================================================

def preprocess_with_opencv(image_bytes):

    # Convert uploaded image bytes to NumPy array
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    # Read image using OpenCV
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not read the image."
        )

    # OpenCV reads BGR
    # Convert BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize exactly like Jupyter
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # Convert to float32
    # IMPORTANT:
    # Do NOT use mobilenet_v2.preprocess_input()
    image = image.astype(
        np.float32
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_bytes):

    try:
        # Convert bytes to NumPy array
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        # Decode image using OpenCV
        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        # Image could not be decoded
        if image is None:
            return False, "Image is not valid."

        # Check image dimensions
        height, width = image.shape[:2]

        if height < 100 or width < 100:
            return False, "Image is too small. Please upload a clear photo."

        # Convert to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Check if image is almost completely blank
        mean_brightness = np.mean(gray)

        if mean_brightness < 15:
            return False, "Image is too dark. Please upload a clear photo."

        if mean_brightness > 245:
            return False, "Image is too bright. Please upload a clear photo."

        # Check whether image has enough variation
        image_std = np.std(gray)

        if image_std < 10:
            return False, "Image appears blank or unclear. Please upload a valid photo."

        return True, "Valid image"
        

    except Exception:
        return False, "Image is not valid. Please upload another photo."

        # ----------------------------------------------------
        # HUMAN FACE DETECTION
        # ----------------------------------------------------

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) > 0:

            return False, (
                "A human face was detected. "
                "Please upload or capture a plant leaf image."
            )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_bytes):

    # OpenCV preprocessing
    processed_image = preprocess_with_opencv(
        image_bytes
    )

    # Model prediction
    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    # Find highest probability
    predicted_index = np.argmax(predictions)

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index] * 100
    )

    return predicted_class, confidence
# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🌿 Plant Disease Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'MobileNetV2 Transfer Learning + OpenCV + Streamlit'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 Project Information")

    st.write(
        "Plant disease classification using "
        "MobileNetV2 transfer learning."
    )

    st.markdown("---")

    st.subheader("Model")

    st.write(
        "MobileNetV2"
    )

    st.write(
        "Input Size: 224 × 224"
    )

    st.write(
        "Classes: 23"
    )

    st.markdown("---")

    st.subheader("OpenCV Processing")

    st.write("✓ Image decoding")
    st.write("✓ BGR → RGB")
    st.write("✓ Resize to 224 × 224")
    st.write("✓ MobileNetV2 preprocessing")


# ============================================================
# INPUT METHOD
# ============================================================

st.subheader("📷 Select Image")

input_method = st.radio(
    "Choose input method:",
    [
        "Upload Image",
        "Use Camera"
    ],
    horizontal=True
)

image_bytes = None


# ============================================================
# UPLOAD IMAGE
# ============================================================

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()


# ============================================================
# CAMERA
# ============================================================

else:

    camera_image = st.camera_input(
        "Take a picture of the plant leaf"
    )

    if camera_image is not None:

        image_bytes = camera_image.getvalue()



# ============================================================
# DISPLAY IMAGE + PREDICTION
# ============================================================

if image_bytes is not None:

    st.markdown("---")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # INPUT IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader("🌿 Input Leaf")

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        display_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if display_image is None:

            st.error(
                "⚠️ Image is not valid. "
                "Please upload a valid image."
            )

        else:

            display_image = cv2.cvtColor(
                display_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                display_image,
                caption="Uploaded / Captured Leaf",
                use_container_width=True
            )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with col2:

        st.subheader("🔍 Prediction")

        # Validate image
        is_valid, validation_message = validate_image(
            image_bytes
        )

        if not is_valid:

            st.error(
                f"⚠️ {validation_message}"
            )

            st.info(
                "Please upload a clear plant leaf image "
                "or take another photo using the camera."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing the leaf..."
                ):

                    predicted_class, confidence = predict_image(
                        image_bytes
                    )

                st.markdown(
                    '<div class="result-box">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### 🌱 Predicted Disease"
                )

                st.markdown(
                    f'<div class="disease">'
                    f'{predicted_class}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'Confidence: {confidence:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#607d63;">
    <b>Plant Disease Detection System</b><br>
    MobileNetV2 Transfer Learning | OpenCV | Streamlit<br>
    Developed by: GAGAN KUMAR J K<br>
    Institution & Batch: Imarticus(PGDA48)<br>
    Project Details: Capstone Project 2
    </div>
    """,
    unsafe_allow_html=True
)