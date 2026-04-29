# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


st.set_page_config(
    page_title="Image Editing App",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Editing App")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("Controls")

blur_value = st.sidebar.slider(
    "Blur",
    min_value=1,
    max_value=25,
    value=1,
    step=2
)

sharpness_value = st.sidebar.slider(
    "Sharpness",
    min_value=0.5,
    max_value=3.0,
    value=2.0,
    step=0.1
)

brightness_value = st.sidebar.slider(
    "Brightness",
    min_value=-100,
    max_value=100,
    value=0,
    step=5
)

contrast_value = st.sidebar.slider(
    "Contrast",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1
)

edge_detection = st.sidebar.checkbox("Edge Detection")

threshold1 = st.sidebar.slider(
    "Threshold 1",
    min_value=0,
    max_value=255,
    value=170
)

threshold2 = st.sidebar.slider(
    "Threshold 2",
    min_value=0,
    max_value=255,
    value=120
)

grayscale = st.sidebar.checkbox("Grayscale")

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image using PIL
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Convert RGB to BGR for OpenCV
    processed = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


    # Apply Blur

    if blur_value > 1:
        if blur_value % 2 == 0:
            blur_value += 1
        processed = cv2.GaussianBlur(
            processed,
            (blur_value, blur_value),
            0
        )

    
    # Apply Sharpness
    
    kernel = np.array([
        [0, -1, 0],
        [-1, 5 * sharpness_value, -1],
        [0, -1, 0]
    ])
    processed = cv2.filter2D(processed, -1, kernel)

    
    # Brightness + Contrast
    
    processed = cv2.convertScaleAbs(
        processed,
        alpha=contrast_value,
        beta=brightness_value
    )

    
    # Grayscale
    
    if grayscale:
        processed = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2GRAY
        )

    
    # Edge Detection
    
    if edge_detection:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2GRAY
            )
        else:
            gray = processed

        processed = cv2.Canny(
            gray,
            threshold1,
            threshold2
        )

    # Convert for display
    if len(processed.shape) == 2:
        display_image = processed
    else:
        display_image = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB
        )


    # Side-by-Side Display
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            img_array,
            width=600
        )

    with col2:
        st.subheader("Processed Image")
        st.image(
            display_image,
            width=600
        )

    
    # Download Button
    
    if len(processed.shape) == 2:
        final_img = Image.fromarray(processed)
    else:
        final_img = Image.fromarray(display_image)

    buf = io.BytesIO()
    final_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="edited_image.png",
        mime="image/png"
    )