import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("📷 Artisan AI Product Image Enhancer")

file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])

if file:
    original_image = Image.open(file).convert("RGB")
    image = np.array(original_image)

    st.write("Original Image")
    st.image(image)

    if st.button("Enhance Image"):

        enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=20)

        blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

        st.write("Enhanced Image")
        st.image(enhanced)

        output_image = Image.fromarray(enhanced)

        image_bytes = io.BytesIO()
        output_image.save(image_bytes, format="PNG")

        st.download_button(
            "Download Enhanced Image",
            image_bytes.getvalue(),
            "enhanced_product.png",
            "image/png"
        )