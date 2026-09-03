import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Artisan AI Image Enhancer", layout="wide")

st.title("📷 Artisan AI Image Enhancer")
st.write("Upload an artisan product photo and enhance its quality using AI.")

@st.cache_resource
def load_ai_model():
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel("FSRCNN_x3.pb")
    model.setModel("fsrcnn", 3)
    return model

file = st.file_uploader(
    "Upload Product Image",
    type=["jpg", "jpeg", "png"]
)

if file:

    original_image = Image.open(file).convert("RGB")
    image_rgb = np.array(original_image)

    left, right = st.columns(2)

    with left:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)

    if st.button("✨ Enhance Image with AI"):

        with st.spinner("AI is improving the image..."):

            model = load_ai_model()

            image_bgr = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR
            )

            height, width = image_bgr.shape[:2]

            if max(height, width) > 600:
                scale = 600 / max(height, width)

                image_bgr = cv2.resize(
                    image_bgr,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA
                )

            enhanced_bgr = model.upsample(image_bgr)

            lab_image = cv2.cvtColor(
                enhanced_bgr,
                cv2.COLOR_BGR2LAB
            )

            l, a, b = cv2.split(lab_image)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            l = clahe.apply(l)

            enhanced_bgr = cv2.cvtColor(
                cv2.merge((l, a, b)),
                cv2.COLOR_LAB2BGR
            )

            enhanced_rgb = cv2.cvtColor(
                enhanced_bgr,
                cv2.COLOR_BGR2RGB
            )

        with right:
            st.subheader("AI Enhanced Image")
            st.image(enhanced_rgb, use_container_width=True)

        image_bytes = io.BytesIO()

        Image.fromarray(enhanced_rgb).save(
            image_bytes,
            format="PNG"
        )

        st.download_button(
            "⬇ Download Enhanced Image",
            image_bytes.getvalue(),
            "ai_enhanced_product.png",
            "image/png"
        )
