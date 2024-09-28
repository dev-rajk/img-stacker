import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage import exposure, filters
from skimage.restoration import denoise_tv_chambolle
from io import BytesIO
import gc

# Function to load and convert images
def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)

# Memory-efficient cropping
def crop_image(image, crop_x1, crop_y1, crop_x2, crop_y2):
    return image[crop_y1:crop_y2, crop_x1:crop_x2]

# Texture enhancement using high-pass filter
def enhance_texture(image, strength=1.5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (21, 21), 3)
    high_pass = cv2.addWeighted(gray_image, strength, blurred, -strength, 0)
    return cv2.merge([high_pass] * 3)

# Clarity using unsharp mask
def enhance_clarity(image, strength=1.5):
    blurred = cv2.GaussianBlur(image, (9, 9), 10)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

# Grain/noise addition
def add_grain(image, amount=0.05):
    noise = np.random.normal(loc=0, scale=255 * amount, size=image.shape).astype(np.uint8)
    return cv2.add(image, noise)

# Detail enhancement by sharpening
def enhance_detail(image, strength=1.5):
    kernel = np.array([[0, -1, 0], [-1, 5 + strength, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Display an image in Streamlit
def display_image(image, title="Image"):
    st.image(image, caption=title, use_column_width=True)

# Basic image export function
def export_image(image, format='JPEG'):
    buf = BytesIO()
    img_pil = Image.fromarray(image)
    img_pil.save(buf, format=format)
    byte_im = buf.getvalue()
    return byte_im

# Streamlit UI
st.title("Deep Sky Image Processor with Enhanced Image Manipulation")

# File uploader for images
image_files = st.file_uploader("Upload Images (JPEG or PNG)", type=["jpeg", "jpg", "png"], accept_multiple_files=True)

if image_files:
    for file in image_files:
        image = load_image(file)
        display_image(image, "Original Image")

        # Add manipulation options
        st.sidebar.title("Image Manipulation Tools")
        
        # Brightness/Contrast
        brightness = st.sidebar.slider("Brightness", -100.0, 100.0, 1.0)
        contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)

        # Cropping
        st.sidebar.subheader("Cropping")
        crop_x1 = st.sidebar.slider("Crop X1", 0, image.shape[1], 0)
        crop_y1 = st.sidebar.slider("Crop Y1", 0, image.shape[0], 0)
        crop_x2 = st.sidebar.slider("Crop X2", crop_x1, image.shape[1], image.shape[1])
        crop_y2 = st.sidebar.slider("Crop Y2", crop_y1, image.shape[0], image.shape[0])

        # Texture, Clarity, Grain, and Detail enhancements
        texture_strength = st.sidebar.slider("Texture Strength", 1.0, 3.0, 1.5)
        clarity_strength = st.sidebar.slider("Clarity Strength", 1.0, 3.0, 1.5)
        grain_amount = st.sidebar.slider("Grain Amount", 0.0, 0.1, 0.05)
        detail_strength = st.sidebar.slider("Detail Enhancement Strength", 1.0, 3.0, 1.5)

        # Apply manipulations sequentially
        manipulated_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        manipulated_image = crop_image(manipulated_image, crop_x1, crop_y1, crop_x2, crop_y2)
        manipulated_image = enhance_texture(manipulated_image, texture_strength)
        manipulated_image = enhance_clarity(manipulated_image, clarity_strength)
        manipulated_image = add_grain(manipulated_image, grain_amount)
        manipulated_image = enhance_detail(manipulated_image, detail_strength)

        # Display manipulated image
        display_image(manipulated_image, "Manipulated Image")

        # Export final image
        if st.button(f"Download Final Image for {file.name}"):
            final_image = export_image(manipulated_image)
            st.download_button(label="Download Image", data=final_image, file_name=f"manipulated_output_{file.name}", mime="image/jpeg")

        # Clear memory after processing each image
        del image, manipulated_image
        gc.collect()
