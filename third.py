import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rawpy import imread as rawpy_read
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
from skimage.restoration import denoise_wavelet
from skimage.registration import phase_cross_correlation
from io import BytesIO
import gc

# Function to load and convert images
def load_image(image_file):
    if image_file.name.lower().endswith('.dng'):
        raw = rawpy_read(image_file)
        image = raw.postprocess()  # Convert raw DNG to a usable format
    else:
        image = Image.open(image_file)
        image = np.array(image)
    return image

# Image Manipulation: Brightness/Contrast Adjustment
def adjust_image(image, brightness=1.0, contrast=1.0):
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return image

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

# Non-Local Means Denoising
def denoise_non_local_means(image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    denoised = cv2.fastNlMeansDenoisingColored(image.astype(np.uint8), None, h, hColor, templateWindowSize, searchWindowSize)
    return denoised

# Streamlit UI
st.title("Deep Sky Image Processor")

# File uploader for light, bias, dark, flat, dark flat frames
light_frames = st.file_uploader("Upload Light Frames (DNG or JPEG)", type=["dng", "jpeg", "jpg"], accept_multiple_files=True)
bias_frames = st.file_uploader("Upload Bias Frames (DNG or JPEG)", type=["dng", "jpeg", "jpg"], accept_multiple_files=True)
dark_frames = st.file_uploader("Upload Dark Frames (DNG or JPEG)", type=["dng", "jpeg", "jpg"], accept_multiple_files=True)
flat_frames = st.file_uploader("Upload Flat Frames (DNG or JPEG)", type=["dng", "jpeg", "jpg"], accept_multiple_files=True)
dark_flat_frames = st.file_uploader("Upload Dark Flat Frames (DNG or JPEG)", type=["dng", "jpeg", "jpg"], accept_multiple_files=True)

# Process if light frames are uploaded
if light_frames and bias_frames and dark_frames and flat_frames:
    st.write("Processing the images...")

    # Load bias and dark frames once
    bias_image = load_image(bias_frames[0])
    dark_image = load_image(dark_frames[0])
    flat_image = load_image(flat_frames[0])
    dark_flat_image = load_image(dark_flat_frames[0]) if dark_flat_frames else None

    # List to hold calibrated images
    calibrated_images = []

    for light_frame in light_frames:
        light_image = load_image(light_frame)
        bias_subtracted = cv2.subtract(light_image, bias_image)
        dark_subtracted = cv2.subtract(bias_subtracted, dark_image)

        # Apply flat correction
        if dark_flat_image is not None:
            flat_image = cv2.subtract(flat_image, dark_flat_image)

        normalized_flat = flat_image / np.mean(flat_image)
        calibrated_image = dark_subtracted / normalized_flat

        calibrated_images.append(calibrated_image)

        # Clear memory
        del light_image
        gc.collect()

    # Align images and stack them
    reference_image = calibrated_images[0]
    aligned_images = [reference_image]

    for image in calibrated_images[1:]:
        shift, error, _ = phase_cross_correlation(reference_image, image)
        aligned_image = np.roll(image, shift.astype(int), axis=(0, 1))
        aligned_images.append(aligned_image)

    # Stack images (mean or median)
    stacked_image = np.median(np.stack(aligned_images), axis=0).astype(np.uint8)

    # Show the stacked image
    display_image(stacked_image, "Processed Deep Sky Image")

    # Add manipulation options
    st.sidebar.title("Image Manipulation Tools")
    brightness = st.sidebar.slider("Brightness", -100.0, 100.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)

    # Apply adjustments to the processed image
    manipulated_image = adjust_image(stacked_image, brightness, contrast)
    display_image(manipulated_image, "Manipulated Image")

    # Noise Reduction Options
    st.sidebar.title("Noise Reduction Methods")
    denoising_method = st.sidebar.selectbox(
        "Choose a denoising method:",
        ("None", "Non-Local Means")
    )

    if denoising_method == "Non-Local Means":
        h = st.sidebar.slider("Filter Strength (h)", 1, 20, 10)
        hColor = st.sidebar.slider("Color Filter Strength (hColor)", 1, 20, 10)
        manipulated_image = denoise_non_local_means(manipulated_image, h=h, hColor=hColor)

    display_image(manipulated_image, "Denoised Image")

    # Export the final image
    if st.button("Download Final Image"):
        final_image = export_image(manipulated_image)
        st.download_button(label="Download Processed Image", data=final_image, file_name="deep_sky_output.jpg", mime="image/jpeg")

    # Clear memory after processing
    del stacked_image, manipulated_image
    gc.collect()
