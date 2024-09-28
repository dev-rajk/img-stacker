import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rawpy import imread as rawpy_read
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
from skimage.restoration import denoise_wavelet
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
from io import BytesIO

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

# Wavelet Denoising
def denoise_wavelet_image(image, wavelet='db1', level=2):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    coeffs_filtered = map(lambda x: pywt.threshold(x, value=10, mode='soft'), coeffs)
    denoised = pywt.waverec2(list(coeffs_filtered), wavelet=wavelet)
    return denoised

# Anisotropic Diffusion (Perona-Malik Filtering)
def denoise_anisotropic_diffusion(image, weight=0.1):
    denoised = denoise_tv_chambolle(image, weight=weight, multichannel=True)
    return denoised

# Total Variation Denoising
def denoise_total_variation(image, weight=0.1):
    denoised = denoise_tv_bregman(image, weight=weight, multichannel=True)
    return denoised

# Median Filtering
def denoise_median_filter(image, kernel_size=3):
    denoised = cv2.medianBlur(image.astype(np.uint8), kernel_size)
    return denoised

# Gaussian Blurring
def denoise_gaussian_blur(image, kernel_size=(5,5), sigma=0):
    denoised = cv2.GaussianBlur(image, kernel_size, sigma)
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

    # Load all images
    light_images = [load_image(f) for f in light_frames]
    bias_images = [load_image(f) for f in bias_frames]
    dark_images = [load_image(f) for f in dark_frames]
    flat_images = [load_image(f) for f in flat_frames]
    dark_flat_images = [load_image(f) for f in dark_flat_frames] if dark_flat_frames else None

    # Processing: Align, calibrate, and stack images (assuming you have this function implemented)
    processed_image = process_deep_sky_images(light_images, bias_images[0], dark_images[0], flat_images[0], dark_flat_images[0] if dark_flat_images else None)

    # Show the stacked image
    display_image(processed_image, "Processed Deep Sky Image")

    # Add manipulation options
    st.sidebar.title("Image Manipulation Tools")
    brightness = st.sidebar.slider("Brightness", -100.0, 100.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)

    # Apply adjustments to the processed image
    manipulated_image = adjust_image(processed_image, brightness, contrast)
    display_image(manipulated_image, "Manipulated Image")

    # Noise Reduction Options
    st.sidebar.title("Noise Reduction Methods")
    denoising_method = st.sidebar.selectbox(
        "Choose a denoising method:",
        ("None", "Non-Local Means", "Wavelet Denoising", "Anisotropic Diffusion", "Total Variation Denoising", "Median Filter", "Gaussian Blur")
    )

    if denoising_method == "Non-Local Means":
        h = st.sidebar.slider("Filter Strength (h)", 1, 20, 10)
        hColor = st.sidebar.slider("Color Filter Strength (hColor)", 1, 20, 10)
        manipulated_image = denoise_non_local_means(manipulated_image, h=h, hColor=hColor)
    elif denoising_method == "Wavelet Denoising":
        manipulated_image = denoise_wavelet_image(manipulated_image)
    elif denoising_method == "Anisotropic Diffusion":
        weight = st.sidebar.slider("Weight", 0.01, 0.2, 0.1)
        manipulated_image = denoise_anisotropic_diffusion(manipulated_image, weight=weight)
    elif denoising_method == "Total Variation Denoising":
        weight = st.sidebar.slider("Weight", 0.01, 0.2, 0.1)
        manipulated_image = denoise_total_variation(manipulated_image, weight=weight)
    elif denoising_method == "Median Filter":
        kernel_size = st.sidebar.slider("Kernel Size", 1, 11, 3, step=2)
        manipulated_image = denoise_median_filter(manipulated_image, kernel_size=kernel_size)
    elif denoising_method == "Gaussian Blur":
        kernel_size = st.sidebar.slider("Kernel Size", 1, 11, 5, step=2)
        sigma = st.sidebar.slider("Sigma", 0, 10, 0)
        manipulated_image = denoise_gaussian_blur(manipulated_image, kernel_size=(kernel_size, kernel_size), sigma=sigma)

    display_image(manipulated_image, "Denoised Image")

    # Export the final image
    if st.button("Download Final Image"):
        final_image = export_image(manipulated_image)
        st.download_button(label="Download Processed Image", data=final_image, file_name="deep_sky_output.jpg", mime="image/jpeg")

# Helper functions for image processing pipeline
def subtract_bias(image, bias_frame):
    return cv2.subtract(image, bias_frame)

def subtract_dark(image, dark_frame):
    return cv2.subtract(image, dark_frame)

def apply_flat_correction(image, flat_frame, dark_flat_frame=None):
    if dark_flat_frame is not None:
        flat_frame = subtract_dark(flat_frame, dark_flat_frame)
    normalized_flat = flat_frame / np.mean(flat_frame)
    return image / normalized_flat

def align_images(images):
    reference_image = images[0]
    aligned_images = [reference_image]
    for image in images[1:]:
        shift, error, _ = phase_cross_correlation(reference_image, image)
        aligned_image = np.roll(image, shift.astype(int), axis=(0, 1))
        aligned_images.append(aligned_image)
    return aligned_images

def stack_images(images, method='median'):
    stack = np.stack(images, axis=0)
    if method == 'median':
        return np.median(stack, axis=0)
    elif method == 'mean':
        return np.mean(stack, axis=0)
    else:
        raise ValueError("Unknown stacking method: choose 'median' or 'mean'")

def process_deep_sky_images(light_frames, bias_frame, dark_frame, flat_frame, dark_flat_frame=None, method='median'):
    calibrated_images = []
    for light in light_frames:
        bias_subtracted = subtract_bias(light, bias_frame)
        dark_subtracted = subtract_dark(bias_subtracted, dark_frame)
        calibrated = apply_flat_correction(dark_subtracted, flat_frame, dark_flat_frame)
        calibrated_images.append(calibrated)
    
    aligned_images = align_images(calibrated_images)
    stacked_image = stack_images(aligned_images, method=method)
    return stacked_image
