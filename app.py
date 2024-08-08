import streamlit as st
from PIL import Image
import os
import numpy as np
import cv2
from potrace import Bitmap, POTRACE_TURNPOLICY_BLACK
from transformers import pipeline
import torch
from diffusers import DiffusionPipeline

# Load the image-segmentation pipeline for background removal
try:
    rmbg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
except Exception as e:
    st.error(f"Error loading background removal model: {e}")

def remove_background(image):
    # Save the uploaded image temporarily
    temp_input_path = "temp_image.png"
    image.save(temp_input_path)

    # Use the segmentation pipeline to remove background
    result = rmbg_pipe(temp_input_path)
    
    # If result is directly an image object, return it
    if isinstance(result, Image.Image):
        return result

    # Otherwise, handle other types of result (e.g., if it's a dictionary)
    if 'image' in result:
        return result['image']
    elif 'path' in result:
        return Image.open(result['path'])

    # Handle unexpected result types
    raise ValueError("Unexpected result type from background removal pipeline")

def enhance_edges(image, dilation_iterations=2, canny_threshold1=50, canny_threshold2=150, blur_ksize=5, erosion_iterations=1):
    gray_image = np.array(image.convert("L"))
    blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred_image, threshold1=canny_threshold1, threshold2=canny_threshold2)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilation_iterations)
    refined_edges = cv2.erode(dilated_edges, np.ones((3, 3), np.uint8), iterations=erosion_iterations)
    edges_image = Image.fromarray(refined_edges)
    return edges_image

def get_unique_filename(directory: str, file_extension: str, fill_type: str):
    # Create a pattern to match filenames
    pattern = f"*{fill_type}{file_extension}"
    existing_files = [f for f in os.listdir(directory) if f.endswith(file_extension) and fill_type in f]

    # Extract the numbers from filenames and ignore non-integer filenames
    numbers = []
    for f in existing_files:
        try:
            # Extract number from the filename
            number = int(os.path.splitext(f)[0].split('_')[0])
            numbers.append(number)
        except ValueError:
            continue  # Ignore files that don't start with a number
    
    if not numbers:
        return "0"
    
    max_number = max(numbers)
    return str(max_number + 1)

def file_to_svg(image, filename: str, output_dir: str, fill_type: str):
    enhanced_image = enhance_edges(image)
    bitmap_data = np.array(enhanced_image, dtype=np.uint8)
    bitmap = Bitmap(bitmap_data)
    path = bitmap.trace(
        turdsize=1,
        turnpolicy=POTRACE_TURNPOLICY_BLACK,
        alphamax=0.5,
        opticurve=False,
        opttolerance=0.1
    )
    
    output_path = os.path.join(output_dir, f"{filename}_{fill_type}.svg")
    with open(output_path, "w") as fp:
        fp.write(
            f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{bitmap_data.shape[1]}" height="{bitmap_data.shape[0]}">')
        for curve in path:
            parts = []
            fs = curve.start_point
            parts.append(f"M{fs.x},{fs.y}")
            for segment in curve.segments:
                if segment.is_corner:
                    a = segment.c
                    b = segment.end_point
                    parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                else:
                    a = segment.c1
                    b = segment.c2
                    c = segment.end_point
                    parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")
            parts.append("z")
            if fill_type == "bw":
                fp.write(f'<path d="{"".join(parts)}" fill="none" stroke="black" stroke-width="1"/>')
            elif fill_type == "filled":
                fp.write(f'<path d="{"".join(parts)}" fill="black" stroke="none" stroke-width="1"/>')
        fp.write("</svg>")
    return output_path

def generate_image(prompt):
    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]
    return image

# Initialize session state for SVG file counter
if 'svg_counter' not in st.session_state:
    st.session_state.svg_counter = 0

# Streamlit app
st.title("Image to SVG Converter")

image_source = st.radio("", ("Upload Image", "Generate Image"))

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.session_state.original_image = uploaded_image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

elif image_source == "Generate Image":
    prompt = st.text_input("Enter a prompt for image generation")
    if st.button("Generate Image"):
        generated_image = generate_image(prompt)
        st.session_state.original_image = generated_image
        st.image(generated_image, caption="Generated Image", use_column_width=True)

if "original_image" in st.session_state:
    image = st.session_state.original_image
    if st.button("Remove Background"):
        image = remove_background(image)
        st.session_state.bg_removed_image = image
        st.image(image, caption="Image with Background Removed", use_column_width=True)

    if st.button("Show Enhanced Image"):
        if "bg_removed_image" in st.session_state:
            image = st.session_state.bg_removed_image
        enhanced_image = enhance_edges(image)
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

    if st.checkbox("Download Image"):
        filename = get_unique_filename(os.path.expanduser("~/Downloads"), ".png")
        downloads_output_path = os.path.expanduser("~/Downloads")
        os.makedirs(downloads_output_path, exist_ok=True)
        temp_image_path = os.path.join(downloads_output_path, f"{filename}.png")
        image.save(temp_image_path)

        with open(temp_image_path, "rb") as file:
            btn = st.download_button(
                label="Download Image",
                data=file,
                file_name=f"{filename}.png",
                mime="image/png"
            )

    # Add option to choose SVG style
    svg_style = st.radio("Select SVG Style", ("Black and White", "Filled"))
    if st.button("Convert to SVG"):
        if "bg_removed_image" in st.session_state:
            image = st.session_state.bg_removed_image

        # Get the fill type based on the selected style
        fill_type = "bw" if svg_style == "Black and White" else "filled"
        
        # Get a unique number for the SVG file
        downloads_output_path = os.path.expanduser("~/Downloads")
        svg_filename = get_unique_filename(downloads_output_path, ".svg", fill_type)
        
        # Save the SVG file
        svg_path = file_to_svg(image, svg_filename, downloads_output_path, fill_type)
        
        st.success(f"SVG saved as {svg_filename}_{fill_type}.svg in the Downloads folder.")
        with open(svg_path, "rb") as file:
            st.download_button(
                label="Download SVG",
                data=file,
                file_name=f"{svg_filename}_{fill_type}.svg",
                mime="image/svg+xml"
            )
