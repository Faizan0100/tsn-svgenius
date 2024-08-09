import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2
import tempfile
from potrace import Bitmap, POTRACE_TURNPOLICY_BLACK
from transformers import pipeline
import torch
from diffusers import DiffusionPipeline

# Check CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Define options
dimensional_options = ["2D", "3D"]

def create_prompt(prompt, cartoon, fourk, dimensional_option):
    options = []
    if cartoon:
        options.append("cartoon")
    if fourk:
        options.append("4k")
    if dimensional_option:
        options.append(dimensional_option.lower())
    return f"{prompt}, {' '.join(options)}"

@st.cache_resource
def load_diffusion_pipeline():
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            variant="fp16" if DEVICE == "cuda" else None,
            disable_progress_bar=True
        ).to(DEVICE)
        return pipe
    except Exception as e:
        st.error(f"Error loading DiffusionPipeline: {e}")
        return None

def generate_image(prompt, cartoon, fourk, dimensional_option, num_inference_steps):
    pipe = load_diffusion_pipeline()
    if pipe is None:
        return None
    
    combined_prompt = create_prompt(prompt, cartoon, fourk, dimensional_option)
    
    try:
        image = pipe(prompt=combined_prompt, num_inference_steps=num_inference_steps, guidance_scale=3).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

@st.cache_resource
def load_background_removal_model():
    try:
        return pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    except Exception as e:
        st.error(f"Error loading background removal model: {e}")
        return None

def remove_background(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Input image must be a PIL image")
    
    rmbg_pipe = load_background_removal_model()
    if rmbg_pipe is None:
        return image
    
    result = rmbg_pipe(image)
    if isinstance(result, Image.Image):
        return result
    if 'image' in result:
        return result['image']
    elif 'path' in result:
        return Image.open(result['path'])
    raise ValueError("Unexpected result type from background removal pipeline")

def enhance_edges(image, dilation_iterations=2, canny_threshold1=50, canny_threshold2=150, blur_ksize=5, erosion_iterations=1):
    gray_image = np.array(image.convert("L"))
    blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred_image, threshold1=canny_threshold1, threshold2=canny_threshold2)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilation_iterations)
    refined_edges = cv2.erode(dilated_edges, np.ones((3, 3), np.uint8), iterations=erosion_iterations)
    edges_image = Image.fromarray(refined_edges)
    return edges_image

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
    prompt = st.text_input("Enter your prompt:")
    
    # Options for image generation
    cartoon = st.checkbox("Cartoon")
    fourk = st.checkbox("4K")
    dimensional_option = st.selectbox("Dimensional option:", dimensional_options)
    
    num_inference_steps = st.slider("Number of inference steps (20-50)", min_value=20, max_value=50, value=50)
    
    st.subheader(f"Inference Steps: {num_inference_steps}")
    st.write("Advantages and Disadvantages:")
    if num_inference_steps < 35:
        st.write("- Faster generation, but potentially lower quality")
    elif num_inference_steps > 35:
        st.write("- Higher quality, but longer generation time")
    else:
        st.write("- Balanced approach between speed and quality")

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            generated_image = generate_image(prompt, cartoon, fourk, dimensional_option, num_inference_steps)
        if generated_image is not None:
            st.session_state.original_image = generated_image
            st.image(generated_image, caption="Generated Image", use_column_width=True)

if "original_image" in st.session_state:
    image = st.session_state.original_image
    if st.button("Remove Background"):
        with st.spinner("Removing background..."):
            image = remove_background(image)
        st.session_state.bg_removed_image = image
        st.image(image, caption="Image with Background Removed", use_column_width=True)

    if st.button("Show Enhanced Image"):
        with st.spinner("Enhancing image..."):
            if "bg_removed_image" in st.session_state:
                image = st.session_state.bg_removed_image
            enhanced_image = enhance_edges(image)
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

    svg_style = st.radio("Select SVG Style", ("Black and White", "Filled"))
    if st.button("Convert to SVG"):
        with st.spinner("Converting to SVG..."):
            if "bg_removed_image" in st.session_state:
                image = st.session_state.bg_removed_image
            else:
                image = st.session_state.original_image

            fill_type = "bw" if svg_style == "Black and White" else "filled"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.svg') as tmp_file:
                svg_filename = os.path.basename(tmp_file.name)
                svg_path = file_to_svg(image, svg_filename, os.path.dirname(tmp_file.name), fill_type)
            
            with open(svg_path, "r") as file:
                svg_content = file.read()
            
            st.download_button(
                label="Download SVG",
                data=svg_content,
                file_name=f"{svg_filename}_{fill_type}.svg",
                mime="image/svg+xml"
            )
            
            os.unlink(svg_path)

        st.success(f"SVG conversion complete. Click the 'Download SVG' button to save the file.")