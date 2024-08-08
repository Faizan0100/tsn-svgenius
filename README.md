# Image to SVG Converter

## Overview

The Image to SVG Converter is a powerful Streamlit application that allows users to convert images into Scalable Vector Graphics (SVG) format. This tool offers various features including background removal, edge enhancement, and the ability to generate images from text prompts.

## Features

- **Image Upload**: Users can upload their own images in PNG, JPG, or JPEG format.
- **Image Generation**: Generate images from text prompts using AI.
- **Background Removal**: Automatically remove the background from uploaded images.
- **Edge Enhancement**: Improve edge detection for better SVG conversion.
- **SVG Conversion**: Convert images to SVG format with options for filled or outline styles.
- **Download Options**: Download processed images and SVG files.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Faizan0100/svgenius.git
   cd svgenius
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: This project requires Python 3.7 or later.

3. Ensure you have the necessary models and libraries:
   - RMBG-1.4 model for background removal
   - playground-v2.5-1024px-aesthetic model for image generation

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Use the interface to:
   - Upload an image or generate one from a text prompt
   - Remove the background (if desired)
   - Enhance edges (optional)
   - Convert the image to SVG
   - Download the processed image or SVG file

## Dependencies

- streamlit
- Pillow
- numpy
- opencv-python
- potrace
- transformers
- torch
- diffusers

## Notes

- The background removal feature uses the RMBG-1.4 model.
- Image generation is powered by the playground-v2.5-1024px-aesthetic model.
- SVG files are saved in the user's Downloads folder.

## Contributing

Contributions to improve the Image to SVG Converter are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## Acknowledgements

- Thanks to the creators of the RMBG-1.4 and playground-v2.5-1024px-aesthetic models.
- This project uses several open-source libraries. We're grateful to their maintainers and contributors.
