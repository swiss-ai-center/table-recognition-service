from paddleocr.ppstructure.utility import parse_args
import os
import io
import numpy as np
import cv2
import tempfile
import zipfile


def prepare_zip_result(input_dir):
    # Create a temporary directory to store the ZIP contents
    with tempfile.TemporaryDirectory():
        # Define the in-memory buffer for the ZIP file
        zip_buffer = io.BytesIO()

        # Create the ZIP archive in the buffer
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zipper:
            # Traverse the input directory and add only .xlsx files to the ZIP
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith('.xlsx'):
                        # Full path of the file
                        file_path = os.path.join(root, file)
                        # Add the file to the ZIP archive with relative path
                        zipper.write(file_path, arcname=os.path.relpath(file_path, input_dir))

        # Prepare the buffer for reading
        zip_buffer.seek(0)
        return zip_buffer.read()


# Function to create ZIP file containing only .xlsx files
def zip_xlsx_files(input_dir, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.xlsx'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, input_dir))
    return output_zip_path


def custom_parse_args(**kwargs):
    # Temporarily override `sys.argv` if necessary
    import sys
    original_argv = sys.argv
    sys.argv = ["main.py"] + [f"--{k}={v}" for k, v in kwargs.items()]

    args = parse_args()

    # Restore original argv
    sys.argv = original_argv
    return args


def save_image(data, output_dir="img_dir"):
    """
    Saves a single image from the data dictionary to the specified directory.

    Args:
    - data: dict, the data dictionary containing the image bytes.
    - output_dir: str, the directory where the image will be saved.

    Returns:
    - The file path to the saved image.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the image bytes from data
    image_bytes = data["image"].data  # Extract the raw bytes of the image
    input_type = data["image"].type

    # Decode the image from bytes
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)

    # Define the path where the image will be saved
    image_path = os.path.join(output_dir, "image.png")

    # Save the image to the specified path
    cv2.imwrite(image_path, img)

    return img, input_type
