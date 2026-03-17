from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
from model.main_ import main as main_model
import json
import shutil
from paddleocr.ppstructure.utility import parse_args
import os
import io
import numpy as np
import cv2
import tempfile
import zipfile

api_description = """
Inputs:
- Document Image: An image of the document containing tables (JPEG, PNG).
- Layout Analysis Results: JSON results from a prior layout analysis model, which provides bounding boxes (bboxes)
    for potential tables in the document.

Outputs:
- A ZIP file containing all the detected tables in CSV format.

Model Specifications:
- Model: SLANet
- Version: 1 M
- Training Dataset: [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) +
[SynthTabNet](https://github.com/IBM/SynthTabNet)
- Model Size: 9.2 MB
- Reference : [SLANet](https://github.com/PaddlePaddle/PaddleOCR)

Capabilities:

    Processes bounding boxes provided by the layout model to crop the regions of interest.
    Extracts table content and structure from cropped images.
    Generates well-structured CSV files from table data.
"""
api_summary = """ Table recognition service processes document-based input
and utilizes a newly trained SLANet from PaddleOCR for robust table recognition.
"""

api_title = "Table Recognition API."
version = "0.0.1"

settings = get_settings()


class MyService(Service):

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Table Recognition",
            slug="table-recognition-service",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_PNG,
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
                FieldDescription(
                    name="layout",
                    type=[
                        FieldDescriptionType.APPLICATION_JSON,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_ZIP]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            # OPTIONAL: CHANGE THE DOCS URL TO YOUR SERVICE'S DOCS
            docs_url="https://docs.swiss-ai-center.ch/reference/core-concepts/service/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        # The objects in the data variable are always bytes. It is necessary to convert them to the desired type
        # before using them.
        args = custom_parse_args(
            vis_font_path="Fonts/arial.ttf",
            use_gpu=False,
            image_dir="img_dir",
            det_model_dir="model/inference_table/en_PP-OCRv3_det_infer",
            rec_model_dir="model/inference_table/en_PP-OCRv3_rec_infer",
            table_model_dir="model/inference_table/model_final",
            rec_char_dict_path="model/dict_table/en_dict.txt",
            table_char_dict_path="model/dict_table/table_structure_dict.txt",
            output="../output",
            layout=False,
        )

        save_image(data)
        layout_res = json.loads(data["layout"].data)
        main_model(args, layout_res)
        zip_data = prepare_zip_result("../output/structure/image")

        shutil.rmtree("img_dir")
        shutil.rmtree("../output")
        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(data=zip_data, type=FieldDescriptionType.APPLICATION_ZIP)
        }

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