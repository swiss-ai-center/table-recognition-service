import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
from model.main_ import main as main_model
from utils import custom_parse_args, save_image, prepare_zip_result
import json
import shutil

settings = get_settings()


class MyService(Service):

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Table Recognition Service",
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
            vis_font_path="Fonts/Arial.ttf",
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


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)



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
- Training Dataset: [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) 
+ [SynthTabNet](https://github.com/IBM/SynthTabNet)
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

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="Table Recognition API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
