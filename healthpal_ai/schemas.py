from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .policy import Policy


class Status(str, Enum):
    accepted = "accepted"
    uncertain = "uncertain"
    no_detection = "no_detection"


class Candidate(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    class_id: int = Field(ge=0, le=91)
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: tuple[float, float, float, float]


class ImageInfo(BaseModel):
    width: int
    height: int
    coordinate_space: Literal["decoded_exif_oriented_image"] = "decoded_exif_oriented_image"


class PolicyOutcome(BaseModel):
    policy: Policy
    status: Status
    reason: str
    suggestion: Candidate | None


class CommonResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    model_version: str
    model_sha256: str
    policy_version: str
    catalogue_version: str
    image: ImageInfo
    candidate: Candidate | None
    # Passing a rule does not establish correctness or permission to save.
    user_confirmation_required: Literal[True] = True
    original_inference_passes: Literal[1] = 1
    brightened_inference_passes: int = Field(ge=0, le=1)
    processing_ms: float = Field(ge=0)


class RecognitionResult(CommonResult, PolicyOutcome):
    pass


class ComparisonResult(CommonResult):
    results: list[PolicyOutcome]


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool


class ErrorResponse(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    error: ErrorDetail
