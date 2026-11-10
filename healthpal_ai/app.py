from contextlib import asynccontextmanager
import json
import logging
import re
import time
import uuid
from typing import Annotated
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from .policy import Policy
from .schemas import ComparisonResult, ErrorResponse, RecognitionResult
from .service import RecognitionService, ServiceError

logger = logging.getLogger("healthpal_ai")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.propagate = False


def error_response(request, code, message, status_code, retryable=False):
    result = ErrorResponse(request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
                           error=dict(code=code, message=message, retryable=retryable))
    return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))


def create_app(service_factory=RecognitionService):
    @asynccontextmanager
    async def lifespan(app):
        app.state.recognizer = service_factory()
        try:
            await run_in_threadpool(app.state.recognizer.start)
        except Exception:
            logger.exception("Model initialization failed")
        yield

    app = FastAPI(title="HealthPal AI Service", version="1.0.0", lifespan=lifespan,
                  description="Frozen 92-output detector and P0/P1/P2 policies for a ten-food catalogue. Suggestions require user confirmation.")

    @app.middleware("http")
    async def trace(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID")
        request.state.request_id = supplied if supplied and re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", supplied) else str(uuid.uuid4())
        start = time.perf_counter()
        if supplied and supplied != request.state.request_id:
            response = error_response(request, "invalid_request_id", "X-Request-ID must contain 1–64 letters, digits, dots, underscores, colons or hyphens.", 400)
        else:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("Recognition request failed", extra={"request_id": request.state.request_id})
                response = error_response(request, "internal_error", "Recognition failed; retry or select food manually.", 500, True)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(json.dumps({"request_id": request.state.request_id, "path": request.url.path,
                               "http_status": response.status_code, "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}))
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return error_response(request, error.code, str(error), error.status_code, error.retryable)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return error_response(request, "invalid_request", "Attach an image file and choose policy P0, P1 or P2.", 422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, error):
        codes = {400: "invalid_request", 404: "not_found", 405: "method_not_allowed"}
        response = error_response(request, codes.get(error.status_code, "http_error"),
                                  "The request could not be processed; check the endpoint, method and upload format.", error.status_code)
        if error.headers:
            response.headers.update(error.headers)
        return response

    @app.get("/health/live")
    async def live():
        return {"status": "alive", "schema_version": "1.0"}

    @app.get("/health/ready", responses={503: {"model": ErrorResponse}})
    async def ready(request: Request):
        service = request.app.state.recognizer
        if not service.ready:
            raise ServiceError("model_not_ready", "Model is not ready.", 503, True)
        return {"status": "ready", "model_version": service.manifest["model_version"], "model_sha256": service.manifest["sha256"],
                "policy_version": service.config["policy_version"]}

    errors = {400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
              422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}

    async def read_upload(request, file):
        service = request.app.state.recognizer
        try:
            raw = await file.read(service.max_bytes + 1)
        finally:
            await file.close()
        return service, raw

    @app.post("/v1/recognize", response_model=RecognitionResult, responses=errors)
    async def recognize(request: Request, file: Annotated[UploadFile, File()], policy: Annotated[Policy, Form()] = Policy.P1):
        service, raw = await read_upload(request, file)
        return await run_in_threadpool(service.recognize, raw, request.state.request_id, policy, False)

    @app.post("/v1/compare", response_model=ComparisonResult, responses=errors)
    async def compare(request: Request, file: Annotated[UploadFile, File()]):
        service, raw = await read_upload(request, file)
        return await run_in_threadpool(service.recognize, raw, request.state.request_id, Policy.P1, True)

    return app


app = create_app()
