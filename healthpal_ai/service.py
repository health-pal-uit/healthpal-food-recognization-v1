from pathlib import Path
import hashlib
import io
import json
import os
import threading
import time
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
from .policy import Policy, decide
from .schemas import Candidate, ImageInfo, PolicyOutcome, RecognitionResult, ComparisonResult

ROOT = Path(__file__).resolve().parents[1]


class ServiceError(Exception):
    def __init__(self, code, message, status_code=400, retryable=False):
        super().__init__(message)
        self.code, self.status_code, self.retryable = code, status_code, retryable


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def brighten(image, factor):
    data = np.asarray(image.convert("YCbCr")).copy()
    data[:, :, 0] = np.clip(np.rint(data[:, :, 0].astype(float) * factor), 0, 255).astype("uint8")
    return Image.fromarray(data, mode="YCbCr").convert("RGB")


class YoloDetector:
    """One verified model per service worker; callers serialize predictor access."""
    def __init__(self, manifest, config):
        import torch
        import ultralytics
        from ultralytics import YOLO
        if ultralytics.__version__ != manifest["ultralytics_version"]:
            raise RuntimeError("Ultralytics version differs from the frozen runtime")
        path = Path(os.environ.get("HEALTHPAL_MODEL_PATH", ROOT / manifest["local_file"]))
        if not path.is_file() or sha256(path) != manifest["sha256"]:
            raise RuntimeError("Selected checkpoint missing or checksum mismatch")
        self.device = os.environ.get("HEALTHPAL_DEVICE", "cpu")
        if self.device == "cpu":
            torch.set_num_threads(4)
        elif not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA device is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = YOLO(str(path))
        if [self.model.names[i] for i in range(len(self.model.names))] != manifest["names"]:
            raise RuntimeError("Checkpoint class catalogue differs from manifest")
        self.args = {**config["prediction"], "device": self.device, "verbose": False}
        warmup = Image.fromarray(np.random.default_rng(20261108).integers(0, 256, (640, 640, 3), dtype=np.uint8), "RGB")
        for _ in range(3):
            self.model.predict(warmup, **self.args)

    def predict(self, image):
        result = self.model.predict(image, **self.args)[0]
        if result.boxes is None or not len(result.boxes):
            return None
        top = int(result.boxes.conf.argmax())
        return {"class_id": int(result.boxes.cls[top]), "confidence": float(result.boxes.conf[top]),
                "xyxy": list(map(float, result.boxes.xyxy[top].cpu().tolist()))}


class RecognitionService:
    def __init__(self, detector_factory=YoloDetector):
        self.manifest = json.loads((ROOT / "models/model-manifest.json").read_text(encoding="utf-8"))
        self.config = json.loads((ROOT / "config/policy-v1.json").read_text(encoding="utf-8"))
        if sha256(ROOT / "config/policy-v1.json") != self.manifest["policy_config_sha256"]:
            raise RuntimeError("Frozen policy configuration checksum mismatch")
        self.detector_factory = detector_factory
        self.detector = None
        self.lock = threading.Lock()
        self.max_bytes = 10 * 1024 * 1024
        self.max_pixels = 20_000_000

    def start(self):
        self.detector = self.detector_factory(self.manifest, self.config)

    @property
    def ready(self):
        return self.detector is not None

    def decode(self, raw):
        if not raw:
            raise ServiceError("empty_image", "Attach a non-empty image.")
        if len(raw) > self.max_bytes:
            raise ServiceError("image_too_large", "Image must be at most 10 MiB.", 413)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format not in {"JPEG", "PNG", "WEBP"}:
                        raise ServiceError("unsupported_image", "Use JPEG, PNG or WebP.", 415)
                    if image.width * image.height > self.max_pixels:
                        raise ServiceError("image_too_large", "Image dimensions exceed 20 megapixels.", 413)
                    # Geometry is reported in this oriented image's coordinates.
                    return ImageOps.exif_transpose(image).convert("RGB")
        except ServiceError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ServiceError("image_too_large", "Image dimensions are too large.", 413)
        except (UnidentifiedImageError, OSError, ValueError):
            raise ServiceError("invalid_image", "The uploaded file cannot be decoded as an image.")

    def candidate(self, record):
        if record is None:
            return None
        return Candidate(class_id=record["class_id"], label=self.manifest["names"][record["class_id"]],
                         confidence=record["confidence"], bbox_xyxy=record["xyxy"])

    def recognize(self, raw, request_id, policy=Policy.P1, compare=False):
        if not self.ready:
            raise ServiceError("model_not_ready", "Recognition is unavailable; try again or select food manually.", 503, True)
        start = time.perf_counter()
        image = self.decode(raw)
        bright = None
        with self.lock:
            original = self.detector.predict(image)
            need_bright = (compare or policy == Policy.P2) and decide(original, None, Policy.P1, self.config).status == "accepted"
            if need_bright:
                bright = self.detector.predict(brighten(image, self.config["brightness_factor"]))
        candidate = self.candidate(original)
        def outcome(p):
            answer = decide(original, bright, p, self.config)
            return PolicyOutcome(policy=p, status=answer.status, reason=answer.reason,
                                 suggestion=candidate if answer.status == "accepted" else None)
        outcomes = [outcome(p) for p in Policy] if compare else [outcome(policy)]
        common = dict(request_id=request_id, model_version=self.manifest["model_version"], model_sha256=self.manifest["sha256"],
                      policy_version=self.config["policy_version"], catalogue_version=self.config["catalogue_version"],
                      image=ImageInfo(width=image.width, height=image.height), candidate=candidate,
                      brightened_inference_passes=int(need_bright), processing_ms=(time.perf_counter() - start) * 1000)

        if compare:
            return ComparisonResult(**common, results=outcomes)
        return RecognitionResult(**common, **outcomes[0].model_dump())
