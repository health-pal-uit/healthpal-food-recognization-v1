# AI recognition contract — schema 1.0

This is the implemented local AI contract. NestJS and Flutter adapters remain the
next integration milestone. Model quality and application end-to-end performance
are separate from contract verification.

## Requests

| Endpoint | Purpose |
|---|---|
| `POST /v1/recognize` | One selected policy outcome. |
| `POST /v1/compare` | All three outcomes, reusing the same detector passes. |
| `GET /health/live` | Process is responding. |
| `GET /health/ready` | Verified model is loaded and ready. |

Recognition requests use `multipart/form-data` with a binary `file` field.
`/v1/recognize` also accepts `policy`: `P0`, `P1` or `P2`; default `P1`.
The upload API follows FastAPI's [files](https://fastapi.tiangolo.com/tutorial/request-files/)
and [combined file/form](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/)
interfaces. A JSON body containing image bytes is not this contract.

`X-Request-ID` is optional: 1–64 letters/digits or `. _ : -`. A UUID is generated
when absent. The same ID is returned in the response header/body and request log.
JPEG, PNG and WebP are supported; files must be at most 10 MiB and 20 megapixels.
Actual image decoding determines format, not the filename or claimed MIME type.

## Successful response fields

| Field | Meaning |
|---|---|
| `schema_version` | JSON contract version, currently `1.0`. |
| `request_id` | Correlation across request, response and logs. |
| `model_version`, `model_sha256` | Exact selected checkpoint identity. |
| `policy`, `policy_version` | Rule applied and frozen parameter version. |
| `catalogue_version` | The ten-food application catalogue. |
| `status` | `accepted`, `uncertain`, or `no_detection`. |
| `reason` | Machine-readable explanation below. |
| `image.width`, `image.height` | Decoded image dimensions after EXIF orientation. |
| `candidate` | Top original-view detection across all 92 outputs, or null. |
| `suggestion` | Candidate only when accepted; otherwise null. |
| `user_confirmation_required` | Always true. No automatic nutrition save. |
| `original_inference_passes`, `brightened_inference_passes` | Actual model-pass counts, not training runs. |
| `processing_ms` | Local decode/queue/inference/policy computation; excludes upload/network, model load/warmup and backend/Mobile. |

A candidate contains `class_id`, `label`, `confidence` and `bbox_xyxy`.
Box coordinates are `[x1,y1,x2,y2]` in pixels of the **EXIF-oriented original
decoded image**, not the internal 640-pixel inference canvas. Overlay the box on
an equivalently oriented image. Confidence is the detector score, not a calibrated
probability that the label is correct.

`/v1/compare` returns shared metadata and `candidate`, plus a `results` array of
three `{policy,status,reason,suggestion}` objects. It has no single overall
acceptance status. The normal app uses `/v1/recognize`; the comparison endpoint
supports a research/demo screen.

## Status, reasons and frontend actions

| Status | Reason | Frontend behavior |
|---|---|---|
| accepted | `accepted` | Show the suggestion; require confirm or edit. |
| uncertain | `below_threshold` | Explain uncertainty; allow retry/manual selection. |
| uncertain | `outside_catalogue` | No accepted suggestion; manual search remains available. |
| uncertain | `no_bright_candidate` | P2 consistency check failed; manual/retry. |
| uncertain | `view_label_disagreement` | P2 views disagree; manual/retry. |
| uncertain | `view_box_disagreement` | P2 box overlap is insufficient; manual/retry. |
| no_detection | `no_candidate` | No detection reached the shared floor; retry/manual. |

Do not interpret no detection as proof that no food is present. Do not treat a raw
`candidate` as an accepted suggestion when `suggestion` is null. The highest
candidate can be outside the catalogue even at a high score; do not search for a
lower-scoring target detection to accept instead.

## Failures

Errors have `{schema_version,request_id,error:{code,message,retryable}}`.
They are not successful `no_detection` results.

| HTTP status | Examples |
|---|---|
| 400 | Empty/invalid image; invalid request ID. |
| 413 | File size or pixel limit exceeded. |
| 415 | Unsupported decoded image format. |
| 422 | Missing file or invalid policy. |
| 503 | Model unavailable or failed readiness verification. |
| 500 | Unexpected service failure; safe message and request ID. |

NestJS owns its outbound HTTP timeout and the frontend-facing timeout/unavailable
mapping. That adapter and end-to-end timeout tests are **not implemented here**.
A backend timeout must not be converted into a fabricated food label or empty
successful recognition. Retry/manual selection remains available.

## NestJS and Flutter acceptance checklist

1. Send multipart `file` and the selected policy to the AI service.
2. Preserve model/policy/schema versions, request ID, status, reason, label,
   confidence and box. Replace the current labels-only adapter.
3. Map an accepted suggestion to food search; allow manual selection or correction.
4. Save only the food/portion explicitly chosen and confirmed by the user.
5. Verify accepted, uncertain, no detection, malformed image, service unavailable,
   timeout, correction, manual search and multiple detector-output scenarios.
6. Measure the real backend/Mobile latency. The main research outcome remains one
   best image-level suggestion; this is not full multi-serving recognition.
