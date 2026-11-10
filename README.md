# HealthPal AI service

This FastAPI service loads the selected 92-output checkpoint and applies the
frozen P0/P1/P2 rules. It is the first integration milestone for HealthPal's
ten-food research catalogue. There is no training in the serving flow.

The source repository includes the frozen model manifest and policy configuration.
**A fresh clone needs the separate model artifact before recognition can run.**
See [installation](#install-on-another-computer) below. In the original HealthPal
workspace, the verified model is already at `models/healthpal-selected-92.pt`.

## Run the service

Open a terminal in this folder and run:

```powershell
python run.py
```

Open **http://127.0.0.1:8001/docs**. Wait for model initialization to finish.

1. Expand **POST /v1/compare** and click **Try it out**.
2. Choose a food photograph and click **Execute**.
3. Read `results`: it contains the P0, P1 and P2 statuses for the same image.
4. Try **POST /v1/recognize** with `policy = P1` for the normal single-policy response.
5. `suggestion` contains an accepted candidate only. `candidate` is raw detector
   evidence and can remain present when the status is uncertain.

**Accepted does not mean verified correct.** The user must confirm or correct the
chosen food before the backend saves a nutrition record. Manual search can still
find foods outside the ten-food suggestion catalogue.

The regular request defaults to P1. The comparison endpoint is for research/demo
use. One original inference is shared; a brightened pass is added only if P1
accepts and P2 is requested. It does not run three separate models.

## Frozen inputs

- Checkpoint: `models/healthpal-selected-92.pt`.
- Model version, checksum and full 92-name list: `models/model-manifest.json`.
- Selection history, measured limits and final counts: `models/model-card.json`.
- Policy configuration: `config/policy-v1.json`.
- Candidate floor: 0.05; P1 threshold: 0.36970099806785583.
- P2: same label, predicted-box IoU at least 0.75, luminance multiplier 1.15.
- Native outputs all compete before the ten-food gate. No target-only filtering
  and no substitution of a lower-scoring target detection.

```powershell
python scripts/verify_release.py
```

The chosen checkpoint is the negative-margin epoch01 candidate. It is not the
latest training trial and did not meet every original experimental goal. The
authentic original has 90 outputs; E0 is an earlier already-trained 92-output
reference. Neither is silently substituted for the selected model.

## API and handover

- [API contract and frontend behavior](docs/api-contract.md)
- [Generated OpenAPI schema](docs/openapi.json)
- [Accepted example](docs/examples/accepted.json)
- [Uncertain example](docs/examples/uncertain.json)
- [Invalid-image example](docs/examples/invalid-image.json)

The example JSON files came from local API smoke checks. They illustrate the
contract and do not establish real-world accuracy or integrated application speed.

The existing NestJS adapter currently returns only `response.data.labels` in
`health-pal-server/src/food-vision/food-vision.service.ts`. The next milestone is
to preserve this service's structured response, map suggested labels to food
search, and handle timeout/unavailability. The current `FOOD_VISION_URL` should
be switched to `/v1/recognize` as part of that adapter change. Flutter then needs
typed response handling and the confirm/correct/manual-selection screens.

No NestJS or Flutter wiring is claimed complete by this repository.

## Tests

On a fresh development environment, install `requirements-dev.txt` first.

```powershell
python -m unittest discover -s tests -v
```

The tests cover accepted, uncertain, no detection, invalid/missing/oversized images,
invalid policies, EXIF orientation, readiness failure, shared inference and the
600 frozen policy decisions. Endpoint unit tests use explicit detector fixtures;
their fabricated scores are not model-quality measurements. Native model/API
smoke checks are recorded separately in `docs/local-verification.json`.

The saved prediction fixtures contain no photographs. They check implementation
parity with an existing result; they are not training data or a new final test.

## Install on another computer

Clone this repository and use Python 3.10 or 3.11 with the tested CPU dependency set:

```powershell
git clone https://github.com/health-pal-uit/healthpal-food-recognization-v1.git
cd healthpal-food-recognization-v1
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

The `.pt` is excluded from ordinary Git commits. Download the selected checkpoint
from the artifact link in `models/model-manifest.json`, then import it:

```powershell
.venv\Scripts\python scripts/import_model.py --source "C:\path\to\candidate.pt"
.venv\Scripts\python scripts/verify_release.py
.venv\Scripts\python run.py
```

The script verifies the checkpoint checksum. A cloned repository needs this model
artifact before readiness can succeed. `HEALTHPAL_MODEL_PATH` can point to a
separately mounted, hash-matching artifact. CPU is the default execution device.
Artifact access is managed separately from access to this source repository.
For Linux/macOS, use `.venv/bin/python` in place of `.venv\Scripts\python`.

## Research scope and provenance

Sources: project-context `context/overview.md`, `context/source-policy.md`, and
`documentation/UIT_HCM_NCKH_SV_KhongHuynhNgocHan.pdf`, pp. 2 and 5. The fixed revised
thesis lists a Python/FastAPI AI service; the inherited fork is Flask-based.
This serving implementation occupies the AI-service role in that architecture.

Original project: https://github.com/lannguyen0910/food-recognition . Its checkpoint
source mapping is in `food-recognition/theseus/utilities/download.py` in the
workspace. A copy of its source-code license is retained in `third_party/` for
attribution; dependencies and model artifacts retain their own applicable terms.

The original final evaluation remains unchanged. Its target-food condition mix
was 70 standard / 3 low-light / 27 angle-or-occlusion rather than the planned
40/30/30. Formal E0/E1/E2 research requirements and the revised thesis's historical
training appendix differ; this API does not resolve that experimental-history
gap. Neither the thesis nor research proposal was edited to create this service.

The service skips the extra P2 pass when P1 already rejects. This preserves policy
decisions. The final benchmark always computed the bright view, so its timing
must not be relabelled as optimized service or backend/Mobile latency. Measure
actual integrated latency separately. No portions or calories are inferred from
the image by this API.
