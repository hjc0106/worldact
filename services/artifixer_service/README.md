# ArtiFixer refinement service (moved)

This directory previously held a duplicate stub HTTP service. The **only**
implementation is now the parent ArtiFixer repo:

```text
ArtiFixer/artifixer_service/
  app.py                 # unified entry: inference + refinement
  refinement/            # WorldAct /v1/refinement-jobs
```

## Start the service

From the ArtiFixer repo root:

```bash
ARTIFIXER_PIPELINE=stub ARTIFIXER_SERVICE_TOKEN=test-token ./run_svc.sh
```

- Health: `GET http://127.0.0.1:8000/healthz`
- Refinement: `POST /v1/refinement-jobs` (Bearer token if configured)
- Inference (flowers): `POST /v1/inference`

## WorldAct client

Use `worldact.stage2_artifixer` against that URL:

```bash
export ARTIFIXER_SERVICE_URL=http://127.0.0.1:8000
export ARTIFIXER_SERVICE_TOKEN=test-token
python scripts/run_artifixer_refinement.py ... --phase all
```

Do not add a second FastAPI server under `worldact/services/`.
