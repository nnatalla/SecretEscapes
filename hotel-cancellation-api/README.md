# Hotel Booking Cancellation API

## Overview

A containerised REST API that predicts hotel booking cancellations using a
RandomForest classifier trained on the Hotel Booking Demand dataset. Built
with FastAPI, scikit-learn, and Docker. Python 3.11.

## Prerequisites

- Python 3.11+
- Docker (for containerised run)
- GNU Make

## Quick Start
```bash
make install      # install Python dependencies
make train        # download data, train model, save to models/
make build        # build Docker image
make run          # start API on localhost:8000
```

Swagger UI: http://localhost:8000/docs  
Health check: http://localhost:8000/health

Example prediction request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hotel": "City Hotel",
    "lead_time": 120,
    "arrival_date_month": "August",
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "reserved_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 95.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "previous_cancellations": 0
  }'
```

Expected response:
```json
{
  "is_cancelled": true,
  "cancellation_probability": 0.7123,
  "model_version": "1.0.0"
}
```

## Project Structure
hotel-cancellation-api/
├── src/
│   ├── features.py      # Feature definitions: column lists, pipeline factory.
│   │                    # Single source of truth shared by training and inference.
│   └── train.py         # Training script. Run via 'make train'. Not in Docker image.
├── app/
│   ├── main.py          # FastAPI app, lifespan, endpoint routing.
│   ├── predictor.py     # Model loading and inference wrapper.
│   └── schemas.py       # Pydantic v2 request/response models.
├── tests/
│   ├── conftest.py      # Shared fixtures and SAMPLE_BOOKING constant.
│   ├── test_api.py      # Endpoint integration tests (mocked predictor).
│   └── test_predictor.py # Predictor unit tests (no disk I/O).
├── models/              # Gitignored. Created by 'make train'.
│   ├── model.joblib     # Serialized sklearn Pipeline.
│   └── metadata.json    # Training metadata and metrics.
├── data/                # Gitignored. CSV downloaded by 'make train'.
├── Dockerfile           # Multi-stage build. Runtime image ~250MB.
└── Makefile             # Project control centre.
## Architecture & Design Decisions

### Training/serving skew prevention

`src/features.py` is the single source of truth for feature definitions.
Both `src/train.py` and `app/predictor.py` import `FEATURE_COLUMNS` from it.
`predictor.py` explicitly selects columns in that order when building the
inference DataFrame. This prevents the most common silent failure mode in
ML systems: model trained on features in one order, served with another.

### Why RandomForestClassifier

Tree-based models are invariant to monotonic feature transformations —
`StandardScaler` on numerical inputs adds no information and is deliberately
absent. `class_weight="balanced"` handles the dataset's 63/37 imbalance
without synthetic oversampling (SMOTE, etc.), which risks leaking information
across the train/test boundary. For production, LightGBM with Optuna
hyperparameter search would be the next iteration.

### Feature selection — what was excluded and why

| Column | Reason excluded |
|---|---|
| `reservation_status` | Directly encodes the target variable. Using it would be a label leak. |
| `reservation_status_date` | Only exists after the booking is resolved — unavailable at prediction time. |

All selected features are available at the moment a booking is made.
`deposit_type` is likely the strongest signal: "No Deposit" bookings carry
zero cancellation cost to the guest, creating a genuine causal mechanism.

### Evaluation metrics

| Metric | Role | Why |
|---|---|---|
| ROC-AUC | Primary | Threshold-independent, unaffected by class imbalance |
| F1 weighted | Secondary | Both false positive and false negative have operational cost |
| Accuracy | Not used | Misleading: a model predicting "never cancel" scores 63% |

### Model artifact in Docker

| Approach | Used here | Notes |
|---|---|---|
| `COPY` at build time | Yes | Simple, reproducible, self-contained. New image per model version. |
| Volume mount | No | Flexible for dev, not suitable for production deployments. |
| Download from S3 at startup | No | Production pattern — see Scaling section. |

`COPY` is appropriate for this scope. The trade-off (image rebuild per model
update) is acceptable at MVP stage and acceptable to discuss in interviews.

### Serialisation — joblib

`joblib` is scikit-learn's recommended serialiser. It uses memory-mapped numpy
arrays internally, making load times faster than `pickle` for large estimators.
`compress=3` reduces file size ~60% with negligible load time penalty.

### In-memory metrics

`_MetricsCollector` uses `threading.Lock` for correctness with uvicorn's thread
pool. It is deliberately simple. Known limitation: counters reset on restart and
are not shared across container replicas. Production replacement: Prometheus
client library or CloudWatch Embedded Metrics Format.

### Single uvicorn worker

The `CMD` in `Dockerfile` runs one uvicorn process. This is intentional:
horizontal scaling is the orchestrator's responsibility (ECS tasks, k8s pods),
not the application's. Running multiple workers inside one container complicates
memory limits, health checking, and graceful shutdown.

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Always 200. Check `status` field for "ok" vs "degraded". |
| `POST` | `/predict` | Cancellation prediction. 422 on bad input, 503 if model not loaded. |
| `GET` | `/metrics` | In-memory request counters and uptime. Resets on restart. |

## Makefile Reference

| Target | Description |
|---|---|
| `make install` | Install Python dependencies |
| `make train` | Download data and train model |
| `make lint` | Run ruff linter |
| `make format` | Auto-format and auto-fix |
| `make format-check` | Check formatting (CI gate, no writes) |
| `make test` | Full test suite with coverage (fails below 80%) |
| `make test-fast` | Tests, stop on first failure |
| `make build` | Build Docker image |
| `make run` | Run containerised API on port 8000 |
| `make run-dev` | Run locally with hot-reload |
| `make clean` | Remove build artefacts |
| `make all` | Full CI pipeline: install→lint→test→build |

## Known Limitations

Documented here rather than discovered in production:

- **Model versioning**: model is baked into the Docker image. Updating the model
  requires a new image build and deployment.
- **Metrics persistence**: counters reset on container restart. Not aggregated
  across replicas.
- **No authentication**: all endpoints are public. Add OAuth2/API key middleware
  before any external exposure.
- **No drift monitoring**: model is static. No alerting if input distribution
  shifts after deployment.
- **Case-sensitive month values**: `arrival_date_month` must be title-case English
  (e.g. "August", not "august"). This matches training data encoding and is
  enforced at the API boundary. Documented rather than silently normalised.
- **Windows compatibility**: `PYTHONPATH=.` in Makefile targets requires Unix/macOS.
  On Windows PowerShell: `$env:PYTHONPATH="."; <command>`.

## Scaling to Production (AWS)

1. **Image registry**: push to Amazon ECR with semantic version tags.
2. **Serving**: deploy on ECS Fargate — one task per replica, ALB for routing
   and health checks. No changes to application code required.
3. **Model artifact**: move from `COPY` to downloading from S3 at container
   start (using boto3 + IAM task role). Enables model updates without image
   rebuilds.
4. **Logging**: uvicorn JSON logs → CloudWatch Logs via awslogs driver.
5. **Metrics**: replace `_MetricsCollector` with CloudWatch Embedded Metrics
   Format or Prometheus + managed Grafana.
6. **Model management**: MLflow on EC2 or SageMaker Experiments for tracking
   training runs, comparing model versions, and managing the model registry.
7. **Higher throughput**: SageMaker Real-Time Endpoints remove the need to
   manage container infrastructure entirely, with built-in autoscaling and
   A/B traffic splitting for model rollouts.