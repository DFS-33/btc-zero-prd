# Passos Magicos — Student Risk Predictor

> MLOps pipeline to predict school dropout risk for students supported by Associacao Passos Magicos, built for the FIAP Pos Tech Datathon.

---

## Overview

### What

This project implements a complete end-to-end MLOps pipeline that predicts whether a student enrolled in the Passos Magicos social program is at risk of **educational setback (defasagem escolar)**. The system covers every stage of the ML lifecycle: data ingestion and preprocessing, feature engineering, model training, REST API serving, containerization, cloud deployment, CI/CD automation, and production observability.

### Why

The Associacao Passos Magicos has been transforming the lives of low-income children and youth in Embu-Guacu, Brazil, since 1992. With over 3,000 student records spanning 2022 to 2024, the program generates rich educational data but lacks automated tools to identify at-risk students before they fall behind. This predictor enables early, targeted intervention — minimizing false negatives (students who are in danger but go undetected) is the primary design goal, which is why **Recall** is the main evaluation metric.

### Who

- **Students and teachers** at Passos Magicos benefit from proactive pedagogical support.
- **FIAP Pos Tech evaluators** assess the MLOps engineering quality, test coverage, and production readiness.
- **ML engineers** maintaining the system use the modular codebase and CI/CD pipeline for iterative improvements.

---

## Architecture

```
+------------------------------------------------------------------+
|                    DATA SOURCES                                  |
|   pede_2022.csv   pede_2023.csv   pede_2024.csv                  |
|   (860 records)   (1,014 records)  (1,156 records)               |
+------------------+-----------------------------------------------+
                   |
                   v
+------------------------------------------------------------------+
|                   ML PIPELINE (src/)                             |
|                                                                  |
|  preprocessing.py -----> feature_engineering.py                  |
|  (merge, normalize        (binary target: Defasagem < 0,         |
|   columns, encode)         normalize, scale)                     |
|          |                         |                             |
|          v                         v                             |
|      training.py ---------> evaluation.py                        |
|  (RandomForest/XGBoost,    (Recall, F1-macro, ROC-AUC,          |
|   5-fold cross-val)         reports)                             |
|          |                                                        |
|          v                                                        |
|   models/model.pkl  (joblib serialization)                       |
+------------------+-----------------------------------------------+
                   |
                   v
+------------------------------------------------------------------+
|               API LAYER (src/api.py)                             |
|                                                                  |
|   FastAPI + Pydantic                                             |
|   POST /predict  ---> returns { risk_score, risk_label }        |
|   GET  /health   ---> returns { status: "ok" }                  |
|   GET  /docs     ---> Swagger UI (auto-generated)               |
|          |                         |                             |
|          v                         v                             |
|   src/monitoring.py         Langfuse Cloud                       |
|   (traces, spans,           (dashboard: latency,                 |
|    error tracking)           errors, prediction volume)          |
+------------------+-----------------------------------------------+
                   |
                   v
+------------------------------------------------------------------+
|               CONTAINER LAYER                                    |
|   Dockerfile (python:3.11-slim)                                  |
|   docker-compose.yml (local dev with env vars)                   |
+------------------+-----------------------------------------------+
                   |
          +--------+--------+
          |                 |
          v                 v
+-----------------+  +---------------------------+
|   LOCAL DEV     |  |   CLOUD PRODUCTION        |
|   localhost:8000|  |   Azure App Service (F1)  |
|                 |  |   via Azure ACR            |
+-----------------+  +----------+----------------+
                               |
                               ^
                   +-----------+-----------+
                   |   CI/CD AUTOMATION    |
                   |   GitHub Actions      |
                   |   .github/workflows/  |
                   |   ci.yml  |  cd.yml   |
                   +-----------+-----------+
```

---

## Tech Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Language | Python | 3.11 | Runtime |
| ML Framework | scikit-learn | >=1.4 | Model training and preprocessing |
| Data Processing | pandas | >=2.1 | Dataset ingestion and transformation |
| Numerical | numpy | >=1.26 | Array operations |
| Model Serialization | joblib | >=1.3 | Optimized serialization for sklearn pipelines |
| API Framework | FastAPI | >=0.110 | Async REST API with Pydantic validation |
| API Server | Uvicorn | >=0.27 | ASGI server |
| Data Validation | Pydantic | >=2.5 | Request/response schema enforcement |
| Observability | Langfuse | >=2.0 | Production monitoring, traces, dashboard |
| Environment | python-dotenv | >=1.0 | Env var management |
| Containerization | Docker | python:3.11-slim | Reproducible packaging |
| Cloud | Azure App Service (F1) | - | API hosting |
| Registry | Azure Container Registry | Basic | Docker image storage |
| CI/CD | GitHub Actions | - | Automated lint, test, build, deploy |
| Linter | ruff | >=0.3 | Fast linting and formatting |
| Testing | pytest + pytest-cov | >=8.0 / >=4.1 | Unit tests with 80% coverage gate |
| Test HTTP Client | httpx | >=0.27 | FastAPI TestClient dependency |
| EDA | Jupyter + matplotlib + seaborn | - | Exploratory data analysis notebooks |

---

## Features

- **Multi-year data ingestion**: Merges heterogeneous CSVs from 2022, 2023, and 2024, normalizing divergent column schemas automatically via `normalize_columns()`.
- **Binary risk target**: Derives a binary label from the `Defasagem` field — students with `Defasagem < 0` are flagged as at risk.
- **Recall-optimized model**: Trained with cross-validation (5-fold) and threshold tuning to maximize Recall, minimizing undetected at-risk students.
- **REST inference API**: FastAPI endpoint at `POST /predict` accepts structured JSON with student indicators and returns a risk score and label.
- **Auto-generated API docs**: Swagger UI available at `/docs` (no extra configuration).
- **Production observability**: Every prediction is traced in Langfuse Cloud, providing real-time dashboards for latency, error rates, and prediction volume.
- **Containerized and reproducible**: Docker image bundles the API, model, and all dependencies. Runs identically in local dev and Azure production.
- **Automated CI/CD**: GitHub Actions runs lint, tests (>=80% coverage), Docker build, and smoke test on every push. CD deploys automatically to Azure on merge to `main`.
- **Modular codebase**: Each concern (preprocessing, feature engineering, training, evaluation, API, monitoring, utilities) lives in its own module for easy testing and maintenance.

---

## Dataset

Three annual research datasets from the Passos Magicos educational program:

| File | Records | Year | Notes |
|------|---------|------|-------|
| `pede_2022.csv` | 860 | 2022 | No IPP column; column names differ (e.g., `Defas`, `Matem`, `Portug`, `Inglês`) |
| `pede_2023.csv` | 1,014 | 2023 | Adds IPP; columns renamed (e.g., `Defasagem`, `Mat`, `Por`, `Ing`) |
| `pede_2024.csv` | 1,156 | 2024 | Adds `Escola` and `Ativo/Inativo` fields |

### Key Features Used by the Model

| Unified Field | Description | Type |
|---------------|-------------|------|
| `inde` | Indice de Desenvolvimento Educacional | numeric |
| `iaa` | Indice de Auto-aprendizagem | numeric |
| `ieg` | Indice de Engajamento | numeric |
| `ips` | Indice Psicossocial | numeric |
| `ipp` | Indice Psicopedagogico (missing in 2022, imputed) | numeric |
| `ipv` | Indice de Ponto de Virada | numeric |
| `ida` | Indice de Desempenho Academico | numeric |
| `ian` | Indice de Adequacao de Nivel | numeric |
| `fase` | Phase/grade level of the student | integer |
| `genero` | Student gender | categorical |
| `instituicao` | Type of school institution | categorical |

### Target Variable

```
target = 1  if Defasagem < 0  (student is behind — at risk)
target = 0  if Defasagem >= 0 (student on track or ahead)
```

### Column Normalization Map (handled by `preprocessing.py`)

| Unified Name | 2022 Column | 2023 Column | 2024 Column |
|--------------|-------------|-------------|-------------|
| `inde` | `INDE 22` | `INDE 2023` | `INDE 2024` |
| `pedra` | `Pedra 22` | `Pedra 2023` | `Pedra 2024` |
| `defasagem` | `Defas` | `Defasagem` | `Defasagem` |
| `mat` | `Matem` | `Mat` | `Mat` |
| `por` | `Portug` | `Por` | `Por` |
| `ing` | `Inglês` | `Ing` | `Ing` |
| `ipp` | (absent — imputed) | `IPP` | `IPP` |

---

## Project Structure

```
btc-zero-prd-claude-code/
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml                    # CI: lint + tests (>=80%) + Docker build
|       +-- cd.yml                    # CD: push to ACR + deploy to App Service
|
+-- data/
|   +-- raw/                          # Original CSVs (symlinked from root during dev)
|   +-- processed/                    # Cleaned data generated by the pipeline
|       +-- .gitkeep
|
+-- models/
|   +-- model.pkl                     # Serialized trained model (joblib)
|   +-- .gitkeep
|
+-- notebooks/
|   +-- 01_eda.ipynb                  # Exploratory data analysis across 3 years
|   +-- 02_feature_analysis.ipynb     # Feature importance and correlation analysis
|   +-- 03_model_experiments.ipynb    # Model selection and hyperparameter tuning
|
+-- src/
|   +-- __init__.py
|   +-- config.py                     # Environment variables, file paths, constants
|   +-- preprocessing.py             # Multi-year merge, normalize_columns(), encoding
|   +-- feature_engineering.py       # Binary target creation, normalization, scaling
|   +-- training.py                  # Cross-validated training, joblib serialization
|   +-- evaluation.py                # Recall, F1, ROC-AUC metrics, classification report
|   +-- api.py                       # FastAPI app: POST /predict, GET /health
|   +-- monitoring.py                # Langfuse SDK wrapper: traces, spans, error tracking
|   +-- utils.py                     # Shared helpers (logging, file I/O, etc.)
|
+-- tests/
|   +-- __init__.py
|   +-- conftest.py                  # Shared fixtures: sample_df, mock_model, test_client
|   +-- test_preprocessing.py
|   +-- test_feature_engineering.py
|   +-- test_training.py
|   +-- test_evaluation.py
|   +-- test_api.py                  # FastAPI TestClient tests for all endpoints
|   +-- test_monitoring.py
|
+-- pede_2022.csv                    # Raw dataset 2022 (860 records)
+-- pede_2023.csv                    # Raw dataset 2023 (1,014 records)
+-- pede_2024.csv                    # Raw dataset 2024 (1,156 records)
+-- Dockerfile                       # python:3.11-slim + uvicorn
+-- docker-compose.yml               # Local dev environment with env vars
+-- requirements.txt                 # Production dependencies
+-- requirements-dev.txt             # Dev/test dependencies (pytest, ruff, jupyter)
+-- pyproject.toml                   # ruff and pytest configuration
+-- .env.example                     # Template for required environment variables
+-- .gitignore
+-- README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for container-based local testing)
- Git

### 1. Clone and Set Up the Environment

```bash
git clone https://github.com/<your-org>/btc-zero-prd-claude-code.git
cd btc-zero-prd-claude-code

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements-dev.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your Langfuse credentials (see Configuration section below).

### 3. Train the Model

The datasets (`pede_2022.csv`, `pede_2023.csv`, `pede_2024.csv`) are already at the project root.

```bash
python -m src.training
```

This produces `models/model.pkl`.

### 4. Run the API Locally (without Docker)

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 5. Run the API Locally (with Docker)

```bash
docker build -t passos-magicos-api:local .

docker run -p 8000:8000 \
  -e LANGFUSE_PUBLIC_KEY=your_public_key \
  -e LANGFUSE_SECRET_KEY=your_secret_key \
  passos-magicos-api:local
```

Or using docker-compose (reads from `.env` automatically):

```bash
docker-compose up --build
```

### 6. Verify the API is Running

```bash
curl http://localhost:8000/health
# -> {"status":"ok"}
```

---

## API Reference

### GET /health

Returns the operational status of the API.

**Response (200 OK):**
```json
{"status": "ok"}
```

---

### POST /predict

Accepts student educational indicators and returns a dropout risk prediction.

**Request body:**
```json
{
  "inde": 7.5,
  "iaa": 8.0,
  "ips": 6.5,
  "ipp": 7.0,
  "ipv": 7.2,
  "fase": 3
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inde` | float | yes | Educational Development Index |
| `iaa` | float | yes | Self-learning Index |
| `ips` | float | yes | Psychosocial Index |
| `ipp` | float | yes | Psychopedagogical Index |
| `ipv` | float | yes | Turning Point Index |
| `fase` | int | yes | Student phase/grade level |

**Response (200 OK):**
```json
{
  "risk_score": 0.73,
  "risk_label": "high"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `risk_score` | float | Probability of dropout risk (0.0 to 1.0) |
| `risk_label` | string | `"high"` if risk_score >= 0.5, `"low"` otherwise |

**cURL example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inde": 7.5, "iaa": 8.0, "ips": 6.5, "ipp": 7.0, "ipv": 7.2, "fase": 3}'
```

**Python example:**
```python
import requests

payload = {
    "inde": 7.5,
    "iaa": 8.0,
    "ips": 6.5,
    "ipp": 7.0,
    "ipv": 7.2,
    "fase": 3,
}
response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
# -> {"risk_score": 0.73, "risk_label": "high"}
```

**Swagger UI:**

Navigate to `http://localhost:8000/docs` for the full interactive documentation with an integrated request builder.

---

## ML Pipeline

### 1. Preprocessing (`src/preprocessing.py`)

- Loads all three annual CSV files independently.
- Applies `normalize_columns()` to unify divergent column names across years (e.g., `Defas` -> `defasagem`, `INDE 22` -> `inde`).
- Fills missing `IPP` values (absent in 2022) with the column median from 2023/2024.
- Encodes categorical variables (`genero`, `instituicao`, `pedra`).
- Concatenates all years into a single unified DataFrame.

### 2. Feature Engineering (`src/feature_engineering.py`)

- Creates the binary target variable: `target = 1` if `Defasagem < 0`, else `target = 0`.
- Selects the feature subset used by the model (INDE, IAA, IPS, IPP, IPV, IDA, IAN, Fase, etc.).
- Applies standard scaling to numerical features.
- Returns `X` (features) and `y` (target) ready for training.

### 3. Training (`src/training.py`)

- Builds a `sklearn.pipeline.Pipeline` combining a `ColumnTransformer` preprocessor and a classifier (default: `RandomForestClassifier`).
- Runs 5-fold cross-validation with `scoring="recall"` as the primary metric.
- Fits the final model on the full training set.
- Serializes the pipeline to `models/model.pkl` using `joblib.dump()`.
- Returns a report with `recall_mean`, `recall_std`, and `f1_macro`.

### 4. Evaluation (`src/evaluation.py`)

- Computes Recall, F1-macro, Precision, and ROC-AUC on a held-out test set.
- Prints a full `classification_report`.
- Supports threshold adjustment to optimize Recall above a configurable minimum (default: 0.75).

### Metric Justification

**Primary metric: Recall**

In this problem, a false negative means a student who is genuinely at risk goes undetected and receives no support. The social cost of this outcome far exceeds the cost of a false positive (providing support to a student who did not strictly need it). Recall minimizes false negatives and is therefore the correct optimization target for this socially impactful use case.

**Secondary metric: F1-macro**

F1-macro balances precision and recall across both classes, providing a single scalar for model comparison without penalizing class imbalance.

---

## Configuration

All configuration is loaded from environment variables. Copy `.env.example` to `.env` and fill in the values before running locally.

### `.env.example`

```dotenv
# Langfuse observability (required for production API)
# Obtain from https://cloud.langfuse.com -> Project Settings
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Model path (optional — defaults to models/model.pkl)
MODEL_PATH=models/model.pkl

# Azure App Service port mapping (set in Azure portal, not locally)
# WEBSITES_PORT=8000
```

### GitHub Secrets (for CI/CD)

Configure the following secrets in your GitHub repository under **Settings > Secrets and variables > Actions**:

| Secret | Description |
|--------|-------------|
| `AZURE_CREDENTIALS` | JSON output from `az ad sp create-for-rbac` |
| `REGISTRY_LOGIN_SERVER` | e.g., `myacr.azurecr.io` |
| `REGISTRY_USERNAME` | ACR username |
| `REGISTRY_PASSWORD` | ACR password |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |

---

## Deploy

### Local Docker

```bash
# Build the image
docker build -t passos-magicos-api:local .

# Run with env vars
docker run -p 8000:8000 \
  -e LANGFUSE_PUBLIC_KEY=pk-lf-... \
  -e LANGFUSE_SECRET_KEY=sk-lf-... \
  passos-magicos-api:local

# Smoke test
curl http://localhost:8000/health
```

### Azure App Service (Manual — Initial Setup)

```bash
# 1. Login to Azure
az login

# 2. Create resource group and ACR
az group create --name rg-datathon --location eastus
az acr create --name myacrname --resource-group rg-datathon --sku Basic

# 3. Build and push image
az acr login --name myacrname
docker build -t myacrname.azurecr.io/api:latest .
docker push myacrname.azurecr.io/api:latest

# 4. Create App Service Plan (F1 free tier)
az appservice plan create \
  --name plan-datathon \
  --resource-group rg-datathon \
  --sku F1 \
  --is-linux

# 5. Create Web App
az webapp create \
  --name app-datathon-api \
  --resource-group rg-datathon \
  --plan plan-datathon \
  --deployment-container-image-name myacrname.azurecr.io/api:latest

# 6. Set port and env vars
az webapp config appsettings set \
  --name app-datathon-api \
  --resource-group rg-datathon \
  --settings \
    WEBSITES_PORT=8000 \
    LANGFUSE_PUBLIC_KEY=pk-lf-... \
    LANGFUSE_SECRET_KEY=sk-lf-...

# 7. Verify
curl https://app-datathon-api.azurewebsites.net/health
```

### Azure App Service (Automated via CI/CD)

After the initial setup above, every `git push origin main` triggers the full CI/CD pipeline automatically.

---

## CI/CD Pipeline

### CI (`ci.yml`) — runs on every push and pull request

```
push / pull_request
      |
      v
  Install dependencies (pip install -r requirements.txt)
      |
      v
  Lint (ruff check src/ tests/)
      |
      v
  Tests (pytest --cov=src --cov-fail-under=80)
      |
      v
  Build Docker image
      |
      v
  Smoke test (docker run + curl /health)
```

### CD (`cd.yml`) — runs on push to `main` after CI passes

```
push to main (CI green)
      |
      v
  Azure login (Service Principal)
      |
      v
  Push image to ACR (myacrname.azurecr.io/api:<sha>)
      |
      v
  Deploy to Azure App Service
      |
      v
  Verify /health endpoint live
```

### Running CI Checks Locally

```bash
# Lint
ruff check src/ tests/

# Tests with coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

# Build Docker
docker build -t api:local .

# Smoke test
docker run -d -p 8000:8000 --name smoke api:local
sleep 5
curl --fail http://localhost:8000/health
docker stop smoke && docker rm smoke
```

---

## Testing

The project targets **>=80% test coverage** enforced by `--cov-fail-under=80` in `pyproject.toml` and the CI pipeline.

### Coverage Strategy

| Module | Target Coverage | Approach |
|--------|----------------|----------|
| `src/preprocessing.py` | 90%+ | Pure transformation functions are fully testable |
| `src/feature_engineering.py` | 90%+ | Pure functions with in-memory DataFrames |
| `src/api.py` | 95%+ | FastAPI TestClient covers all endpoints and error paths |
| `src/utils.py` | 90%+ | Simple helper functions |
| `src/config.py` | 85%+ | Env var loading with defaults |
| `src/monitoring.py` | 75%+ | Mocked Langfuse client |
| `src/training.py` | 70%+ | Interface tested; training mocked with small fixtures |
| `src/evaluation.py` | 75%+ | Metrics computed on synthetic fixtures |

### Running Tests

```bash
# All tests with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Single module
pytest tests/test_api.py -v

# With HTML coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Observability

The API integrates **Langfuse Cloud** for production observability. Every call to `POST /predict` creates a trace with:

- Input features (student indicators)
- Output (risk score and label)
- Latency
- Exception details (if any)

### Accessing the Dashboard

1. Create a free account at `https://cloud.langfuse.com`
2. Create a project and copy the public/secret keys to your `.env` and GitHub Secrets
3. After the first prediction, traces appear in the Langfuse dashboard under **Traces**

### Fallback Behavior

If Langfuse credentials are not configured, the API logs predictions to a local structured JSON file and continues operating normally. No predictions are lost due to a monitoring failure.

---

## Contributing

1. Fork the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Install dev dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Make your changes and ensure all checks pass:
   ```bash
   ruff check src/ tests/
   pytest tests/ --cov=src --cov-fail-under=80
   ```

4. Commit using conventional commit format:
   ```bash
   git commit -m "feat(preprocessing): add support for 2025 schema"
   ```

5. Open a pull request. CI will run automatically on the PR.

### Code Standards

- Linting and formatting: `ruff` (configuration in `pyproject.toml`)
- Docstrings: Google-style for all public functions
- Type hints: required for all function signatures
- Test coverage: maintain >= 80% on `src/`

---

## License

This project is developed for the FIAP Pos Tech Datathon (academic context). The educational data belongs to Associacao Passos Magicos and is used solely for research and demonstration purposes. All student records in the datasets are anonymized.

---

## References

- [Associacao Passos Magicos](https://passosmagicos.org.br/)
- [FIAP Pos Tech](https://postech.fiap.com.br/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Azure App Service for Containers](https://learn.microsoft.com/en-us/azure/app-service/configure-custom-container)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html)
